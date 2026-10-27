#!/usr/bin/env python
# File Name: train.py
#
# Date Created: Oct 07,2026
#
# Last Modified: Tue Oct 27 18:47:59 2026
#
# Author: samuelo
#
# Description:
#
##################################################################
import torch
import torch.nn as nn
import torchmetrics
import time, secrets, warnings, contextlib

class Initializer:
    @staticmethod
    def use_he_init(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class Trainer:
    def __init__(
            self,
            model,
            train_dataloader,
            valid_dataloader,
            criterion=None,
            optimizer=None,
            metric=None,
            lr_scheduler=None,
            default_lr = 1e-3,
            weight_decay=1e-3,
            checkpoint_path=None,
            device=None,
            num_classes=None,
            amp=True,
            amp_dtype=None,
            compile=False,
            channels_last=False,
            ):
        self.channels_last = channels_last
        if channels_last:
            model = model.to(memory_format=torch.channels_last)

        if device is None:
            device = next(model.parameters()).device
            if device.type == 'cpu': #not explicitly set perhaps
                device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    
        #torch.set_default_device(device)
        self.device = torch.device(device)

        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=default_lr, weight_decay=weight_decay)

        self._task = None
    
        if metric is None:
            self._task = self._get_task(num_classes)
            metric = torchmetrics.Accuracy(task=self._task, num_classes=num_classes)
        self.metric = metric.to(device)
        
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss() if self._get_task(num_classes) =='binary' else nn.CrossEntropyLoss()
        

        # torchmetrics metrics declare higher_is_better; assume True otherwise
        self.higher_is_better = getattr(self.metric, 'higher_is_better', None) is not False

        
        self.criterion = criterion or nn.CrossEntropyLoss()
        


        self.lr_scheduler = lr_scheduler
        # OneCycleLR and similar schedulers must be stepped every batch, not every epoch
        self._step_per_batch = False
        self.checkpoint_path = checkpoint_path or f'{secrets.token_hex(8)}.pt'



        # mixed precision: bf16 on CUDA when supported, else fp16 (needs a GradScaler); off on CPU
        if amp_dtype is None:
            if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
        self.amp_enabled = amp and self.device.type in ('cuda', 'mps')
        self.amp_dtype = amp_dtype
        self.scaler = torch.amp.GradScaler(self.device.type,
                                           enabled=self.amp_enabled and amp_dtype is torch.float16)

        # compile=True for the default mode, or a mode string e.g. 'reduce-overhead', 'max-autotune'.
        # self.model stays uncompiled so state_dict keys don't get an '_orig_mod.' prefix.
        self.compile = compile
        if compile:
            self._fwd = torch.compile(self.model, mode=compile if isinstance(compile, str) else None)
            if not getattr(train_dataloader, 'drop_last', True):
                warnings.warn('compile=True without drop_last=True on the train dataloader: '
                              'the smaller last batch triggers a recompile.')
        else:
            self._fwd = self.model
        self._needs_warmup = bool(compile)

        self.history ={"train_loss":[], "train_metric":[], "valid_metric":[], "learning_rate":[]}


    def _synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elif self.device.type == 'mps':
            torch.mps.synchronize()

    def _to_device(self, X, y):
        X = X.to(self.device, non_blocking=True)
        if self.channels_last and X.dim() == 4:
            X = X.contiguous(memory_format=torch.channels_last)
        return X, y.to(self.device, non_blocking=True)
    
    def _sample(self, dataLoader):
        sample= next(iter(dataLoader))
        return self._to_device(*sample)

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled)

    @torch.inference_mode()
    def _evaluate(self, dataLoader=None):
        if dataLoader is None:
            dataLoader = self.valid_dataloader
        self.model.eval()
        self.metric.reset()
            
        for X,y in dataLoader:
            X, y = self._to_device(X, y)
            with self._autocast():
                y_pred = self._fwd(X)
            self.metric.update(y_pred, y)
        return self.metric.compute()

    def _get_task(self, num_classes=None):
        if self._task is not None:
            return self._task 
        
        if num_classes is None:
            #try to infer from dataset if possible
            try:
                _ds = self.train_dataloader.dataset
                while isinstance(_ds, torch.utils.data.Subset):
                    _ds = _ds.dataset
                num_classes = len(_ds.classes)
            except AttributeError:
                sample_x, _ = self._sample(self.train_dataloader)
                _l = self.model(sample_x).shape
                num_classes = _l[1] if len(_l) > 2 else _l[-1]   ##### This is probably not correct!
        
        self._task = 'binary' if num_classes <= 2 else 'multiclass'
        return self._task
        

    def evaluate(self, dataloader):
        return self._evaluate(dataLoader=dataloader)


    def train_one_cycle(
            self,
            n_epochs=5,
            early_stopping=False,
            patience=10,
            max_lr = 1e-2,
            pct_start = 0.3,
            anneal_strategy='cos'
            ):
        steps_per_epoch = len(self.train_dataloader)
        with self._temporary_lr_schedule():
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr = max_lr, pct_start=pct_start,
                                  anneal_strategy=anneal_strategy, epochs=n_epochs, steps_per_epoch=steps_per_epoch)
            self._step_per_batch = True
            self.train(n_epochs, early_stopping, patience)

    @contextlib.contextmanager
    def _temporary_lr_schedule(self):
        # restores the scheduler and the optimizer hyperparameters (lr, betas, ...) a temporary scheduler changes
        saved_scheduler, saved_step_per_batch = self.lr_scheduler, self._step_per_batch
        saved_hparams = [{k: v for k, v in g.items() if k != 'params'} for g in self.optimizer.param_groups]
        try:
            yield
        finally:
            self.lr_scheduler, self._step_per_batch = saved_scheduler, saved_step_per_batch
            for group, hparams in zip(self.optimizer.param_groups, saved_hparams):
                params = group['params']
                group.clear()
                group.update(hparams, params=params)


    def sync_optimizer(self):
        params = list(self.model.parameters())
        live = {id(p) for p in params}
        removed, reshaped = 0,0

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if id(p) not in live:
                    self.optimizer.state.pop(p,None)
                    removed += 1
            group['params'] = [p for p in group['params'] if id(p) in live]
        known = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        added = [p for p in params if id(p) not in known]
        self.optimizer.param_groups[0]['params'].extend(added)

        for p in params:
            state = self.optimizer.state.get(p,{})
            if any(torch.is_tensor(v) and v.dim() > 0 and v.shape != p.shape for v in state.values()):
                del self.optimizer.state[p]
                reshaped += 1
        
        if removed or added or reshaped:
            warnings.warn(f'optimizer synced with model: {len(added)} params added, {removed} removed, '
                          f'{reshaped} reset after a shape change')


    def train(self,n_epochs=5, early_stopping=False, patience = 10, lr=None):
        self.sync_optimizer()
        # lr overrides the optimizer's learning rate for this and subsequent runs; ignored when a scheduler is set
        if lr is not None:
            if self.lr_scheduler is not None:
                warnings.warn('lr passed to train() is ignored because an lr_scheduler is set.')
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        best_metric = None
        patience_counter = 0
        for epoch in range(n_epochs):
            self.model.train()
            self.metric.reset()
            # accumulate on device to avoid a CPU<->GPU sync every batch
            total_loss = torch.zeros((), device=self.device)
            t0 = time.time()
            for X,y in self.train_dataloader:
                X, y = self._to_device(X, y)
                with self._autocast():
                    y_pred = self._fwd(X)
                    loss = self.criterion(y_pred, y)
                total_loss += loss.detach()
                # scaler is a no-op unless training in fp16
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self._step_per_batch:
                    self.lr_scheduler.step()

                self.metric.update(y_pred.detach(), y)

            self._synchronize()
            t1 = time.time()
            self.history['train_loss'].append(total_loss.item()/len(self.train_dataloader))
            self.history['train_metric'].append(self.metric.compute().item())
            self.history['valid_metric'].append(self._evaluate().item())

            self.history['learning_rate'].append(self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]['lr'])

            valid_metric =self.history['valid_metric'][-1]
            best=''
            improved = (best_metric is None
                        or (valid_metric > best_metric if self.higher_is_better else valid_metric < best_metric))
            if improved:
                best_metric = valid_metric
                patience_counter  = 0
                best='(best)'
                if self.checkpoint_path is not None:
                    torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1

            print(f'{epoch + 1}/{n_epochs}: ',
              f' train loss: {self.history["train_loss"][-1]:.4f} |',
              f' train metric: {self.history["train_metric"][-1]:.4f} |',
              f' valid metric:  {valid_metric:.4f} {best}|',
              f'  in {t1 - t0:.1f}s{" (incl. compile)" if self._needs_warmup else ""}',
              f' lr:{self.history["learning_rate"][-1]:.8f}'
             )

            self._needs_warmup = False

            if early_stopping and (patience_counter >= patience):
                print("Early Stopping!")
                break

            if self.lr_scheduler is not None and not self._step_per_batch:
                self.lr_scheduler.step()
        if self.checkpoint_path is not None:
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device, weights_only=True))
        #return self.history
