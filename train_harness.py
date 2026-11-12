#!/usr/bin/env python
# File Name: train.py
#
# Date Created: Oct 07,2026
#
# Last Modified: Wed Oct 28 22:59:18 2026
#
# Author: samuelo, claude
#
# Description:
#
##################################################################
import torch
import torch.nn as nn
import torchmetrics
import time, math, secrets, warnings, contextlib, reprlib, copy
from collections.abc import Mapping
try:
    from tqdm.auto import tqdm   # widget bars in Jupyter, text bars in a terminal
except ImportError:
    tqdm = None


class _NoBar:
    # stands in for a tqdm bar when progress bars are off or tqdm isn't installed
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable)
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def update(self, n=1): pass
    def set_postfix(self, *args, **kwargs): pass

class Initializer:
    @staticmethod
    def use_he_init(module: nn.Module):
        """He (Kaiming) uniform init with zero bias for Linear and Conv2d layers; use as model.apply(Initializer.use_he_init).

        module: the layer to initialize; layers of other types are left unchanged.
        """
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
            checkpoint_path='auto',
            device='auto',
            num_classes=None,
            amp=True,
            amp_dtype=None,
            compile=False,
            channels_last=False,
            prepare_batch=None,
            loss_weights=None,
            monitor=None,
            verbose=True,
            progress=True,
            grad_clip=None,
            reset_state=None,
            ):
        """Training loop with mixed precision, checkpointing, early stopping and lr scheduling.

        model: the nn.Module to train. If a batch's inputs are a plain tuple they are passed as
            positional arguments, model(*X); anything else (tensor, dict, list, BatchEncoding) as model(X).
            The model may return one output, or a tuple/list/dict of outputs.
        train_dataloader, valid_dataloader: yield (inputs, targets) batches (see prepare_batch otherwise).
            For multiple outputs, targets have the same tuple/dict structure (None skips an output),
            or are a single tensor shared by every output.
        criterion: the loss; a list/dict gives one loss per output, a single loss is applied to every
            output. Default: BCEWithLogitsLoss for one logit per sample, else CrossEntropyLoss
            (single-output models only).
        optimizer: default AdamW(lr=default_lr, weight_decay=weight_decay).
        metric: a torchmetrics metric; a list/dict gives one per output (None for no metric), a single
            metric is copied for every output. Default: Accuracy for single-output models, none otherwise.
        lr_scheduler: stepped every epoch.
        default_lr, weight_decay: used only for the default optimizer.
        checkpoint_path: where the best model is saved and reloaded after training;
            'auto' picks a random file name, None disables checkpointing.
        device: 'auto' (default; None means the same) keeps a model that is on an accelerator where it is,
            and moves a cpu model to cuda/mps when available, with a warning; 'model' always keeps the
            model where it is; anything else ('cpu', 'mps', torch.device(...)) is used as given.
            A model spread over several devices is never moved by 'auto'/'model'.
        num_classes: used for the default metric; inferred from the model output if not given.
        amp: mixed precision on cuda/mps. amp_dtype: default bf16 where supported, else fp16.
        compile: True or a torch.compile mode string ('reduce-overhead', 'max-autotune').
        channels_last: use channels_last memory format for the model and 4-D input tensors.
        prepare_batch: prepare_batch(batch) -> (inputs, targets), applied to every batch before it is
            moved to the device, e.g. to convert dtypes or split a dict batch.
        loss_weights: per-output weights for the total loss, as a list/dict matching the outputs;
            default 1.0 for each output (a sum). Can be changed later through train(loss_weights=...).
        monitor: what picks the best checkpoint and drives early stopping: 'loss' (the weighted
            validation loss), 'metric' (single output), or the name/index of an output with a metric.
            Default: 'metric' for single-output models, 'loss' for multiple outputs.
        verbose: print a summary of the setup, what is monitored, each epoch's results and which
            checkpoint was restored. Warnings are shown either way. Can be changed later via .verbose.
        progress: show tqdm progress bars, one over the epochs and one for each epoch's batches
            (off when verbose=False or tqdm isn't installed). Can be changed later via .progress.
        grad_clip: max norm for gradient clipping. After every backward pass, if the total L2 norm of
            all the gradients exceeds grad_clip, they are all scaled down to that norm (the direction
            is kept); below it they are left alone. None, 0 or False (default): off. float('inf')
            never clips but still records the norms, to help choose a value. While it is on, the
            mean pre-clip norm and the fraction of clipped steps are recorded per epoch in
            history['train_grad_norm'] and history['train_clipped']. Can be changed later through
            train(grad_clip=...).
        reset_state: for stateful models that carry a hidden state from one batch to the next (e.g. an
            RNN/LSTM trained with truncated backprop through time). True calls model.reset(); a callable
            is called as reset_state(model). It runs at the start of every pass over a dataloader (each
            training epoch, each validation, evaluate()) and around the forward passes the Trainer makes
            to inspect the model's output, so no state leaks between training and validation. None or
            False (default): never. Not needed for models that start from a fresh hidden state on every
            call (nn.LSTM called without a hidden state), which already work as they are.
            The model itself must still:
              - detach the state between batches (self.h = h.detach()), otherwise the second batch fails
                with "Trying to backward through the graph a second time";
              - keep the state in a plain attribute, not a registered buffer, so it stays out of the
                state_dict and the checkpoints;
              - handle a smaller last batch, e.g. reset when x.size(0) changes, or use drop_last=True.
            With compile=True, changing Python attributes inside forward can cause graph breaks/recompiles.
        """
        self.verbose = verbose
        self.progress = progress
        self._set_grad_clip(grad_clip)
        self._set_reset_state(reset_state, model)
        self.channels_last = channels_last
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.prepare_batch = prepare_batch

        if channels_last:
            model = model.to(memory_format=torch.channels_last)

        # a model built on the cpu and one deliberately moved there look the same, so 'auto' moves a cpu
        # model to an accelerator but says so; a model spread over several devices was placed on purpose
        model_devices = {t.device for t in (*model.parameters(), *model.buffers())}
        spread = len(model_devices) > 1
        device_note = ' (spread over ' + ', '.join(sorted(map(str, model_devices))) + ')' if spread else ''
        if device is None or device in ('auto', 'model'):
            placement = device or 'auto'
            if spread:
                device = next(model.parameters()).device   # inputs go to the first layer's device
            else:
                device = model_devices.pop() if model_devices else torch.device('cpu')
                best = self._best_accelerator()
                if placement == 'auto' and device.type == 'cpu' and best != 'cpu':
                    warnings.warn(f"Trainer moved the model from cpu to {best}; pass device='cpu' to train "
                                  "on the cpu, or device='model' to always keep the model where it is.")
                    device = best
                    device_note = ' (moved from cpu)'
        elif spread:
            raise ValueError(f'the model is spread over {sorted(map(str, model_devices))}; moving it all to {device} '
                             "would undo that. Pass device='model' to keep it as placed, or move it yourself first.")

        #torch.set_default_device(device)
        self.device = torch.device(device)

        self.model = model if spread else model.to(self.device)
        optimizer_note = '' if optimizer else ' (default)'
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=default_lr, weight_decay=weight_decay)

        criterion_note = metric_note = ''
        if metric is None or criterion is None:
            y_pred = self._probe()
            if self._heads(y_pred) is not None:
                if criterion is None:
                    raise ValueError("the model returns multiple outputs, so the loss can't be inferred: pass "
                                     'criterion= (one loss for every output, or a list/dict with one per output)')
                if metric is None:
                    metric_note = ' (default for multiple outputs)'
            else:
                task, num_classes = self._get_task(y_pred.shape, num_classes)
                inferred = f' (default, inferred {task} task' + (f' with {num_classes} classes)' if task == 'multiclass' else ')')
                if metric is None:
                    metric = torchmetrics.Accuracy(task=task, num_classes=num_classes)
                    metric_note = inferred
                if criterion is None:
                    criterion = nn.BCEWithLogitsLoss() if task == 'binary' else nn.CrossEntropyLoss()
                    criterion_note = inferred
        self.criterion = criterion
        self.metric = metric
        self.loss_weights = loss_weights
        self.monitor = monitor
        # single vs multiple outputs is settled on the first training/validation forward pass
        self._outputs_ready = False

        self.lr_scheduler = lr_scheduler
        # OneCycleLR and similar schedulers must be stepped every batch, not every epoch
        self._step_per_batch = False
        # 'auto' saves the best model to a random file name; None disables checkpointing
        self.checkpoint_path = f'{secrets.token_hex(8)}.pt' if checkpoint_path == 'auto' else checkpoint_path



        # mixed precision: bf16 on CUDA when supported, else fp16 (needs a GradScaler); off on CPU
        if amp_dtype is None:
            if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
        self.amp_enabled = amp and self.device.type in ('cuda', 'mps')
        amp_note = 'off (amp=False)' if not amp else f'off (not supported on {self.device.type})'
        # autocast covers a single device type, so a model spread over e.g. mps and cpu would mix dtypes
        if self.amp_enabled and len({d.type for d in model_devices}) > 1:
            warnings.warn('mixed precision is disabled: the model is spread over more than one device type.')
            self.amp_enabled = False
            amp_note = 'off (the model is spread over more than one device type)'
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

        # train_/valid_ loss and metric per epoch; with multiple outputs also per output, e.g. 'valid_metric/aux'
        self.history = {}

        if self.verbose:
            # several of these are defaults or guessed from the model's output, so say what was picked
            n_params = sum(p.numel() for p in self.model.parameters())
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            group = self.optimizer.param_groups[0]
            if self.amp_enabled:
                amp_note = 'on, ' + str(amp_dtype).removeprefix('torch.') + (
                    ' with GradScaler' if self.scaler.is_enabled() else '')
            rows = {
                'device': f'{self.device}{device_note}',
                'model': f'{type(self.model).__name__}, {self._count(n_params)} params '
                         f'({self._count(n_trainable)} trainable)',
                'criterion': f'{self._describe_spec(criterion)}{criterion_note}',
                'metric': f'{self._describe_spec(metric)}{metric_note}',
                'optimizer': f'{type(self.optimizer).__name__}{optimizer_note}, lr {group["lr"]:g}'
                             + (f', weight decay {group["weight_decay"]:g}' if group.get('weight_decay') else ''),
                'scheduler': f'{type(lr_scheduler).__name__}, stepped every epoch' if lr_scheduler else 'none',
                'grad clip': self._describe_grad_clip(),
                'reset state': self._describe_reset_state(),
                'amp': amp_note,
                'compile': f'on ({compile if isinstance(compile, str) else "default mode"})' if compile else None,
                'channels_last': 'on' if channels_last else None,
                'checkpoint': f'{self.checkpoint_path} (best weights are restored after training)'
                              if self.checkpoint_path else 'off (the final weights are kept)',
                'data': f'train {self._describe_loader(train_dataloader)} | '
                        f'valid {self._describe_loader(valid_dataloader)}',
            }
            width = max(map(len, rows)) + 2
            print('Trainer setup')
            for name, value in rows.items():
                if value is not None:
                    print(f'  {name + ":":<{width}}{value}')


    @staticmethod
    def _best_accelerator():
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

    def _synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elif self.device.type == 'mps':
            torch.mps.synchronize()

    def _move(self, obj, channels_last=False):
        # moves every tensor in a nested batch (dicts, lists, tuples, namedtuples) to the device;
        # leaves without a .to() (strings, ids kept for debugging) pass through unchanged
        if torch.is_tensor(obj):
            obj = obj.to(self.device, non_blocking=True)
            if channels_last and obj.dim() == 4:
                obj = obj.contiguous(memory_format=torch.channels_last)
            return obj
        if callable(getattr(obj, 'to', None)):   # e.g. a transformers BatchEncoding
            return obj.to(self.device)
        if isinstance(obj, Mapping):
            moved = {k: self._move(v, channels_last) for k, v in obj.items()}
            try:
                return type(obj)(moved)
            except TypeError:
                return moved
        if isinstance(obj, tuple) and hasattr(obj, '_fields'):   # namedtuple
            return type(obj)(*(self._move(v, channels_last) for v in obj))
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move(v, channels_last) for v in obj)
        return obj

    @staticmethod
    def _has_tensor(obj):
        if torch.is_tensor(obj) or callable(getattr(obj, 'to', None)):
            return True
        if isinstance(obj, Mapping):
            return any(Trainer._has_tensor(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(Trainer._has_tensor(v) for v in obj)
        return False

    def _to_device(self, batch):
        if self.prepare_batch is not None:
            batch = self.prepare_batch(batch)
        if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
            raise TypeError(
                f'expected each batch to be an (inputs, targets) pair, got {self._describe(batch)}. '
                'Return (X, y) from your dataset/collate_fn, or pass prepare_batch=... to Trainer to split it, '
                "e.g. prepare_batch=lambda b: ({k: v for k, v in b.items() if k != 'label'}, b['label']) "
                'for a dict batch, or prepare_batch=lambda b: (tuple(b[:-1]), b[-1]) for (x1, x2, ..., y)')
        X, y = batch
        hints = {'inputs': '',
                 'targets': ' (e.g. torch.tensor(labels, dtype=torch.float32) for BCEWithLogitsLoss, '
                            'dtype=torch.long for CrossEntropyLoss)'}
        for name, part in (('inputs', X), ('targets', y)):
            if not self._has_tensor(part):
                raise TypeError(
                    f'batch {name} contain no tensors: got {self._describe(part)}. '
                    f'Return tensors from your dataset/collate_fn{hints[name]}, '
                    'or pass prepare_batch=... to Trainer to convert them.')
        return self._move(X, self.channels_last), self._move(y)

    @staticmethod
    def _describe(obj):
        return f'{type(obj).__name__} {reprlib.repr(obj)}'

    def _sample(self, dataLoader):
        return self._to_device(next(iter(dataLoader)))

    def _forward(self, X, model=None):
        # a plain tuple of inputs is unpacked into positional arguments; everything else is one argument
        model = model or self._fwd
        if isinstance(X, tuple) and not hasattr(X, '_fields'):
            return model(*X)
        return model(X)

    def _probe(self, train_mode=False):
        # one forward pass on a training batch, without gradients or batchnorm updates; train_mode=True
        # shows outputs some models return only while training (e.g. auxiliary classifiers)
        # a stateful model is reset before and after, so the probe neither uses nor leaves a hidden state
        sample_x, _ = self._sample(self.train_dataloader)
        was_training = self.model.training
        saved_buffers = {n: b.clone() for n, b in self.model.named_buffers()}
        self.model.train(train_mode)
        try:
            self._reset_model_state()
            with torch.no_grad():
                y_pred = self._forward(sample_x, self.model)
                for n, b in self.model.named_buffers():
                    b.copy_(saved_buffers[n])
        finally:
            self._reset_model_state()
            self.model.train(was_training)
        return y_pred

    @staticmethod
    def _heads(obj):
        # a model's outputs (or the targets) as {name: value}, or None for a single output
        if isinstance(obj, Mapping):
            return dict(obj)
        if isinstance(obj, tuple) and hasattr(obj, '_fields'):
            return obj._asdict()
        if isinstance(obj, (list, tuple)):
            return dict(enumerate(obj))
        return None

    @staticmethod
    def _count(n):
        for unit, size in (('B', 1e9), ('M', 1e6), ('K', 1e3)):
            if n >= size:
                return f'{n / size:.1f}{unit}'
        return str(n)

    @staticmethod
    def _describe_loader(loader):
        # an iterable dataset may have no length, and a custom batch_sampler no batch_size
        try:
            text = f'{len(loader):,} batches'
        except TypeError:
            text = 'unknown number of batches'
        batch_size = getattr(loader, 'batch_size', None)
        return text + (f' of {batch_size}' if batch_size else '')

    @staticmethod
    def _describe_spec(spec):
        # a short name for a criterion/metric setting, which may be a list/dict with one per output
        if spec is None:
            return 'none'
        if isinstance(spec, Mapping):
            return '{' + ', '.join(f'{k}: {Trainer._describe_spec(v)}' for k, v in spec.items()) + '}'
        if isinstance(spec, (list, tuple)):
            return '[' + ', '.join(Trainer._describe_spec(v) for v in spec) + ']'
        return type(spec).__name__

    @staticmethod
    def _per_head(spec, keys, name, default=None, broadcast=lambda v: v):
        # spreads a per-output setting over the output names: a list/tuple matches the outputs by
        # position, a dict by name, anything else applies to every output
        if spec is None:
            return {k: default for k in keys}
        if isinstance(spec, Mapping):
            given = dict(spec)
        elif isinstance(spec, (list, tuple)):
            given = dict(enumerate(spec))
        else:
            return {k: broadcast(spec) for k in keys}
        extra = [k for k in given if k not in keys]
        if extra:
            raise ValueError(f'{name} has entries for {extra}, which match no model output; the outputs are {keys}')
        return {k: given.get(k, default) for k in keys}

    def _setup_outputs(self, y_pred):
        # runs once, on the first output: decides single vs multiple outputs and checks the per-output
        # settings (criterion, metric, loss_weights, monitor) against the output names
        heads = self._heads(y_pred)
        if heads is None:
            self._out_keys = None
            if any(isinstance(s, (Mapping, list, tuple)) for s in (self.criterion, self.metric)):
                raise ValueError('criterion/metric are given per output, but the model returns a single output')
            if self.loss_weights is not None:
                warnings.warn('loss_weights is ignored: the model returns a single output.')
            self.metric = self.metric.to(self.device)
            self._metrics = {None: self.metric}
            monitor = 'metric' if self.monitor is None else self.monitor
            if monitor not in ('metric', 'loss'):
                raise ValueError(f"monitor must be 'metric' or 'loss' for a single-output model, got {monitor!r}")
        else:
            self._out_keys = list(heads)
            metrics = self._per_head(self.metric, self._out_keys, 'metric', broadcast=copy.deepcopy)
            self._metrics = {k: m.to(self.device) for k, m in metrics.items() if m is not None}
            self._check_loss_config()
            monitor = 'loss' if self.monitor is None else self.monitor
            if monitor != 'loss' and monitor not in self._metrics:
                raise ValueError(f"monitor must be 'loss' or an output with a metric ({list(self._metrics)}), "
                                 f'got {monitor!r}')
        for m in self._metrics.values():
            m.reset()
        self._monitor = monitor
        self._monitor_key = 'loss' if monitor == 'loss' else 'metric' if monitor == 'metric' else f'metric/{monitor}'
        if monitor == 'loss':
            self.higher_is_better = False
        else:
            # torchmetrics metrics declare higher_is_better; assume True otherwise
            m = self._metrics[None if monitor == 'metric' else monitor]
            self.higher_is_better = getattr(m, 'higher_is_better', None) is not False
        self._outputs_ready = True
        if self.verbose:
            if self._out_keys is not None:
                self._print('Outputs: ' + ', '.join(
                    f'{k} (loss weight {self._weights[k]:g}, '
                    f'metric {self._describe_spec(self._metrics.get(k))})' for k in self._out_keys))
            self._print(f'Monitoring: valid {self._monitor_key.replace("/", " ")} '
                  f'({"higher" if self.higher_is_better else "lower"} is better)')

    def _check_loss_config(self):
        criteria = self._per_head(self.criterion, self._out_keys, 'criterion')
        weights = self._per_head(self.loss_weights, self._out_keys, 'loss_weights', default=1.0)
        missing = [k for k in self._out_keys if criteria[k] is None and weights[k] != 0]
        if missing:
            raise ValueError(f'no criterion for outputs {missing}: give one per output or set their loss_weights to 0')
        self._criteria, self._weights = criteria, weights

    def _split(self, y_pred, y):
        # pairs each output with its target
        preds = self._heads(y_pred)
        if preds is None:
            # e.g. GoogLeNet/Inception return their auxiliary outputs only in train mode
            preds = {self._out_keys[0]: y_pred}
        unknown = [k for k in preds if k not in self._out_keys]
        if unknown:
            raise ValueError(f'the model returned outputs {unknown} it did not return before; outputs are {self._out_keys}')
        targets = self._heads(y)
        if targets is None:   # one target shared by every output, e.g. auxiliary classifiers
            return preds, {k: y for k in preds}
        unknown = [k for k in targets if k not in self._out_keys]
        if unknown:
            raise ValueError(f'targets {unknown} match no model output; outputs are {self._out_keys}')
        return preds, targets

    def _check_single(self, y_pred):
        if self._heads(y_pred) is not None:
            raise ValueError('the model returned multiple outputs, but it returned a single output before '
                             '(e.g. auxiliary outputs only in train mode): run train() before evaluate()')

    def _compute_loss(self, y_pred, y):
        # returns the (weighted) total loss and, for multiple outputs, the unweighted loss of each output
        if self._out_keys is None:
            self._check_single(y_pred)
            return self.criterion(y_pred, self._align(y, y_pred)), {}
        preds, targets = self._split(y_pred, y)
        losses = {}
        for k, pred in preds.items():
            if self._weights[k] == 0:
                continue
            if k not in targets:
                raise ValueError(f'no target for output {k!r}: return one, use None to skip that output, '
                                 'or set its loss weight to 0')
            if targets[k] is not None:
                losses[k] = self._criteria[k](pred, self._align(targets[k], pred))
        if not losses:
            raise ValueError('no output has both a target and a non-zero loss weight')
        return sum(self._weights[k] * loss for k, loss in losses.items()), losses

    def _align(self, t, ref=None):
        # for models spread over devices, whose outputs can end up on another device than the inputs:
        # moves t to ref's device (targets to their output), or to the trainer's device (for the metrics)
        device = ref.device if torch.is_tensor(ref) else self.device
        if torch.is_tensor(t) and t.device != device:
            return t.to(device)
        return t

    @staticmethod
    def _metric_target(metric, y):
        # torchmetrics classification metrics expect int targets, but BCEWithLogitsLoss needs float ones
        if (type(metric).__module__.startswith('torchmetrics.classification')
                and torch.is_tensor(y) and y.is_floating_point()):
            return y.long()
        return y

    @torch.no_grad()
    def _update_metrics(self, y_pred, y):
        if self._out_keys is None:
            self.metric.update(self._align(y_pred), self._metric_target(self.metric, y))
            return
        preds, targets = self._split(y_pred, y)
        for k, m in self._metrics.items():
            if k in preds and targets.get(k) is not None:
                m.update(self._align(preds[k]), self._metric_target(m, self._align(targets[k])))

    def _compute_metrics(self):
        if self._out_keys is None:
            return {'metric': self.metric.compute().item()}
        # an output the model only returns in train mode has no validation updates
        return {f'metric/{k}': m.compute().item() for k, m in self._metrics.items()
                if getattr(m, '_update_count', 1) > 0}

    # how often the batch bar shows the running loss; reading it makes the cpu wait for the gpu
    _postfix_every = 25

    def _bars_on(self):
        return self.progress and self.verbose and tqdm is not None

    def _bar(self, iterable=None, **kwargs):
        return tqdm(iterable, dynamic_ncols=True, **kwargs) if self._bars_on() else _NoBar(iterable)

    def _print(self, msg):
        # tqdm.write prints above the progress bars instead of breaking them
        (tqdm.write if tqdm is not None else print)(msg)

    def _set_grad_clip(self, grad_clip):
        # 0 and False mean off, like None; True is rejected rather than read as a max norm of 1
        if grad_clip is True or (grad_clip is not None and grad_clip < 0):
            raise ValueError('grad_clip is the max gradient norm: a positive number, '
                             'or None/0/False to turn clipping off')
        self.grad_clip = grad_clip or None

    def _describe_grad_clip(self):
        if self.grad_clip is None:
            return None
        if math.isinf(self.grad_clip):
            return 'measure only (max norm inf): grad norms are recorded, nothing is clipped'
        return f'max norm {self.grad_clip:g}'

    def _set_reset_state(self, reset_state, model):
        # True means model.reset(); a callable is called with the model; None/False means off
        if reset_state is None or reset_state is False:
            self._reset_fn = None
        elif reset_state is True:
            if not callable(getattr(model, 'reset', None)):
                raise ValueError(f'reset_state=True calls model.reset(), but {type(model).__name__} has no '
                                 'reset() method: add one, or pass a callable, reset_state=lambda m: ...')
            self._reset_fn = lambda m: m.reset()
        elif callable(reset_state):
            self._reset_fn = reset_state
        else:
            raise TypeError(f'reset_state must be None, False, True or a callable, got {self._describe(reset_state)}')
        self._reset_desc = (None if self._reset_fn is None else 'model.reset()' if reset_state is True
                            else getattr(reset_state, '__name__', type(reset_state).__name__))

    def _describe_reset_state(self):
        if self._reset_fn is None:
            return None
        return f'{self._reset_desc}, at the start of every training epoch and validation'

    def _reset_model_state(self):
        # called on self.model, which a compiled self._fwd shares
        if self._reset_fn is not None:
            self._reset_fn(self.model)

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled)

    def _run_epoch(self, dataLoader, training, desc=None):
        # one pass over dataLoader (with optimizer steps when training); returns the mean loss per batch,
        # per-output losses and metrics, e.g. {'loss': ..., 'loss/aux': ..., 'metric/aux': ...}
        # a stateful model starts every pass fresh: a state left by validation was made under
        # inference_mode and can't be used for backward, and one left by training would leak into validation
        self._reset_model_state()
        if self._outputs_ready:
            for m in self._metrics.values():
                m.reset()
        # accumulate on device to avoid a CPU<->GPU sync every batch
        total_loss = torch.zeros((), device=self.device)
        head_losses = {}
        n_batches = 0
        clipping = training and self.grad_clip is not None
        if clipping:
            # pre-clip norm summed over the steps where it was finite (an fp16 overflow gives inf,
            # and GradScaler skips that step), and the number of steps that were clipped
            grad_norm_sum = torch.zeros((), device=self.device)
            n_finite = torch.zeros((), device=self.device)
            n_clipped = torch.zeros((), device=self.device)
        with self._bar(dataLoader, desc=desc, unit='batch', leave=False) as bar:
            for batch in bar:
                X, y = self._to_device(batch)
                with self._autocast():
                    y_pred = self._forward(X)
                    if not self._outputs_ready:
                        self._setup_outputs(y_pred)
                    loss, losses = self._compute_loss(y_pred, y)
                if training:
                    # scaler is a no-op unless training in fp16
                    self.scaler.scale(loss).backward()
                    if clipping:
                        # clip the real gradients, not the fp16-scaled ones; a no-op when the scaler is off
                        self.scaler.unscale_(self.optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        norm = norm.to(self.device)
                        finite = norm.isfinite()
                        grad_norm_sum += torch.where(finite, norm, 0)
                        n_finite += finite
                        n_clipped += finite & (norm > self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self._step_per_batch:
                        self.lr_scheduler.step()

                total_loss += loss.detach()
                for k, head_loss in losses.items():
                    head_losses[k] = head_losses.get(k, 0) + head_loss.detach()
                self._update_metrics(y_pred, y)
                n_batches += 1
                if self._bars_on() and n_batches % self._postfix_every == 0:
                    bar.set_postfix(loss=f'{total_loss.item() / n_batches:.4f}')
        if n_batches == 0:
            raise ValueError('the dataloader yielded no batches')

        results = {'loss': total_loss.item() / n_batches}
        results.update({f'loss/{k}': head_loss.item() / n_batches for k, head_loss in head_losses.items()})
        results.update(self._compute_metrics())
        if clipping:
            results['grad_norm'] = (grad_norm_sum / n_finite).item() if n_finite.item() else float('nan')
            results['clipped'] = n_clipped.item() / n_batches
        return results

    @torch.inference_mode()
    def _evaluate(self, dataLoader=None):
        if dataLoader is None:
            dataLoader = self.valid_dataloader
        self.model.eval()
        return self._run_epoch(dataLoader, training=False, desc='valid')

    def _get_task(self, out_shape, num_classes=None):
        # the task follows from the model's output, not the class count: one logit per sample is binary
        # (BCEWithLogitsLoss), C logits along dim 1 is multiclass (CrossEntropyLoss), even when C == 2
        n_logits = 1 if len(out_shape) == 1 else out_shape[1]

        if n_logits == 1:
            return 'binary', num_classes

        if num_classes is None:
            try:
                _ds = self.train_dataloader.dataset
                while isinstance(_ds, torch.utils.data.Subset):
                    _ds = _ds.dataset
                num_classes = len(_ds.classes)
            except AttributeError:
                num_classes = n_logits
        if num_classes != n_logits:
            warnings.warn(f'model outputs {n_logits} logits but there are {num_classes} classes; '
                          f'using {n_logits}.')
            num_classes = n_logits

        return 'multiclass', num_classes


    def evaluate(self, dataloader):
        """Evaluate the model on a dataloader without training.

        dataloader: yields batches in the same form as the training data.
        Returns the metric value for a single-output model; for multiple outputs, a dict with the
        weighted 'loss' and each output's 'loss/<name>' and 'metric/<name>'.
        """
        if not self._outputs_ready:
            # set up from a train-mode pass, so outputs returned only while training are known
            self._setup_outputs(self._probe(train_mode=True))
        results = self._evaluate(dataLoader=dataloader)
        return results['metric'] if self._out_keys is None else results


    def train_one_cycle(
            self,
            n_epochs=5,
            early_stopping=False,
            patience=10,
            max_lr = 1e-2,
            pct_start = 0.3,
            anneal_strategy='cos',
            loss_weights=None,
            grad_clip=None,
            ):
        """Train with a one-cycle lr schedule; the optimizer's lr and momentum settings are restored afterwards.

        n_epochs, early_stopping, patience, loss_weights, grad_clip: as for train().
        max_lr: peak learning rate.
        pct_start: fraction of the steps spent increasing the learning rate.
        anneal_strategy: 'cos' or 'linear'.
        """
        steps_per_epoch = len(self.train_dataloader)
        with self._temporary_lr_schedule():
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr = max_lr, pct_start=pct_start,
                                  anneal_strategy=anneal_strategy, epochs=n_epochs, steps_per_epoch=steps_per_epoch)
            self._step_per_batch = True
            if self.verbose:
                print(f'Scheduler: OneCycleLR for this run, max lr {max_lr:g}, stepped every batch')
            self.train(n_epochs, early_stopping, patience, loss_weights=loss_weights, grad_clip=grad_clip)

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
        """Make the optimizer match the model's current parameters; train() calls this automatically.

        Parameters no longer in the model are dropped, new ones are added to the first param group,
        and optimizer state is reset for parameters whose shape changed. Takes no arguments.
        """
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


    def train(self,n_epochs=5, early_stopping=False, patience = 10, lr=None, loss_weights=None, grad_clip=None):
        """Train the model, validating after every epoch and reloading the best checkpoint at the end.

        n_epochs: number of epochs.
        early_stopping: stop when the monitored value hasn't improved for `patience` epochs.
        patience: epochs without improvement before stopping.
        lr: new learning rate for this and later runs; ignored when an lr_scheduler is set.
        loss_weights: new per-output loss weights (list/dict) for this and later runs.
        grad_clip: new max gradient norm for this and later runs (see Trainer); 0 or False turns
            clipping off, None (default) keeps the current setting.
        """
        self.sync_optimizer()
        if grad_clip is not None:
            self._set_grad_clip(grad_clip)
        # lr overrides the optimizer's learning rate for this and subsequent runs; ignored when a scheduler is set
        if lr is not None:
            if self.lr_scheduler is not None:
                warnings.warn('lr passed to train() is ignored because an lr_scheduler is set.')
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        if loss_weights is not None:
            self.loss_weights = loss_weights
            if self._outputs_ready:
                if self._out_keys is None:
                    warnings.warn('loss_weights is ignored: the model returns a single output.')
                else:
                    self._check_loss_config()
        best_value = best_epoch = None
        patience_counter = 0
        # the epochs bar stays when done (e.g. 7/20 after early stopping); the batch bars are removed
        with self._bar(total=n_epochs, desc='training', unit='epoch') as epochs_bar:
            for epoch in range(n_epochs):
                self.model.train()
                t0 = time.time()
                train_results = self._run_epoch(self.train_dataloader, training=True,
                                                desc=f'epoch {epoch + 1}')
                self._synchronize()
                t1 = time.time()
                valid_results = self._evaluate()
                t2 = time.time()   # the metrics' .item() calls in _evaluate have already synchronized
                for split, results in (('train', train_results), ('valid', valid_results)):
                    for k, v in results.items():
                        self.history.setdefault(f'{split}_{k}', []).append(v)
                self.history.setdefault('learning_rate', []).append(
                    self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]['lr'])

                value = valid_results[self._monitor_key]
                improved = (best_value is None
                            or (value > best_value if self.higher_is_better else value < best_value))
                if improved:
                    best_value, best_epoch = value, epoch + 1
                    patience_counter  = 0
                    if self.checkpoint_path is not None:
                        torch.save(self.model.state_dict(), self.checkpoint_path)
                else:
                    patience_counter += 1

                # per-output losses are in self.history; the printout shows the total loss and the metrics
                parts = [f'{split} {k.replace("/", " ")}: {v:.4f}' + (' (best)' if improved and split == 'valid'
                                                                        and k == self._monitor_key else '')
                         for split, results in (('train', train_results), ('valid', valid_results))
                         for k, v in results.items()
                         if not k.startswith('loss/') and k not in ('grad_norm', 'clipped')]
                if 'grad_norm' in train_results:
                    parts.append(f'grad norm: {train_results["grad_norm"]:.3g}'
                                 + (f' (clipped {train_results["clipped"]:.0%})'
                                    if not math.isinf(self.grad_clip) else ''))
                if self.verbose:
                    self._print(f'{epoch + 1}/{n_epochs}:  ' + ' | '.join(parts)
                                + f' | in {t1 - t0:.1f}s{" (incl. compile)" if self._needs_warmup else ""}'
                                + f' + {t2 - t1:.1f}s valid | lr: {self.history["learning_rate"][-1]:.8f}')
                epochs_bar.update()
                epochs_bar.set_postfix({f'best valid {self._monitor_key.replace("/", " ")}': f'{best_value:.4f}'})

                self._needs_warmup = False

                if early_stopping and (patience_counter >= patience):
                    if self.verbose:
                        self._print(f'Early stopping: no improvement for {patience} epoch{"s" if patience != 1 else ""}')
                    break

                if self.lr_scheduler is not None and not self._step_per_batch:
                    self.lr_scheduler.step()
        if self.checkpoint_path is not None and best_value is not None:
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device, weights_only=True))
            if self.verbose:
                print(f'Restored the best weights, from epoch {best_epoch} '
                      f'(valid {self._monitor_key.replace("/", " ")}: {best_value:.4f})')
        #return self.history
