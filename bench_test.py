import sys, time, copy, torch, torch.nn as nn, torchmetrics
from train_harness import Trainer
from torch.utils.data import TensorDataset, DataLoader
dev = 'mps'
torch.manual_seed(0)
V, T, N = 2000, 30, 4096
X = torch.randint(1, V, (N, T)); y = torch.randint(1, V, (N, T))
dl = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
vdl = DataLoader(TensorDataset(X[:512], y[:512]), batch_size=32)
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.e = nn.Embedding(V, 256); s.g = nn.GRU(256, 256, 2, batch_first=True); s.o = nn.Linear(256, V)
    def forward(s, x): return s.o(s.g(s.e(x))[0]).permute(0, 2, 1)
base = M().to(dev)
def notebook(model, epochs):
    opt = torch.optim.NAdam(model.parameters()); lf = nn.CrossEntropyLoss(ignore_index=0)
    met = torchmetrics.Accuracy(task='multiclass', num_classes=V).to(dev)
    for _ in range(epochs):
        model.train(); met.reset(); tot = 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev); p = model(xb); l = lf(p, yb); tot += l.item()
            l.backward(); opt.step(); opt.zero_grad(); met.update(p, yb)
        met.compute(); model.eval(); met.reset()
        with torch.no_grad():
            for xb, yb in vdl:
                xb, yb = xb.to(dev), yb.to(dev); met.update(model(xb), yb)
        met.compute().item()
def harness(model, epochs, **kw):
    t = Trainer(model, dl, vdl, criterion=nn.CrossEntropyLoss(ignore_index=0),
                optimizer=torch.optim.NAdam(model.parameters()),
                metric=torchmetrics.Accuracy(task='multiclass', num_classes=V).to(dev), **kw)
    t.train(n_epochs=epochs)
for name, fn in [('notebook', notebook), ('harness default', harness),
                 ('harness amp=False', lambda m, e: harness(m, e, amp=False))]:
    m = copy.deepcopy(base); fn(m, 1)  # warm-up
    torch.mps.synchronize(); t0 = time.time(); fn(m, 2); torch.mps.synchronize()
    print(f'### {name}: {(time.time() - t0) / 2:.2f}s/epoch', flush=True)
