import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 100        # temperature (same knob as before)
EPOCHS = 40    # passes over the training data
BATCH = 64
LR = 1e-3      # learning rate (size of each nudge)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("using", dev)

# ---------- load data ----------
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.npz")
d = np.load(path)
x = torch.tensor(d["x"]).float().to(dev)            # (N, 12, 8, 8)
slots = torch.tensor(d["slots"]).long().to(dev)     # (N, 5)
cps = torch.tensor(d["cps"]).float().to(dev)        # (N, 5)
masks = torch.tensor(d["masks"]).to(dev)            # (N, 4096) bool
N = len(x)

# ---------- build the answer keys: softmax(cps / T) placed at the slots ----------
probs = F.softmax(cps.clamp(-1000, 1000) / T, dim=1)      # (N, 5)
target = torch.zeros(N, 4096, device=dev)
target.scatter_(1, slots, probs)                           # put probs[i][j] at slot slots[i][j]

# ---------- split: most for training, some held out ----------
g = torch.Generator().manual_seed(0)
perm = torch.randperm(N, generator=g).to(dev)
n_val = 500
val_idx, train_idx = perm[:n_val], perm[n_val:]


# ---------- the model ----------
class Net(nn.Module):
    def __init__(self, ch=64, blocks=2):
        super().__init__()
        self.inp = nn.Conv2d(12, ch, 3, padding=1)
        self.res = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
            ) for _ in range(blocks)
        ])
        self.head = nn.Linear(ch * 64, 4096)

    def forward(self, x):
        x = F.relu(self.inp(x))
        for block in self.res:
            x = F.relu(x + block(x))
        return self.head(x.flatten(1))


def loss_fn(logits, tgt, mask):
    logits = logits.masked_fill(~mask, -1e9)               # illegal slots -> ~0%
    return -(tgt * F.log_softmax(logits, dim=1)).sum(1).mean()


model = Net().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR)


# ---------- measure how we're doing ----------
@torch.no_grad()
def evaluate(idx):
    model.eval()
    logits = model(x[idx])
    loss = loss_fn(logits, target[idx], masks[idx]).item()
    best = logits.masked_fill(~masks[idx], -1e9).argmax(1)       # model's pick
    top1 = (best == slots[idx][:, 0]).float().mean().item()      # same as Stockfish's #1?
    in5 = (best.unsqueeze(1) == slots[idx]).any(1).float().mean().item()  # in Stockfish's top 5?
    model.train()
    return loss, top1, in5


# ---------- train ----------
for epoch in range(1, EPOCHS + 1):
    order = train_idx[torch.randperm(len(train_idx), device=dev)]   # shuffle
    for i in range(0, len(order), BATCH):
        b = order[i:i + BATCH]
        logits = model(x[b])                           # guess
        loss = loss_fn(logits, target[b], masks[b])    # how wrong
        opt.zero_grad()
        loss.backward()                                # which way to nudge each knob
        opt.step()                                     # nudge

    tr_loss, tr_top1, _ = evaluate(train_idx[:500])
    va_loss, va_top1, va_in5 = evaluate(val_idx)
    print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} top1 {tr_top1:.0%} "
          f"| val loss {va_loss:.3f} top1 {va_top1:.0%} in-top5 {va_in5:.0%}")
torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt"))
print("saved model.pt")