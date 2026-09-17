# made by - Karthik
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
N_MODELS = 4
N_FOLDS = 10
EPOCHS = 100
JITTER = 0.005
BATCH = 16
LR = 2e-3
WD = 1e-4
WARMUP_FRAC = 0.03
K_EDGE = 20
SCALES = (8, 16, 32)
TTA = 8
LOG_EVERY = 20
N_CLASSES = 8
N_PRIM = 4
LABEL_SMOOTH = 0.05
PRES_W = 0.5
PRES_THR = 0.5
SMOOTH_K = 1
SHARE_Q = 0.05

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def knn_idx(x, k):
    k = min(k, x.shape[1])
    return torch.cdist(x, x).topk(k, dim=-1, largest=False).indices


def gather_nb(x, idx):
    B, N, C = x.shape
    k = idx.shape[-1]
    out = torch.gather(x, 1, idx.reshape(B, N * k).unsqueeze(-1).expand(-1, -1, C))
    return out.reshape(B, N, k, C)


@torch.no_grad()
def geo_features(pts):
    feats = [pts.norm(dim=-1, keepdim=True)]
    for k in SCALES:
        idx = knn_idx(pts, k)
        k = idx.shape[-1]
        nb = gather_nb(pts, idx)
        cen = nb.mean(2)
        dif = nb - cen.unsqueeze(2)
        cov = dif.transpose(-1, -2) @ dif / k
        ev, evec = torch.linalg.eigh(cov)
        ev = ev.clamp_min(1e-12)
        s = ev.sum(-1, keepdim=True)
        nrm = evec[..., 0]
        dots = (gather_nb(nrm, idx) * nrm.unsqueeze(2)).sum(-1).abs()
        rad = dif.norm(dim=-1).mean(-1, keepdim=True).clamp_min(1e-9)
        off = (pts - cen).norm(dim=-1, keepdim=True) / rad
        noff = ((pts - cen) * nrm).sum(-1, keepdim=True).abs() / rad
        feats += [ev / s, torch.log(s), off, noff, dots.mean(-1, keepdim=True), dots.min(-1, keepdim=True).values,
                  torch.log(rad)]
    return torch.cat(feats, -1)


class EdgeConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(2 * cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))

    def forward(self, x, idx):
        nb = gather_nb(x, idx)
        c = x.unsqueeze(2).expand_as(nb)
        e = torch.cat([c, nb - c], -1).permute(0, 3, 1, 2)
        return self.mlp(e).max(-1).values.permute(0, 2, 1)


class Net(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.e1 = EdgeConv(cin, 64)
        self.e2 = EdgeConv(64, 64)
        self.e3 = EdgeConv(64, 128)
        self.glob = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(
            nn.Conv1d(256 + 512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, N_PRIM + 1, 1),
        )
        self.pres = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256, N_CLASSES))

    def forward(self, xyz, f):
        x1 = self.e1(torch.cat([xyz, f], -1), knn_idx(xyz, K_EDGE))
        x2 = self.e2(x1, knn_idx(x1, K_EDGE))
        x3 = self.e3(x2, knn_idx(x2, K_EDGE))
        cat = torch.cat([x1, x2, x3], -1).permute(0, 2, 1)
        gv = self.glob(cat).max(-1).values
        g = gv.unsqueeze(-1).expand(-1, -1, cat.shape[-1])
        return self.head(torch.cat([cat, g], 1)), self.pres(gv)


def joint(lg):
    pp = F.softmax(lg[:, :N_PRIM], 1)
    pe = torch.sigmoid(lg[:, N_PRIM:])
    return torch.stack([pp * (1 - pe), pp * pe], 2).flatten(1, 2)


def rand_rot(b, gen):
    q, r = torch.linalg.qr(torch.randn(b, 3, 3, generator=gen, device=DEVICE))
    return q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)


@torch.no_grad()
def predict(model, xyz_list, f_list, rots):
    model.eval()
    probs, pres = [], []
    for xyz, f in zip(xyz_list, f_list):
        x = xyz.unsqueeze(0) @ rots
        ff = f.unsqueeze(0).expand(rots.shape[0], -1, -1)
        lg, pl = model(x, ff)
        probs.append(joint(lg).mean(0).cpu())
        pres.append(torch.sigmoid(pl).mean(0).cpu())
    model.train()
    return probs, pres


def decode(prob, pres, xyz, min_share, smooth_k=SMOOTH_K, pres_thr=PRES_THR):
    p = prob
    if smooth_k > 1:
        idx = knn_idx(xyz.detach().cpu().unsqueeze(0), smooth_k)[0]
        p = p[:, idx].mean(-1)
    n = p.shape[1]
    alive = pres >= pres_thr
    alive[p.sum(1).argmax()] = True
    while True:
        pred = p.masked_fill(~alive[:, None], -1.0).argmax(0)
        cnt = torch.bincount(pred, minlength=N_CLASSES)
        small = [c for c in range(N_CLASSES) if 0 < int(cnt[c]) < min_share * n]
        if not small:
            return pred
        alive[min(small, key=lambda c: int(cnt[c]))] = False


def write_sub(test, preds, submission_out):
    submission = pd.DataFrame({
        "task_id": test["task_id"].values,
        "target_json": [json.dumps(p.tolist(), separators=(",", ":")) for p in preds],
    })
    submission.to_csv(submission_out, index=False)


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    train_labels = pd.read_csv(public_dir / "train_labels.csv")
    label_map = dict(zip(train_labels["task_id"], train_labels["target_json"]))
    train_points = np.load(public_dir / "train_points.npz")
    test_points = np.load(public_dir / "test_points.npz")

    tr_xyz = [torch.tensor(train_points[t], dtype=torch.float32, device=DEVICE) for t in train["task_id"]]
    tr_y = [np.asarray(json.loads(label_map[t]), dtype=np.int64) for t in train["task_id"]]
    te_xyz = [torch.tensor(test_points[t], dtype=torch.float32, device=DEVICE) for t in test["task_id"]]

    tr_f = [geo_features(x.unsqueeze(0))[0] for x in tr_xyz]
    te_f = [geo_features(x.unsqueeze(0))[0] for x in te_xyz]
    allf = torch.cat(tr_f, 0)
    mu, sd = allf.mean(0), allf.std(0).clamp_min(1e-6)
    tr_f = [(f - mu) / sd for f in tr_f]
    te_f = [(f - mu) / sd for f in te_f]

    shares = []
    for y in tr_y:
        c = np.bincount(y, minlength=N_CLASSES)
        shares.append(c[c > 0] / len(y))
    min_share = float(np.quantile(np.concatenate(shares), SHARE_Q))
    counts = np.bincount(np.concatenate(tr_y) // 2, minlength=N_PRIM).astype(np.float64)
    w = (counts / counts.sum()).clip(1e-6) ** -0.5
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)
    print("min_share", round(min_share, 5), "prim weights", [round(v, 3) for v in w.tolist()], flush=True)

    rng = np.random.RandomState(SEED)
    folds = np.array_split(rng.permutation(len(train)), N_FOLDS)
    nmax = max(x.shape[0] for x in tr_xyz)
    tta_gen = torch.Generator(device=DEVICE)
    tta_gen.manual_seed(SEED + 1)
    tta_rots = rand_rot(TTA, tta_gen)

    sum_p = [torch.zeros(N_CLASSES, x.shape[0]) for x in te_xyz]
    sum_q = [torch.zeros(N_CLASSES) for _ in te_xyz]
    out = None
    for m in range(N_MODELS):
        va_idx = folds[m]
        fit_idx = np.setdiff1d(np.arange(len(train)), va_idx)
        X, Fe, Y = [], [], []
        for i in fit_idx:
            n_i = tr_xyz[i].shape[0]
            sel = np.concatenate([np.arange(n_i), rng.randint(0, n_i, nmax - n_i)])
            yy = tr_y[i][sel].copy()
            yy[n_i:] = -1
            X.append(tr_xyz[i][sel])
            Fe.append(tr_f[i][sel])
            Y.append(torch.tensor(yy, device=DEVICE))
        X, Fe, Y = torch.stack(X), torch.stack(Fe), torch.stack(Y)
        PG = torch.stack([(Y == c).any(1) for c in range(N_CLASSES)], 1).float()
        YP = torch.where(Y >= 0, Y // 2, Y)
        VALID = Y >= 0
        YE = (Y % 2).float()

        torch.manual_seed(SEED + m)
        model = Net(3 + Fe.shape[-1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        steps_per_epoch = (len(fit_idx) + BATCH - 1) // BATCH
        total = EPOCHS * steps_per_epoch
        warm = max(1, int(total * WARMUP_FRAC))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(SEED + 100 + m)

        va_xyz = [tr_xyz[i] for i in va_idx]
        va_f = [tr_f[i] for i in va_idx]
        va_y = [tr_y[i] for i in va_idx]

        for ep in range(1, EPOCHS + 1):
            order = torch.randperm(len(fit_idx), generator=gen, device=DEVICE)
            tot = 0.0
            for b in range(steps_per_epoch):
                bi = order[b * BATCH:(b + 1) * BATCH]
                xb = X[bi] @ rand_rot(len(bi), gen)
                xb = xb + JITTER * torch.randn(xb.shape, generator=gen, device=DEVICE)
                logits, pl = model(xb, Fe[bi])
                vb = VALID[bi]
                loss = F.cross_entropy(logits[:, :N_PRIM], YP[bi], weight=w, ignore_index=-1, label_smoothing=LABEL_SMOOTH)
                loss = loss + F.binary_cross_entropy_with_logits(logits[:, N_PRIM][vb], YE[bi][vb])
                loss = loss + PRES_W * F.binary_cross_entropy_with_logits(pl, PG[bi])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
            msg = f"model {m} epoch {ep} loss {tot / steps_per_epoch:.4f}"
            if ep % LOG_EVERY == 0 or ep == EPOCHS:
                vp, vq = predict(model, va_xyz, va_f, tta_rots)
                acc = np.mean([(decode(p, q, x, min_share).numpy() == y).mean() for p, q, x, y in zip(vp, vq, va_xyz, va_y)])
                msg += f" val_acc {acc:.4f}"
            print(msg, flush=True)

        if out is None:
            vp, vq = predict(model, va_xyz, va_f, tta_rots)
            out = {"probs": vp, "pres": vq, "xyz": va_xyz, "y": va_y, "min_share": min_share}
        tp, tq = predict(model, te_xyz, te_f, tta_rots)
        for j in range(len(te_xyz)):
            sum_p[j] += tp[j]
            sum_q[j] += tq[j]
        preds = [decode(sum_p[j] / (m + 1), sum_q[j] / (m + 1), te_xyz[j], min_share) for j in range(len(te_xyz))]
        write_sub(test, preds, submission_out)
        print(f"model {m} done, submission written", flush=True)

    return out


if __name__ == "__main__":
    main()
