"""EEGNet 改善実験。μ帯+運動野ch+LRスケジューラ+epoch数を試す。"""
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys; sys.path.insert(0, "..")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
import mi_utils as U

SEED = 0; np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logo = LeaveOneGroupOut()
MOTOR_CH = ["FC5", "FC6", "T7", "T8", "F3", "F4"]

class EEGNet(nn.Module):
    def __init__(self, n_ch, n_time, n_classes=2, F1=8, D=2, F2=16, kern_len=64, p_drop=0.5):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len//2), bias=False),
            nn.BatchNorm2d(F1))
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p_drop))
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p_drop))
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_time)
            feat_dim = self.separable(self.spatial(self.temporal(dummy))).numel()
        self.classify = nn.Linear(feat_dim, n_classes)
    def forward(self, x):
        return self.classify(self.separable(self.spatial(self.temporal(x))).flatten(1))

def standardize(tr, *others):
    m = tr.mean(axis=(0,2), keepdims=True); s = tr.std(axis=(0,2), keepdims=True)+1e-7
    return [(a-m)/s for a in (tr,*others)]

def to_t(X, y):
    return (torch.tensor(X[:,None,:,:], dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))

def eval_config(name, Xw, yw, sw, epochs=120, batch=32, lr=1e-3, wd=1e-4, use_sched=True):
    accs = []
    for tr, te in logo.split(Xw, yw, groups=sw):
        Xtr, Xte = standardize(Xw[tr], Xw[te])
        Xtr_t, ytr_t = to_t(Xtr, yw[tr]); Xte_t, yte = to_t(Xte, yw[te])
        n_ch, n_t = Xtr.shape[1], Xtr.shape[2]
        model = EEGNet(n_ch, n_t).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        if use_sched:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.CrossEntropyLoss()
        ds_tr = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
        loader = torch.utils.data.DataLoader(ds_tr, batch_size=batch, shuffle=True)
        for ep in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
            if use_sched: sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte_t.to(device)).argmax(1).cpu().numpy()
        accs.append((pred == yw[te]).mean())
    sc = np.array(accs)
    print(f"{sc.mean():.3f} ±{sc.std():.3f}  {name}  [{','.join(f'{a:.2f}' for a in sc)}]")
    return sc.mean()

results = []
print("=== EEGNet 改善実験 ===")

# ベースライン (前回と同じ)
print("\n--- ベースライン (全14ch, 8-30Hz, 60epoch) ---")
ds = U.load_dataset()
Xw, yw, sw = U.make_windows(ds)
results.append(eval_config("EEGNet baseline (60ep, 全14ch, 8-30Hz)", Xw, yw, sw,
                            epochs=60, use_sched=False))

# epoch 増加 + LR scheduler
print("\n--- epoch / scheduler の影響 (全14ch, 8-30Hz) ---")
for ep in [100, 150, 200]:
    results.append(eval_config(f"EEGNet sched CosineAnnealing {ep}ep 全14ch", Xw, yw, sw,
                                epochs=ep, use_sched=True))

# μ帯フィルタ
print("\n--- 帯域変更 (120ep + scheduler) ---")
for bp, label in [((8,13),"μ 8-13Hz"), ((8,30),"8-30Hz"), ((13,30),"β 13-30Hz")]:
    ds2 = U.load_dataset(bandpass=bp)
    Xw2, yw2, sw2 = U.make_windows(ds2)
    results.append(eval_config(f"EEGNet {label} 全14ch 120ep", Xw2, yw2, sw2,
                                epochs=120, use_sched=True))

# 運動野ch + μ帯
print("\n--- 運動野6ch + μ帯 ---")
ds3 = U.load_dataset(bandpass=(8,13))
Xw3, yw3, sw3 = U.make_windows(ds3)
idx = [ds3.ch_names.index(c) for c in MOTOR_CH if c in ds3.ch_names]
Xw3m = Xw3[:, idx, :]
results.append(eval_config("EEGNet μ 運動野6ch 120ep", Xw3m, yw3, sw3,
                            epochs=120, use_sched=True))

# dropout 調整
print("\n--- Dropout rate (μ 全14ch 120ep) ---")
ds4 = U.load_dataset(bandpass=(8,13))
Xw4, yw4, sw4 = U.make_windows(ds4)
for p in [0.25, 0.5]:
    accs = []
    for tr, te in logo.split(Xw4, yw4, groups=sw4):
        Xtr, Xte = standardize(Xw4[tr], Xw4[te])
        Xtr_t, ytr_t = to_t(Xtr, yw4[tr]); Xte_t, _ = to_t(Xte, yw4[te])
        n_ch, n_t = Xtr.shape[1], Xtr.shape[2]
        model = EEGNet(n_ch, n_t, p_drop=p).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
        lossf = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)
        for ep in range(120):
            model.train()
            for xb, yb in loader:
                opt.zero_grad(); lossf(model(xb.to(device)), yb.to(device)).backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte_t.to(device)).argmax(1).cpu().numpy()
        accs.append((pred == yw4[te]).mean())
    sc = np.array(accs)
    name = f"EEGNet μ 全14ch dropout={p} 120ep"
    print(f"{sc.mean():.3f} ±{sc.std():.3f}  {name}  [{','.join(f'{a:.2f}' for a in sc)}]")
    results.append(sc.mean())

print("\n=== TOP EEGNet 精度 ===")
print(f"  best: {max(results):.3f}")
