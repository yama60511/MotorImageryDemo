"""
精度改善実験スクリプト。
LOSO CV で複数の手法・前処理パラメータを網羅的に比較する。
"""
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, "..")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import mne
mne.set_log_level("ERROR")

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from mne.decoding import CSP
import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM, FgMDM

import mi_utils as U

logo = LeaveOneGroupOut()
results = []

MOTOR_CH = ["FC5", "FC6", "T7", "T8", "F3", "F4"]  # 運動野近傍

def run(name, X, y, groups, pipe):
    sc = cross_val_score(pipe, X, y, groups=groups, cv=logo, scoring="accuracy")
    mu, sd = sc.mean(), sc.std()
    results.append((mu, sd, name))
    print(f"{mu:.3f} ±{sd:.3f}  {name}  [{','.join(f'{s:.2f}' for s in sc)}]")

print("=" * 70)
print("条件ごとに LOSO accuracy を比較")
print("=" * 70)

# ------------------------------------------------------------------ #
# 1. ベースライン (全ch, 8-30Hz, CAR)
# ------------------------------------------------------------------ #
print("\n--- 帯域 / チャンネル選択の影響 ---")
for bp, label in [((8, 30), "8-30Hz"), ((8, 13), "μ帯 8-13Hz"), ((13, 30), "β帯 13-30Hz")]:
    for ch_sel, ch_label in [(None, "全14ch"), (MOTOR_CH, "運動野6ch")]:
        ds = U.load_dataset(bandpass=bp)
        X, y, sess = ds.X, ds.y, ds.session
        if ch_sel is not None:
            idx = [ds.ch_names.index(c) for c in ch_sel if c in ds.ch_names]
            X = X[:, idx, :]
        run(f"CSP+LDA | {label} | {ch_label}", X, y, sess,
            Pipeline([("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
                      ("clf", LDA())]))

# ------------------------------------------------------------------ #
# 2. CSP コンポーネント数
# ------------------------------------------------------------------ #
print("\n--- CSP n_components の影響 (8-30Hz, 全ch) ---")
ds = U.load_dataset(bandpass=(8, 30))
X, y, sess = ds.X, ds.y, ds.session
for nc in [2, 4, 6, 8]:
    run(f"CSP(n={nc})+LDA",  X, y, sess,
        Pipeline([("csp", CSP(n_components=nc, reg="ledoit_wolf", log=True)),
                  ("clf", LDA())]))
    run(f"CSP(n={nc})+SVM",  X, y, sess,
        Pipeline([("csp", CSP(n_components=nc, reg="ledoit_wolf", log=True)),
                  ("clf", SVC(kernel="rbf", C=1.0))]))

# ------------------------------------------------------------------ #
# 3. Riemannian (MDM / TangentSpace+LDA)
# ------------------------------------------------------------------ #
print("\n--- Riemannian geometry ---")
for bp_label, bp in [("8-30Hz", (8, 30)), ("μ 8-13Hz", (8, 13))]:
    for ch_sel, ch_label in [(None, "全14ch"), (MOTOR_CH, "運動野6ch")]:
        ds = U.load_dataset(bandpass=bp)
        X2, y2, s2 = ds.X, ds.y, ds.session
        if ch_sel is not None:
            idx = [ds.ch_names.index(c) for c in ch_sel if c in ds.ch_names]
            X2 = X2[:, idx, :]
        run(f"MDM | {bp_label} | {ch_label}", X2, y2, s2,
            Pipeline([("cov", Covariances("lwf")), ("clf", MDM())]))
        run(f"TangentSpace+LDA | {bp_label} | {ch_label}", X2, y2, s2,
            Pipeline([("cov", Covariances("lwf")), ("ts", TangentSpace()),
                      ("clf", LDA())]))
        run(f"FgMDM | {bp_label} | {ch_label}", X2, y2, s2,
            Pipeline([("cov", Covariances("lwf")), ("clf", FgMDM())]))

# ------------------------------------------------------------------ #
# 4. onset_s オフセット（CUE後 0.5 / 1.0s 待ってから切り出す）
# ------------------------------------------------------------------ #
print("\n--- epoch onset オフセットの影響 (全ch, 8-30Hz, TangentSpace+LDA) ---")
import numpy as _np, json
from pathlib import Path
from bci.config import PreprocessingConfig
from bci.preprocessing import Preprocessor
from bci.epoching import extract_trials

DATA_ROOT = U.find_data_dir()
for onset_s in [0.0, 0.5, 1.0]:
    all_X, all_y, all_s = [], [], []
    for si, sd in enumerate(sorted(DATA_ROOT.glob("*/recording.npz"))):
        rec = _np.load(sd, allow_pickle=True)
        raw = rec["raw"].astype(_np.float64)
        ts = rec["timestamps"].astype(_np.float64)
        mt = rec["marker_times"].astype(_np.float64)
        ml = rec["marker_labels"].astype(int)
        fs = float(rec["fs"])
        ch_names = [str(c) for c in rec["channel_order"]]
        mi_s = 4.0 - onset_s  # 残り時間
        meta = sd.parent / "metadata.json"
        if meta.exists():
            mi_s_full = float(json.loads(meta.read_text()).get("mi_s", 4.0))
            mi_s = mi_s_full - onset_s
        pre = Preprocessor(PreprocessingConfig(fs=fs, channel_order=ch_names, bandpass=(8,30)))
        filt = pre.transform(raw)
        markers = list(zip(mt.tolist(), ml.tolist()))
        X_e, y_e = extract_trials(filt, ts, markers, mi_s, fs, onset_s=onset_s)
        if X_e.shape[0] == 0: continue
        all_X.append(X_e); all_y.append(y_e)
        all_s.append(_np.full(len(y_e), si, dtype=int))
    Xa = _np.concatenate(all_X); ya = _np.concatenate(all_y); sa = _np.concatenate(all_s)
    run(f"TS+LDA onset={onset_s}s  shape={Xa.shape}", Xa, ya, sa,
        Pipeline([("cov", Covariances("lwf")), ("ts", TangentSpace()), ("clf", LDA())]))

# ------------------------------------------------------------------ #
# 結果サマリー
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("TOP 10 (LOSO 平均精度)")
print("=" * 70)
for mu, sd, name in sorted(results, reverse=True)[:10]:
    print(f"  {mu:.3f} ±{sd:.3f}  {name}")
