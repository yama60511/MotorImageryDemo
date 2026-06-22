"""実験第2ラウンド: FBCSP, アンサンブル, 改良EEGNet"""
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys; sys.path.insert(0, "..")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import mne; mne.set_log_level("ERROR")

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_val_predict
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM, FgMDM
import mi_utils as U

logo = LeaveOneGroupOut()
results = []
MOTOR_CH = ["FC5", "FC6", "T7", "T8", "F3", "F4"]

def run(name, X, y, groups, pipe):
    sc = cross_val_score(pipe, X, y, groups=groups, cv=logo, scoring="accuracy")
    mu, sd = sc.mean(), sc.std()
    results.append((mu, sd, name))
    print(f"{mu:.3f} ±{sd:.3f}  {name}  [{','.join(f'{s:.2f}' for s in sc)}]")
    return mu

# ------------------------------------------------------------------ #
# 1. Filter Bank CSP (μ + β を個別にCSP → 特徴を連結)
# ------------------------------------------------------------------ #
print("\n--- Filter Bank CSP ---")

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import butter, sosfilt, sosfilt_zi

class FilterBankCSP(BaseEstimator, TransformerMixin):
    """各帯域で独立に CSP を学習し特徴を連結する。"""
    def __init__(self, bands=((8,13),(13,30)), n_components=4, fs=128.0):
        self.bands = bands; self.n_components = n_components; self.fs = fs
    def _bandpass(self, X, low, high):
        sos = butter(4, [low, high], btype='band', fs=self.fs, output='sos')
        return sosfilt(sos, X, axis=-1)
    def fit(self, X, y):
        self.csps_ = []
        for lo, hi in self.bands:
            Xf = self._bandpass(X, lo, hi)
            csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True)
            csp.fit(Xf, y); self.csps_.append(csp)
        return self
    def transform(self, X):
        feats = []
        for (lo, hi), csp in zip(self.bands, self.csps_):
            Xf = self._bandpass(X, lo, hi)
            feats.append(csp.transform(Xf))
        return np.concatenate(feats, axis=1)

for ch_sel, ch_label in [(None, "全14ch"), (MOTOR_CH, "運動野6ch")]:
    ds = U.load_dataset(bandpass=(4, 40))  # 広め帯域で読んでFBで分割
    X, y, sess = ds.X, ds.y, ds.session
    if ch_sel:
        idx = [ds.ch_names.index(c) for c in ch_sel if c in ds.ch_names]
        X = X[:, idx, :]
    run(f"FBCSP(μ+β)+LDA | {ch_label}", X, y, sess,
        Pipeline([("fb", FilterBankCSP(bands=((8,13),(13,30)), n_components=4, fs=ds.fs)),
                  ("clf", LDA())]))
    run(f"FBCSP(μ+β)+LR | {ch_label}", X, y, sess,
        Pipeline([("fb", FilterBankCSP(bands=((8,13),(13,30)), n_components=4, fs=ds.fs)),
                  ("sc", StandardScaler()),
                  ("clf", LogisticRegression(C=1.0, max_iter=1000))]))

# 3 帯域版
for ch_sel, ch_label in [(MOTOR_CH, "運動野6ch")]:
    ds = U.load_dataset(bandpass=(4, 40))
    X, y, sess = ds.X, ds.y, ds.session
    idx = [ds.ch_names.index(c) for c in ch_sel if c in ds.ch_names]
    X = X[:, idx, :]
    run(f"FBCSP(α+μ+β)+LDA | {ch_label}", X, y, sess,
        Pipeline([("fb", FilterBankCSP(bands=((4,8),(8,13),(13,30)), n_components=4, fs=ds.fs)),
                  ("clf", LDA())]))

# ------------------------------------------------------------------ #
# 2. Filter Bank Riemannian (各帯域の共分散行列を連結 TS+LDA)
# ------------------------------------------------------------------ #
print("\n--- Filter Bank Riemannian ---")

from scipy.signal import butter, sosfilt

class FilterBankCovariances(BaseEstimator, TransformerMixin):
    """各帯域でバンドパスして共分散行列を推定し、タンジェント特徴を連結。"""
    def __init__(self, bands=((8,13),(13,30)), fs=128.0, estimator='lwf'):
        self.bands = bands; self.fs = fs; self.estimator = estimator
    def _bandpass(self, X, lo, hi):
        sos = butter(4, [lo, hi], btype='band', fs=self.fs, output='sos')
        return sosfilt(sos, X, axis=-1)
    def fit(self, X, y):
        self.ts_list_ = []
        self.cov_list_ = []
        for lo, hi in self.bands:
            Xf = self._bandpass(X, lo, hi)
            cov = Covariances(self.estimator)
            Xc = cov.fit_transform(Xf)
            ts = TangentSpace(); ts.fit(Xc)
            self.cov_list_.append(cov); self.ts_list_.append(ts)
        return self
    def transform(self, X):
        feats = []
        for (lo, hi), cov, ts in zip(self.bands, self.cov_list_, self.ts_list_):
            Xf = self._bandpass(X, lo, hi)
            Xc = cov.transform(Xf)
            feats.append(ts.transform(Xc))
        return np.concatenate(feats, axis=1)

for ch_sel, ch_label in [(MOTOR_CH, "運動野6ch"), (None, "全14ch")]:
    ds = U.load_dataset(bandpass=(4, 40))
    X, y, sess = ds.X, ds.y, ds.session
    if ch_sel:
        idx = [ds.ch_names.index(c) for c in ch_sel if c in ds.ch_names]
        X = X[:, idx, :]
    run(f"FB-Riemann(μ+β)+LDA | {ch_label}", X, y, sess,
        Pipeline([("fb", FilterBankCovariances(bands=((8,13),(13,30)), fs=ds.fs)),
                  ("clf", LDA())]))
    run(f"FB-Riemann(μ+β)+LR | {ch_label}", X, y, sess,
        Pipeline([("fb", FilterBankCovariances(bands=((8,13),(13,30)), fs=ds.fs)),
                  ("sc", StandardScaler()),
                  ("clf", LogisticRegression(C=1.0, max_iter=1000))]))

# ------------------------------------------------------------------ #
# 3. アンサンブル: CSP(μ)+LDA と TS(μ)+LDA の soft voting
# ------------------------------------------------------------------ #
print("\n--- アンサンブル (soft voting) ---")

class ProbaPipe(BaseEstimator):
    """パイプラインをラップして predict_proba を使えるようにする。"""
    def __init__(self, pipe): self.pipe = pipe
    def fit(self, X, y): self.pipe.fit(X, y); return self
    def predict(self, X): return self.pipe.predict(X)
    def predict_proba(self, X): return self.pipe.predict_proba(X)

# μ帯 Riemann が既にベスト。さらに上積みできるか試す
for ch_sel, ch_label in [(MOTOR_CH, "運動野6ch")]:
    ds_mu = U.load_dataset(bandpass=(8, 13))
    ds_b  = U.load_dataset(bandpass=(13, 30))
    idx = [ds_mu.ch_names.index(c) for c in ch_sel if c in ds_mu.ch_names]
    Xmu, y, sess = ds_mu.X[:, idx, :], ds_mu.y, ds_mu.session
    Xb  = ds_b.X[:, idx, :]

    # fold ごとに組み合わせるため、同じ X を使う combined 特徴空間で試す
    # 簡易アンサンブル: FB-Riemann が実質 ensemble 相当なので比較
    # ここでは確率的ソフト投票を手動で LOSO で評価
    p_fgmdm  = cross_val_predict(
        Pipeline([("cov", Covariances("lwf")), ("clf", FgMDM())]),
        Xmu, y, groups=sess, cv=logo, method="predict_proba")
    p_ts_lda = cross_val_predict(
        Pipeline([("cov", Covariances("lwf")), ("ts", TangentSpace()),
                  ("clf", LDA())]),
        Xmu, y, groups=sess, cv=logo, method="predict_proba")
    p_csp = cross_val_predict(
        Pipeline([("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
                  ("clf", LDA())]),
        Xmu, y, groups=sess, cv=logo, method="predict_proba")

    for w_sets, w_label in [
        ((1,1,0), "FgMDM+TS"),
        ((1,0,1), "FgMDM+CSP"),
        ((1,1,1), "FgMDM+TS+CSP"),
    ]:
        p_ens = sum(w * p for w, p in zip(w_sets, [p_fgmdm, p_ts_lda, p_csp]))
        pred = p_ens.argmax(1)
        acc = (pred == y).mean()
        results.append((acc, 0.0, f"Ensemble({w_label}) μ | {ch_label}"))
        print(f"{acc:.3f}         Ensemble({w_label}) μ | {ch_label}")

# ------------------------------------------------------------------ #
# 結果サマリー
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("TOP 10")
print("=" * 70)
for mu, sd, name in sorted(results, reverse=True)[:10]:
    print(f"  {mu:.3f} ±{sd:.3f}  {name}")
