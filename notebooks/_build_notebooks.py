"""体験授業用 notebook を nbformat で生成するビルダー。

このスクリプト自体は成果物ではなく、notebook (.ipynb) を組み立てるための
ものです。生成後に削除して構いません。
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


# =====================================================================
# Notebook 1: CSP + 古典的機械学習
# =====================================================================
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    md(r"""
# 脳波デコーディング体験 ① — CSP + 機械学習（古典手法）

このノートブックでは、**運動想起（Motor Imagery, MI）** の脳波から
「左手を動かす想像」か「右手を動かす想像」かを当てる **デコーダ** を、
古典的な機械学習で作ります。

## 運動想起と脳波
- 手を動かす（または動かす想像をする）と、反対側の **感覚運動野** で
  μ波（8–13Hz）・β波（13–30Hz）の **パワーが下がる** 現象が起きます
  （**ERD: 事象関連脱同期**）。
- 左手の想像 → 右半球で ERD、右手の想像 → 左半球で ERD。
  この **左右差** を機械学習で読み取るのがデコーダの基本アイデアです。
- 古典 BCI では C3 / C4（左右の手の運動野直上）を使いますが、今回の
  Emotiv EPOC X キャップには C3/C4 電極が無いため、感覚運動野に近い
  **FC5 / FC6 / T7 / T8** あたりが手がかりになります。

## このノートの流れ
1. 記録データの読み込み
2. 生波形と前処理（バンドパス + ノッチ + CAR）の確認
3. クラスごとのパワースペクトル（ERD の可視化）
4. **CSP（Common Spatial Patterns）** で左右差を強調する空間フィルタを学習
5. CSP + 分類器（LDA / SVM）で交差検証
6. まとめと演習

> 💡 ここで使う前処理・エポック切り出しは、実際のリアルタイム BCI
> （`bci/` パッケージ）と **同じコード** を呼び出しています。
"""),
    code(r"""
# 必要なライブラリ
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

import mne
from mne.decoding import CSP
mne.set_log_level("WARNING")

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import FgMDM

# 共有ユーティリティ（実 BCI と同じ前処理・エポック化を呼ぶ）
import mi_utils as U

U.setup_japanese_font()  # グラフの日本語が豆腐(□)にならないように
plt.rcParams["figure.figsize"] = (9, 4)
np.random.seed(0)
print("準備完了")
"""),
    md(r"""
## 1. データの読み込み

5 セッション分の記録を読み込みます。各セッションは「左想像 10 回 +
右想像 10 回」。`load_dataset()` の中で、セッションごとに
**ノッチ(50Hz) → バンドパス(8–30Hz) → CAR（共通平均基準）** をかけ、
MI 区間（4 秒）を 1 トライアルとして切り出しています。
"""),
    code(r"""
ds = U.load_dataset()
print(ds.summary())

# X: (トライアル数, チャンネル数, サンプル数), y: 0=left / 1=right
X, y, fs, ch_names = ds.X, ds.y, ds.fs, ds.ch_names
print("\nX shape:", X.shape, "| y shape:", y.shape, "| fs:", fs, "Hz")
"""),
    md(r"""
## 2. 前処理済み波形を見る

運動野近傍のチャンネルで、1 トライアル分の波形を表示します。
バンドパス済みなので μ/β 帯の振動が残り、ゆっくりしたドリフトや
電源ノイズは除去されています。
"""),
    code(r"""
motor_ch = [c for c in ["FC5", "FC6", "T7", "T8", "F3", "F4"] if c in ch_names]
idx = [ch_names.index(c) for c in motor_ch]

trial = 0
t = np.arange(X.shape[2]) / fs
fig, ax = plt.subplots()
offset = 0
for c, i in zip(motor_ch, idx):
    ax.plot(t, X[trial, i] + offset, label=c, lw=0.8)
    offset += 60  # 見やすいように縦にずらす
ax.set(xlabel="時間 [s]", yticks=[], title=f"前処理済み EEG（trial {trial}, クラス={U.CLASS_NAMES[y[trial]]}）")
ax.legend(loc="upper right", ncol=len(motor_ch), fontsize=8)
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 3. クラスごとのパワースペクトル（ERD を探す）

左想像と右想像で、各チャンネルの **8–30Hz 帯のパワー** に差が出るはず
です。Welch 法でパワースペクトル密度（PSD）を推定し、クラス平均を
重ねて表示します。左右で差が大きいチャンネルほどデコードの手がかりに
なります。
"""),
    code(r"""
def band_psd(sig, fs):
    f, p = welch(sig, fs=fs, nperseg=min(256, sig.shape[-1]))
    return f, p

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
for ax, c in zip(axes.ravel(), motor_ch):
    ci = ch_names.index(c)
    for label, color in [(U.LABEL_LEFT, "tab:blue"), (U.LABEL_RIGHT, "tab:red")]:
        psd = np.array([band_psd(X[k, ci], fs)[1] for k in np.where(y == label)[0]])
        f = band_psd(X[0, ci], fs)[0]
        m = psd.mean(0)
        ax.plot(f, m, color=color, label=U.CLASS_NAMES[label])
    ax.set(title=c, xlim=(4, 35))
    ax.legend(fontsize=8)
axes[1, 0].set_xlabel("周波数 [Hz]"); axes[0, 0].set_ylabel("PSD")
fig.suptitle("クラス平均パワースペクトル（左 vs 右）")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 4. CSP（Common Spatial Patterns）

CSP は「**一方のクラスで分散が最大、もう一方で最小**」になるような
チャンネルの重み付け（空間フィルタ）を学習する手法です。
バンドパス後の信号の分散はその帯域の **パワー** に相当するので、
CSP は「左右で最もパワー差が出る合成チャンネル」を作ってくれます。

まず可視化用に全データで CSP を学習し、空間パターンを頭皮上に描きます
（赤/青が強いほどその電極の寄与が大きい）。
"""),
    code(r"""
import mne

# 電極名は標準 10-20 系なので、頭皮図（topomap）が描ける
info = mne.create_info(ch_names, sfreq=fs, ch_types="eeg")
info.set_montage("standard_1020")

csp_vis = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
csp_vis.fit(X, y)

csp_vis.plot_patterns(info, components=range(4), ch_type="eeg", units="a.u.", size=1.5)
plt.show()
"""),
    md(r"""
### CSP 特徴量の様子

各トライアルを CSP に通すと、成分ごとの log 分散（= log パワー）が
特徴ベクトルになります。第 1 成分と最終成分は、左右クラスで最も
分離しやすいように設計されています。散布図で確認しましょう。
"""),
    code(r"""
feat = csp_vis.transform(X)  # (n_trials, n_components)
fig, ax = plt.subplots(figsize=(6, 5))
for label, color in [(U.LABEL_LEFT, "tab:blue"), (U.LABEL_RIGHT, "tab:red")]:
    m = y == label
    ax.scatter(feat[m, 0], feat[m, -1], c=color, label=U.CLASS_NAMES[label], alpha=0.7)
ax.set(xlabel="CSP 成分 0 (log-var)", ylabel="CSP 成分 3 (log-var)",
       title="CSP 特徴空間でのクラス分布")
ax.legend(); plt.tight_layout(); plt.show()
"""),
    md(r"""
## 5. CSP + 分類器で交差検証

CSP と分類器（LDA や SVM）を `Pipeline` でつなぎ、交差検証で精度を
測ります。**ここが最重要ポイント**：

- ❌ トライアルをランダムに分けると、同じ日の似た脳波が訓練とテストの
  両方に入り、精度が **過大評価** されがちです。
- ✅ 今回は **Leave-One-Session-Out**（あるセッション全体をテストに回す）
  で評価します。これは「別の日のデータでも当たるか？」という、BCI で
  本当に問われる汎化性能に近い評価です。

> CSP は交差検証の **各 fold の訓練データだけ** で学習し直します
> （`Pipeline` がこれを自動でやってくれます）。さっきの可視化用 CSP とは
> 別物である点に注意。
"""),
    code(r"""
logo = LeaveOneGroupOut()

pipelines = {
    "CSP + LDA": Pipeline([("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)), ("clf", LDA())]),
    "CSP + SVM": Pipeline([("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
                            ("clf", SVC(kernel="rbf", C=1.0))]),
}

for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X, y, groups=ds.session, cv=logo, scoring="accuracy")
    print(f"{name:12s} | セッション別精度: "
          + ", ".join(f"{s:.2f}" for s in scores)
          + f" | 平均 {scores.mean():.3f} ± {scores.std():.3f}")

print("\n（チャンスレベル = 0.50。2 クラスなので 0.5 を上回れば学習できている）")
"""),
    md(r"""
### 混同行列

CSP + LDA で LOSO の予測を集めて混同行列にします。
"""),
    code(r"""
pipe = Pipeline([("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)), ("clf", LDA())])
y_pred = cross_val_predict(pipe, X, y, groups=ds.session, cv=logo)

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[U.CLASS_NAMES[0], U.CLASS_NAMES[1]])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"CSP + LDA  (LOSO, acc={(y_pred==y).mean():.2f})")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 6. さらに精度を上げる — リーマン幾何 (FgMDM)

### なぜリーマン幾何？

CSP+LDA は「分散（パワー）の最大化」という線形な視点でした。
**リーマン幾何**は脳波の **共分散行列** を、**正定値対称行列の多様体（湾曲した空間）**
上の点として扱い、その空間上の距離で分類します。

- 共分散行列は「どのチャンネル同士が同期しているか」という**空間的なパターン**
  を丸ごと保持している。
- 湾曲した多様体上の距離を正しく測ることで、小規模データでも頑健に動く。

**FgMDM (Frechet-geodesic Minimum Distance to Mean)**: 各クラスの
リーマン重心（平均共分散行列）を学習し、テスト時は最も近い重心のクラスに割り当てる。

### μ帯（8–13Hz）に絞る

実験すると **μ帯のみ（8–13Hz）** でリーマン法を使うと精度が上がる場合があります。
ERD は μ 帯（手の運動野）に最も強く出るため、β 成分の混入がノイズになることがあります。

### 運動野チャンネル選択

FC5 / FC6 / T7 / T8 / F3 / F4 ── 手の感覚運動野に近い 6 チャンネルに絞ります。
頭頂・後頭部のチャンネルは運動想起とは関係が薄く、ノイズになりやすいです。
"""),
    code(r"""
# μ帯 + 運動野 6ch のみで読み込み直す
MOTOR_CH = [c for c in ["FC5", "FC6", "T7", "T8", "F3", "F4"] if c in ch_names]
ds_mu = U.load_dataset(bandpass=(8, 13))
X_mu = ds_mu.X[:, [ds_mu.ch_names.index(c) for c in MOTOR_CH], :]
y_mu, sess_mu = ds_mu.y, ds_mu.session
print(f"μ帯 + 運動野 {len(MOTOR_CH)}ch: {X_mu.shape}")

# FgMDM: 共分散行列の推定 (lwf=Ledoit-Wolf) + リーマン最近傍
pipe_riemann = Pipeline([
    ("cov", Covariances("lwf")),
    ("clf", FgMDM()),
])

scores_riemann = cross_val_score(pipe_riemann, X_mu, y_mu, groups=sess_mu, cv=logo)
print(f"FgMDM (μ帯 + 運動野6ch) | セッション別: "
      + ", ".join(f"{s:.2f}" for s in scores_riemann)
      + f" | 平均 {scores_riemann.mean():.3f} ± {scores_riemann.std():.3f}")
"""),
    md(r"""
### 手法比較まとめ
"""),
    code(r"""
# 全手法の精度を並べてグラフ化
comparisons = {
    "CSP+LDA\n(8-30Hz 全ch)": cross_val_score(
        Pipeline([("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)), ("clf", LDA())]),
        X, y, groups=ds.session, cv=logo),
    "CSP+LDA\n(μ帯 全ch)": cross_val_score(
        Pipeline([("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)), ("clf", LDA())]),
        ds_mu.X, y_mu, groups=sess_mu, cv=logo),
    "FgMDM\n(μ帯 運動野6ch)": scores_riemann,
    "TangentSpace+LDA\n(μ帯 運動野6ch)": cross_val_score(
        Pipeline([("cov", Covariances("lwf")), ("ts", TangentSpace()), ("clf", LDA())]),
        X_mu, y_mu, groups=sess_mu, cv=logo),
}

means = [v.mean() for v in comparisons.values()]
stds  = [v.std()  for v in comparisons.values()]
labels = list(comparisons.keys())

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(labels, means, xerr=stds, color="steelblue", alpha=0.8, capsize=5)
ax.axvline(0.5, color="tomato", ls="--", label="チャンスレベル (0.50)")
for bar, m in zip(bars, means):
    ax.text(m + 0.005, bar.get_y() + bar.get_height()/2, f"{m:.2f}",
            va="center", fontsize=9)
ax.set(xlabel="LOSO accuracy", title="手法別精度比較（平均 ± 標準偏差）", xlim=(0.3, 0.85))
ax.legend(loc="lower right"); plt.tight_layout(); plt.show()
"""),
    md(r"""
## 7. まとめと演習

**実験から分かったこと**
| 手法 | 精度 (LOSO) | ポイント |
|---|---|---|
| CSP + LDA (8–30Hz, 全14ch) | ~0.57 | ベースライン |
| CSP + LDA (μ帯, 全14ch) | ~0.59 | 帯域を絞るだけで改善 |
| FgMDM (μ帯, 運動野6ch) | ~0.64 | リーマン + ch選択がベスト |

- **μ帯 (8–13Hz)** が ERD の主役。β は人によって効いたり効かなかったりする。
- **チャンネル選択**でノイズを減らすとリーマン法が安定する。
- リーマン幾何は「共分散行列の形」ごと分類するため、特徴設計が少なくて済む。
- それでも LOSO で 0.64 程度：C3/C4 が無い EPOC X での MI は難しく、
  「DL も含め、データ量と電極配置が最大のボトルネック」という事実を確認。

**演習（試してみよう）**
1. `MOTOR_CH` に `"P7", "P8"` を足すと精度はどう変わる？（後頭部は逆効果？）
2. `FgMDM` を `MDM`（正則化なし）に替えると？
3. `Covariances("lwf")` の `"lwf"` を `"oas"` や `"scm"` に替えると？
4. ノート ② の EEGNet と今回の FgMDM の精度を並べて比較しよう。
"""),
]
nb1.metadata["kernelspec"] = {
    "display_name": "Python (MotorImageryDemo)",
    "language": "python",
    "name": "python3",
}
(HERE / "01_csp_ml_classical_decoder.ipynb").write_text(nbf.writes(nb1))
print("wrote 01_csp_ml_classical_decoder.ipynb")


# =====================================================================
# Notebook 2: EEGNet (深層学習)
# =====================================================================
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md(r"""
# 脳波デコーディング体験 ② — EEGNet（深層学習）

ノート ① では「前処理 → CSP（特徴抽出）→ 分類器」と段階を分けました。
ここでは **EEGNet** という小型の畳み込みニューラルネットで、
前処理済み波形から **end-to-end**（特徴設計なし）で左右を当てます。

## EEGNet とは
EEG 用に設計されたコンパクトな CNN（Lawhern et al., 2018）。主役は 3 種類の畳み込み：
1. **時間方向の畳み込み** … 各チャンネルで帯域フィルタのようなものを学習
2. **Depthwise 畳み込み（空間）** … チャンネル方向に重みを学習＝ CSP のような空間フィルタの役割
3. **Separable 畳み込み** … 時間パターンを効率よくまとめる

CSP+LDA を「人間が設計した空間フィルタ＋線形分類」とすれば、EEGNet は
それを **データから丸ごと学習** する版、と捉えると分かりやすいです。

> ⚠️ 今回のデータは 100 トライアルと **とても小規模**。深層学習は本来
> 大量データで強みを発揮するので、古典手法（CSP）に精度で負けることも
> あります。「DL なら必ず勝つ」わけではない、というのも大事な学びです。
"""),
    code(r"""
import mi_utils as U          # 先頭で KMP_DUPLICATE_LIB_OK を設定済み
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

U.setup_japanese_font()  # グラフの日本語が豆腐(□)にならないように

# 再現性
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)

# Apple Silicon なら MPS、無ければ CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device, "| torch", torch.__version__)
"""),
    md(r"""
## 1. データの準備（窓に展開してデータ拡張）

実際のリアルタイム BCI は **2 秒の窓** を 0.5 秒ずつずらして分類して
います。同じ要領で 4 秒トライアルを 2 秒窓に切り分けると、サンプル数が
増えて小規模データでの学習が安定します。
"""),
    code(r"""
# μ帯(8-13Hz)に絞る：実験でリーマン法と同様 μ帯が EEGNet でも有効
ds = U.load_dataset(bandpass=(8, 13))
Xw, yw, sess_w = U.make_windows(ds, window_s=2.0, stride_s=0.5)
print("窓データ:", Xw.shape, "(窓数, チャンネル, サンプル)")
print("ラベル内訳:", np.bincount(yw), "| セッション:", np.bincount(sess_w))

n_ch, n_time = Xw.shape[1], Xw.shape[2]
"""),
    md(r"""
## 2. EEGNet の実装

論文の標準構成（F1=8, D=2, F2=16）を、14ch・256 サンプル（2 秒）・2 クラス
向けに用意します。各層が何をしているかコメントを付けています。
"""),
    code(r"""
class EEGNet(nn.Module):
    def __init__(self, n_ch, n_time, n_classes=2, F1=8, D=2, F2=16,
                 kern_len=64, p_drop=0.5):
        super().__init__()
        # --- ブロック1: 時間畳み込み + 空間(depthwise)畳み込み ---
        # 時間方向の畳み込み = 学習可能な帯域フィルタ群（F1 枚）
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        # depthwise: チャンネル(空間)方向にまとめる ≒ CSP 的な空間フィルタ
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),      # 時間方向にダウンサンプル
            nn.Dropout(p_drop),
        )
        # --- ブロック2: separable 畳み込み（時間パターンの集約） ---
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p_drop),
        )
        # 出力サイズを動的に求めて全結合層を作る
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_time)
            feat_dim = self.separable(self.spatial(self.temporal(dummy))).numel()
        self.classify = nn.Linear(feat_dim, n_classes)

    def forward(self, x):                 # x: (N, 1, n_ch, n_time)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = x.flatten(1)
        return self.classify(x)

# 形状チェック
_m = EEGNet(n_ch, n_time)
print(_m(torch.zeros(2, 1, n_ch, n_time)).shape, "(バッチ, クラス)")
print("パラメータ数:", sum(p.numel() for p in _m.parameters()))
"""),
    md(r"""
## 3. 学習と評価（Leave-One-Session-Out）

ノート ① と同じく、汎化性能を見るため **セッションをまたいで** 評価します。
各 fold で:
1. テスト用に 1 セッションを除外
2. 残りで **チャンネルごとの z 標準化** の統計量を計算（訓練データのみ）
3. EEGNet を学習し、除外セッションで精度を測る
"""),
    code(r"""
def standardize(train, *others):
    # チャンネルごとに平均・分散を訓練データから求めて適用
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True) + 1e-7
    return [( (a - mean) / std ) for a in (train, *others)]

def to_tensor(X, y):
    Xt = torch.tensor(X[:, None, :, :], dtype=torch.float32)  # (N,1,C,T)
    yt = torch.tensor(y, dtype=torch.long)
    return Xt, yt

def train_one_fold(Xtr, ytr, Xte, yte, epochs=120, batch=32, lr=1e-3,
                   weight_decay=1e-4, use_scheduler=True, verbose=False):
    Xtr, Xte = standardize(Xtr, Xte)
    Xtr_t, ytr_t = to_tensor(Xtr, ytr)
    Xte_t, yte_t = to_tensor(Xte, yte)

    model = EEGNet(n_ch, n_time).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # CosineAnnealing: LR を余弦関数的に減衰させ、収束を安定させる
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_scheduler else None
    lossf = nn.CrossEntropyLoss()
    ds_tr = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    loader = torch.utils.data.DataLoader(ds_tr, batch_size=batch, shuffle=True)

    history = []
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        if scheduler: scheduler.step()
        # エポックごとのテスト精度を記録（学習曲線の可視化用）
        model.eval()
        with torch.no_grad():
            pred = model(Xte_t.to(device)).argmax(1).cpu().numpy()
        acc = (pred == yte).mean()
        history.append(acc)
        if verbose and (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:3d} | test acc {acc:.2f}")
    return pred, np.array(history)
"""),
    code(r"""
logo = LeaveOneGroupOut()
fold_acc, all_true, all_pred = [], [], []
first_history = None

for k, (tr, te) in enumerate(logo.split(Xw, yw, groups=sess_w)):
    pred, hist = train_one_fold(Xw[tr], yw[tr], Xw[te], yw[te],
                                epochs=120, use_scheduler=True, verbose=(k == 0))
    acc = (pred == yw[te]).mean()
    fold_acc.append(acc)
    all_true.append(yw[te]); all_pred.append(pred)
    if first_history is None:
        first_history = hist
    print(f"fold {k} (test=session {sess_w[te][0]}): acc {acc:.3f}")

print(f"\nEEGNet  LOSO 平均精度: {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}")
print("（チャンスレベル = 0.50）")
"""),
    md(r"""
## 4. 結果の可視化

学習曲線（fold 0）と、全 fold をまとめた混同行列を見ます。
"""),
    code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(first_history)
ax1.set(xlabel="epoch", ylabel="test accuracy", title="学習曲線 (fold 0)", ylim=(0, 1))
ax1.axhline(0.5, color="gray", ls="--", label="chance")
ax1.legend()

yt = np.concatenate(all_true); yp = np.concatenate(all_pred)
cm = confusion_matrix(yt, yp)
ConfusionMatrixDisplay(cm, display_labels=[U.CLASS_NAMES[0], U.CLASS_NAMES[1]]).plot(
    cmap="Oranges", values_format="d", ax=ax2, colorbar=False)
ax2.set_title(f"EEGNet (LOSO, acc={(yt==yp).mean():.2f})")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 5. まとめと演習

**改善のポイント（ベースラインからの変更点）**
| 変更 | 効果 |
|---|---|
| `bandpass=(8,13)` μ帯に絞る | 運動想起の ERD が最も強い帯域に集中 |
| `epochs=120`（60→120）| 小バッチで収束が遅いため、十分な反復が必要 |
| `CosineAnnealingLR` | LR を余弦関数で滑らかに減衰させ、収束を安定化 |
| `weight_decay=1e-4` | L2 正則化でわずかに過学習を抑制 |

**まとめ**
- μ帯フィルタ + LR スケジューラで EEGNet の安定性が向上する。
- depthwise 畳み込みは CSP のような **空間フィルタ** を学習しており、
  古典手法と発想は地続き。
- それでも LOSO でリーマン手法（FgMDM ~0.64）に追いつくのは難しい。
  **DL が真に強みを発揮するにはデータ量が 1 桁以上必要**。

**演習**
1. `use_scheduler=False` に戻すと学習曲線はどう変わる？
2. `epochs` を 60 / 200 に変えると精度は？（過学習 vs 未学習の境界を探そう）
3. `make_windows(window_s=..., stride_s=...)` で窓の長さ・刻みを変えると？
4. ノート ① の FgMDM と今回の EEGNet の精度を並べて比べよう。
   「どちらが良い」より「なぜ差が出るのか」を考えてみよう。
"""),
]
nb2.metadata["kernelspec"] = {
    "display_name": "Python (MotorImageryDemo)",
    "language": "python",
    "name": "python3",
}
(HERE / "02_eegnet_dl_decoder.ipynb").write_text(nbf.writes(nb2))
print("wrote 02_eegnet_dl_decoder.ipynb")
