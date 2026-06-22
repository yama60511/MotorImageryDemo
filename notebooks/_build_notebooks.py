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

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mne
from mne.decoding import CSP

mne.set_log_level("WARNING")  # CSP の冗長ログを抑制

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

最も素直な CSP + LDA で、Leave-One-Session-Out の予測を全 fold 集めて
混同行列にします。
"""),
    code(r"""
from sklearn.model_selection import cross_val_predict

pipe = Pipeline([("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)), ("clf", LDA())])
y_pred = cross_val_predict(pipe, X, y, groups=ds.session, cv=logo)

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[U.CLASS_NAMES[0], U.CLASS_NAMES[1]])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"CSP + LDA  (LOSO, acc={(y_pred==y).mean():.2f})")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 6. まとめと演習

**まとめ**
- 前処理（帯域制限）→ CSP（空間フィルタで左右差を強調）→ 軽い分類器、
  という流れだけで運動想起をデコードできる。
- 評価はセッションをまたいで行うのが誠実（過大評価を避ける）。
- 古典手法は **データが少なくても安定** しやすいのが強み。

**演習（コードを書き換えて試そう）**
1. `load_dataset(bandpass=(...))` で帯域を μ 帯のみ(8–13)や β 帯のみ(13–30)に
   変えると精度はどう変わる？
2. `CSP(n_components=...)` を 2 / 6 / 8 に変えると？
3. 1 セッションだけ取り出して、そのセッション内で `StratifiedKFold` した
   精度と LOSO の精度を比べてみよう（差が「日による脳波の揺らぎ」）。
4. 次のノートブック ② では、同じデータを **EEGNet（深層学習）** で
   デコードします。古典手法と精度・手間を比べてみましょう。
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
ds = U.load_dataset()
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

def train_one_fold(Xtr, ytr, Xte, yte, epochs=60, batch=32, lr=1e-3, verbose=False):
    Xtr, Xte = standardize(Xtr, Xte)
    Xtr_t, ytr_t = to_tensor(Xtr, ytr)
    Xte_t, yte_t = to_tensor(Xte, yte)

    model = EEGNet(n_ch, n_time).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
        # エポックごとのテスト精度を記録（学習曲線の可視化用）
        model.eval()
        with torch.no_grad():
            pred = model(Xte_t.to(device)).argmax(1).cpu().numpy()
        acc = (pred == yte).mean()
        history.append(acc)
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:3d} | test acc {acc:.2f}")
    return pred, np.array(history)
"""),
    code(r"""
logo = LeaveOneGroupOut()
fold_acc, all_true, all_pred = [], [], []
first_history = None

for k, (tr, te) in enumerate(logo.split(Xw, yw, groups=sess_w)):
    pred, hist = train_one_fold(Xw[tr], yw[tr], Xw[te], yw[te],
                                verbose=(k == 0))
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

**まとめ**
- EEGNet は前処理済み波形から特徴設計なしで左右をデコードできる。
- depthwise 畳み込みは CSP のような **空間フィルタ** を学習しており、
  古典手法と発想は地続き。
- ただしデータが小規模だと、CSP+LDA の方が安定して強いことも多い。
  「手法の派手さ」より「データ量と評価設計」が効く、という感覚を持とう。

**演習**
1. `epochs` や `lr`、`p_drop` を変えると学習曲線・精度はどう動く？
2. `make_windows(window_s=..., stride_s=...)` で窓の長さ・刻みを変えると？
3. ノート ① の CSP+LDA と精度を並べて比べよう。どちらが安定している？
4. （発展）`standardize` を外すと学習がどう不安定になるか観察しよう。
"""),
]
nb2.metadata["kernelspec"] = {
    "display_name": "Python (MotorImageryDemo)",
    "language": "python",
    "name": "python3",
}
(HERE / "02_eegnet_dl_decoder.ipynb").write_text(nbf.writes(nb2))
print("wrote 02_eegnet_dl_decoder.ipynb")
