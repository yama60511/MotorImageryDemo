# 脳波デコーディング体験ノートブック

脳波解析体験授業向けの、運動想起（Motor Imagery）EEG デコーディング教材です。
このリポジトリの Emotiv EPOC X（14ch, 128Hz）で記録した「左手 / 右手を動かす想像」
のデータを使い、2 つのアプローチでデコーダを作って比較します。

| ノートブック | 手法 | 内容 |
|---|---|---|
| [`01_csp_ml_classical_decoder.ipynb`](01_csp_ml_classical_decoder.ipynb) | **CSP + 機械学習**（古典） | バンドパス → CSP（空間フィルタ）→ LDA/SVM。ERD の可視化、CSP パターンの頭皮図、Leave-One-Session-Out 評価。 |
| [`02_eegnet_dl_decoder.ipynb`](02_eegnet_dl_decoder.ipynb) | **EEGNet**（深層学習） | 前処理済み波形から end-to-end でデコード。PyTorch 実装、学習曲線、古典手法との比較。 |

`①`（古典）→ `②`（DL）の順に進めるのがおすすめです。

## 必要環境

conda 環境 `MotorImageryDemo` を使います（`environment.yml`）。EEGNet ノート用に
PyTorch が追加で必要です。

```bash
conda activate MotorImageryDemo
pip install torch          # ② EEGNet ノートのみ必要
```

Jupyter はこの環境の Python で起動してください。

```bash
conda activate MotorImageryDemo
jupyter lab    # または jupyter notebook
```

> **macOS の OpenMP 警告について**: PyTorch と numpy/mne が OpenMP を二重ロード
> すると `OMP: Error #15` でクラッシュすることがあります。`mi_utils` が
> `KMP_DUPLICATE_LIB_OK=TRUE` を自動設定するので、**`mi_utils` を最初に import**
> しておけば回避できます（両ノートとも対応済み）。

## データの場所

`mi_utils.load_dataset()` が以下を順に探します（最初に見つかったものを使用）:

1. 環境変数 `MI_DATA_DIR`
2. `/Users/yutayamamura/Downloads/MI_yamamura_data`
3. リポジトリ内 `data/MI_yamamura_data` または `data/`

別の場所のデータを使うときは:

```python
import os
os.environ["MI_DATA_DIR"] = "/path/to/your/recordings"
```

各セッションは `recording.npz`（連続波形・タイムスタンプ・マーカー）と
`metadata.json` を含みます。

## 共有ユーティリティ `mi_utils.py`

両ノートが共通で使う読み込み・前処理を切り出したモジュールです。
**実際のリアルタイム BCI（`bci/` パッケージ）と同じ前処理・エポック化コード**を
呼んでいるので、ノートで学ぶ流れがそのまま本番システムの中身になります。

- `load_dataset()` … 全セッションを読み込み、notch→band-pass→CAR をかけて
  MI 区間（4 秒）のエポックに切り出す。
- `make_windows()` … 各トライアルを 2 秒窓に展開（リアルタイム推論と同じ）。
- `setup_japanese_font()` … グラフの日本語が豆腐（□）にならないようにする。

## 注意：チャンネル配置

古典 BCI で使う C3/C4（手の運動野直上）の電極が EPOC X には無いため、
感覚運動野に近い **FC5 / FC6 / T7 / T8** を手がかりにしています。
そのため精度は控えめ（チャンスレベル 50% を少し超える程度）になることがあります。
「手法の派手さより、電極配置・データ量・評価設計が効く」という点も含めて
体験してもらう構成です。
