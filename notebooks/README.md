# EEG Decoding Notebook

Motor Imagery（MI）EEGから、左手・右手の運動イメージを分類するための教材です。
`eeg-decoding.ipynb` では、同じデータに対して古典的手法の CSP + LDA と、
深層学習モデルの EEGNet / EEG-Conformer（`braindecode`）を比較します。

## ファイル構成

```text
notebooks/
├── eeg-decoding.ipynb  # 解析・可視化・デコーディング
├── utils.py            # EEGデータの読み込み
└── assets/             # notebookで使用する画像
```

## Notebookの内容

1. 生EEGデータの読み込みと確認
2. MNEによる前処理（ノッチフィルタ、バンドパス、CAR）とエポック化
3. 前処理後の信号確認と不良試行の除外
4. ERDSマップの計算と可視化
5. CSP特徴量の可視化
6. CSP + LDA、EEGNet、EEG-Conformerの学習・評価・比較

## データ配置

`notebooks` と `data` を同じ階層に配置してください。

```text
MotorImageryDemo/
├── notebooks/
│   ├── eeg-decoding.ipynb
│   ├── utils.py
│   └── assets/
└── data/
    ├── session-1/
    │   └── recording.npz
    ├── session-2/
    │   └── recording.npz
    └── ...
```

`utils.py` は次の順にデータディレクトリを探索します。

1. 環境変数 `MI_DATA_DIR`
2. `../data`
3. `./data`

別の場所にデータがある場合は、環境変数を設定するか、
`U.load_raw_eeg("/path/to/data")` のように直接指定してください。

各 `recording.npz` には、少なくとも次の配列が必要です。

- `raw`: EEG信号（channels × samples、単位は µV）
- `timestamps`: 各サンプルのタイムスタンプ
- `marker_times`: 試行開始時刻
- `marker_labels`: ラベル（0=left、1=right、99=rest）
- `fs`: サンプリング周波数
- `channel_order`: チャンネル名

## Google Colabで実行

1. `MotorImageryDemo` フォルダをGoogle Driveの `MyDrive` 直下に配置します。
2. `eeg-decoding.ipynb` をGoogle Colabで開きます。
3. 先頭のSetupセルを実行します。
4. 以降のセルを上から順に実行します。

Setupセルは `mne` と `braindecode` をインストールし、Google Driveをマウントして
`/content/drive/MyDrive/MotorImageryDemo/notebooks` に移動します。配置先を変更した場合は、
Setupセル内の `%cd` も変更してください。

深層学習部分は、Colabの「ランタイム」→「ランタイムのタイプを変更」からGPUを選ぶと
高速に実行できます。CPUでも実行可能です。

## ローカルで実行

プロジェクトの環境を有効化してJupyterを起動します。

```bash
conda activate MotorImageryDemo
jupyter lab notebooks/eeg-decoding.ipynb
```

データを標準配置に置かない場合は、起動前に `MI_DATA_DIR` を設定してください。

```bash
export MI_DATA_DIR=/path/to/data
```

## `utils.py`

- `load_raw_eeg(data_dir=None)`: 全セッションの `recording.npz` を読み込み、連結した
  `mne.io.RawArray` を返します。µVの入力はVへ変換され、試行ラベルはMNE Annotationとして付与されます。
- `print_raw_summary(raw)`: チャンネル数、サンプル数、収録時間、サンプリング周波数、
  Annotation数を表示します。
- `LABEL_NAMES`: `{0: "left", 1: "right", 99: "rest"}` のラベル対応です。

## 電極配置について

Emotiv EPOC Xには、運動野直上の標準的なMI用電極 C3/C4 がありません。
このnotebookでは近傍の FC5/FC6 などを使用するため、電極配置やデータ量によっては
分類精度がチャンスレベル（50%）に近くなる場合があります。
