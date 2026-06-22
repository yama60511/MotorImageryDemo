"""体験授業用の共有ユーティリティ。

2 つの notebook（CSP+ML / EEGNet）が共通で使う「データ読み込み・前処理・
トライアル切り出し」をここにまとめています。実際の BCI サービス
(`bci/` パッケージ) と同じ前処理・エポック化コードを再利用しているので、
notebook で学ぶ流れがそのままリアルタイム推論システムの中身になります。

使い方::

    from mi_utils import load_dataset
    ds = load_dataset()           # data ディレクトリを自動探索
    print(ds.X.shape, ds.y.shape)

データの置き場所は環境変数 ``MI_DATA_DIR`` でも上書きできます。
"""

from __future__ import annotations

# --- macOS / conda の OpenMP 二重ロード対策 -------------------------------
# PyTorch と numpy/mne がそれぞれ libomp を読み込むと "OMP: Error #15" で
# クラッシュする。torch を import する前にこのフラグを立てておく。
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# プロジェクトルート (bci/ がある場所) を import パスに追加する。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bci.config import PreprocessingConfig  # noqa: E402
from bci.epoching import extract_trials, sliding_windows  # noqa: E402
from bci.preprocessing import Preprocessor  # noqa: E402

# クラスラベル（記録メタデータと bci/epoching.py の規約に一致）
LABEL_LEFT = 0
LABEL_RIGHT = 1
CLASS_NAMES = {LABEL_LEFT: "left", LABEL_RIGHT: "right"}

# データの既定の探索先。先頭から順に試し、最初に見つかったものを使う。
_DEFAULT_DATA_DIRS = [
    Path("/Users/yutayamamura/Downloads/MI_yamamura_data"),
    _REPO_ROOT / "data" / "MI_yamamura_data",
    _REPO_ROOT / "data",
]


@dataclass
class Dataset:
    """切り出し済みのデータをまとめて持つ箱。

    Attributes
    ----------
    X : (n_trials, n_channels, n_samples) float64
        前処理（notch + band-pass + CAR）済みの MI 区間エポック。
    y : (n_trials,) int
        クラスラベル（0=left, 1=right）。
    session : (n_trials,) int
        各トライアルがどのセッション由来かを示す番号。
        被験者内でもセッション間で脳波分布はずれるため、
        汎化性能を正しく測るには「セッションをまたいだ」評価が要る。
        この配列を sklearn の ``groups`` に渡すと leave-one-session-out
        などが組める。
    fs : float
        サンプリング周波数 [Hz]。
    ch_names : list[str]
        チャンネル名（10-20 系の電極ラベル）。
    session_ids : list[str]
        セッション番号 -> 元のセッションディレクトリ名。
    """

    X: np.ndarray
    y: np.ndarray
    session: np.ndarray
    fs: float
    ch_names: list[str]
    session_ids: list[str]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    def summary(self) -> str:
        import numpy as _np

        lines = [
            f"trials      : {self.X.shape[0]}",
            f"channels    : {self.n_channels}  {self.ch_names}",
            f"samples/trial: {self.X.shape[2]}  ({self.X.shape[2] / self.fs:.1f}s @ {self.fs:.0f}Hz)",
            f"sessions    : {len(self.session_ids)}",
        ]
        for s_idx, s_id in enumerate(self.session_ids):
            mask = self.session == s_idx
            counts = {CLASS_NAMES[c]: int((self.y[mask] == c).sum()) for c in CLASS_NAMES}
            lines.append(f"  [{s_idx}] {s_id}: {counts}")
        total = {CLASS_NAMES[c]: int((self.y == c).sum()) for c in CLASS_NAMES}
        lines.append(f"class totals: {total}")
        return "\n".join(lines)


def setup_japanese_font() -> str | None:
    """matplotlib のグラフで日本語が豆腐(□)にならないよう、利用可能な
    日本語フォントを選んで設定する。見つからなければ何もしない。"""
    from matplotlib import font_manager as fm
    from matplotlib import pyplot as plt

    available = {f.name for f in fm.fontManager.ttflist}
    for cand in (
        "Hiragino Sans",
        "Hiragino Maru Gothic Pro",
        "YuGothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "Arial Unicode MS",
        "AppleGothic",
    ):
        if cand in available:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の豆腐回避
            return cand
    return None


def find_data_dir() -> Path:
    """記録データの入ったディレクトリを探す。"""
    env = os.environ.get("MI_DATA_DIR")
    candidates = [Path(env)] if env else []
    candidates += _DEFAULT_DATA_DIRS
    for d in candidates:
        if d.is_dir() and list(d.glob("*/recording.npz")):
            return d
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "recording.npz を含むデータディレクトリが見つかりませんでした。\n"
        f"探した場所:\n  {searched}\n"
        "環境変数 MI_DATA_DIR にデータの場所を指定してください。"
    )


def load_dataset(
    data_dir: str | os.PathLike | None = None,
    reference: str = "auto",
    bandpass: tuple[float, float] = (8.0, 30.0),
    notch_hz: float | None = 50.0,
) -> Dataset:
    """全セッションを読み込み、前処理して MI エポックに切り出す。

    実サービスと同じ :class:`bci.preprocessing.Preprocessor`
    （notch -> band-pass -> CAR）でセッションごとに連続波形を整形し、
    :func:`bci.epoching.extract_trials` で MI 区間（``mi_s`` 秒）を
    切り出す。マーカーは MI 開始時刻に打たれているので ``onset_s=0``。

    Parameters
    ----------
    data_dir : path, optional
        データの場所。省略時は :func:`find_data_dir` で自動探索。
    reference : {"auto", "car", "none"}
        再基準化。14ch では "auto" で CAR が有効になる。
    bandpass : (low, high)
        帯域通過フィルタ [Hz]。mu/beta 帯（運動想起の ERD/ERS が出る帯域）。
    notch_hz : float or None
        ノッチ周波数 [Hz]。東日本の商用電源は 50Hz。
    """
    root = Path(data_dir) if data_dir is not None else find_data_dir()
    session_dirs = sorted(p.parent for p in root.glob("*/recording.npz"))
    if not session_dirs:
        raise FileNotFoundError(f"{root} に recording.npz がありません。")

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_sess: list[np.ndarray] = []
    session_ids: list[str] = []
    fs_ref: float | None = None
    ch_ref: list[str] | None = None

    for s_idx, sdir in enumerate(session_dirs):
        rec = np.load(sdir / "recording.npz", allow_pickle=True)
        raw = rec["raw"].astype(np.float64)  # (ch, T)
        timestamps = rec["timestamps"].astype(np.float64)
        marker_times = rec["marker_times"].astype(np.float64)
        marker_labels = rec["marker_labels"].astype(int)
        fs = float(rec["fs"])
        ch_names = [str(c) for c in rec["channel_order"]]

        if fs_ref is None:
            fs_ref, ch_ref = fs, ch_names
        # mi_s はメタデータから（無ければ既定 4.0）
        meta_path = sdir / "metadata.json"
        mi_s = 4.0
        if meta_path.exists():
            import json

            mi_s = float(json.loads(meta_path.read_text()).get("mi_s", 4.0))

        pre = Preprocessor(
            PreprocessingConfig(
                fs=fs,
                channel_order=ch_names,
                bandpass=bandpass,
                notch_hz=notch_hz,
                reference=reference,
            )
        )
        filtered = pre.transform(raw)  # (ch, T)

        markers = list(zip(marker_times.tolist(), marker_labels.tolist()))
        X, y = extract_trials(filtered, timestamps, markers, mi_s, fs, onset_s=0.0)
        if X.shape[0] == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        all_sess.append(np.full(len(y), s_idx, dtype=np.int64))
        session_ids.append(sdir.name)

    if not all_X:
        raise RuntimeError("有効なトライアルが 1 つも切り出せませんでした。")

    return Dataset(
        X=np.concatenate(all_X, axis=0),
        y=np.concatenate(all_y, axis=0),
        session=np.concatenate(all_sess, axis=0),
        fs=float(fs_ref),
        ch_names=list(ch_ref),  # type: ignore[arg-type]
        session_ids=session_ids,
    )


def make_windows(
    ds: Dataset, window_s: float = 2.0, stride_s: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各トライアルを重なりありの短い窓に展開する（データ拡張）。

    実サービスのリアルタイム推論は ``window_s`` 秒の窓を ``stride`` ごとに
    分類している。これに合わせて訓練データを窓に割ると、サンプル数が増えて
    特に DL（EEGNet）の学習に効く。

    Returns
    -------
    Xw : (n_windows, ch, win_samples)
    yw : (n_windows,)
    sess_w : (n_windows,)  各窓の元セッション番号（リーク防止のグループ用）
    """
    win = int(round(window_s * ds.fs))
    stride = int(round(stride_s * ds.fs))
    Xw, yw, trial_group = sliding_windows(ds.X, ds.y, win, stride)
    # trial_group は「何番目のトライアル由来か」。セッション番号に変換する。
    sess_w = ds.session[trial_group]
    return Xw, yw, sess_w
