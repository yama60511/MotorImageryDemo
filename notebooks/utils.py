"""Shared utilities for the teaching notebooks.

Centralises data loading, preprocessing, and trial extraction used by both
notebooks (CSP+ML and EEGNet). The preprocessing pipeline (notch → bandpass → CAR)
and epoching logic mirror the real-time BCI service, making offline analysis
directly comparable to live inference.

    from utils import load_dataset
    ds = load_dataset()          # auto-searches ../data
    print(ds.summary())

The data directory can be set via the ``data_dir`` argument or the
``MI_DATA_DIR`` environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

# Prevent OpenMP double-load crash (OMP: Error #15) that occurs when PyTorch and
# numpy both load OpenMP on macOS/conda. Must be imported before torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Class labels (must match the recording metadata convention)
LABEL_LEFT = 0
LABEL_RIGHT = 1
CLASS_NAMES = {LABEL_LEFT: "left", LABEL_RIGHT: "right"}

# Fallback data directories used when data_dir and MI_DATA_DIR are both unset.
# Students typically place recordings directly under ``data/`` (see README); the
# remaining entries cover the author's local layouts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIRS = [
    _REPO_ROOT / "data",
    _REPO_ROOT / "data" / "eeg",
    _REPO_ROOT / "data" / "MI_yamamura_data",
]


# === Preprocessing (notch → bandpass → CAR) ====================================
@dataclass
class PreprocessingConfig:
    fs: float
    channel_order: list[str]
    bandpass: tuple[float, float] = (8.0, 30.0)
    bandpass_order: int = 4
    notch_hz: float | None = 50.0  # Tokyo mains frequency (East Japan = 50 Hz)
    notch_q: float = 30.0
    reference: str = "auto"  # "auto" | "car" | "none"

    @property
    def apply_car(self) -> bool:
        if self.reference == "car":
            return True
        if self.reference == "none":
            return False
        return len(self.channel_order) >= 4  # auto: enable CAR when ≥ 4 channels


class Preprocessor:
    """Applies notch → bandpass → CAR in sequence.

    Each filter is initialised to steady state at the first sample
    (``sosfilt_zi * x0``) to suppress the large DC offset inherent to
    Emotiv recordings and prevent a ringing transient at the start.
    """

    def __init__(self, cfg: PreprocessingConfig) -> None:
        self.cfg = cfg
        if cfg.notch_hz is not None:
            b, a = iirnotch(w0=cfg.notch_hz, Q=cfg.notch_q, fs=cfg.fs)
            self.sos_notch: np.ndarray | None = tf2sos(b, a)
        else:
            self.sos_notch = None
        self.sos_bp = butter(
            cfg.bandpass_order, list(cfg.bandpass), btype="band", fs=cfg.fs, output="sos"
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Filter an array of shape (ch, T) or (n_trials, ch, T)."""
        arr = np.asarray(X, dtype=np.float64)
        if self.sos_notch is not None:
            arr = self._sosfilt_steady(self.sos_notch, arr)
        arr = self._sosfilt_steady(self.sos_bp, arr)
        if self.cfg.apply_car:
            arr = arr - arr.mean(axis=-2, keepdims=True)
        return arr

    @staticmethod
    def _sosfilt_steady(sos: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """Apply ``sosfilt`` along the last axis, initialised to steady state at the
        first sample of each lane to suppress DC-offset transients."""
        if arr.shape[-1] == 0:
            return arr
        x2 = arr.reshape(-1, arr.shape[-1])  # (n_lanes, T)
        zi = sosfilt_zi(sos)[:, None, :] * x2[None, :, 0, None]
        y2, _ = sosfilt(sos, x2, axis=-1, zi=zi)
        return y2.reshape(arr.shape)


# === Epoch extraction ====================================================
_TRIAL_LABELS = frozenset({LABEL_LEFT, LABEL_RIGHT})  # discard non-MI markers (e.g. 99=rest)


def extract_trials(
    raw: np.ndarray,
    timestamps: np.ndarray,
    markers: list[tuple[float, int]],
    trial_len_s: float,
    fs: float,
    onset_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed-length trials from continuous EEG.

    ``markers`` is a list of ``(timestamp_lsl_s, label)`` tuples; only labels 0 and 1
    are kept. Returns ``X:(n_trials, ch, n_samples)`` and ``y:(n_trials,)``.
    """
    n_trial = int(round(trial_len_s * fs))
    onset_samples = int(round(onset_s * fs))
    trials: list[np.ndarray] = []
    labels: list[int] = []
    for t_lsl, label in markers:
        if label not in _TRIAL_LABELS:
            continue
        i0 = int(np.searchsorted(timestamps, t_lsl, side="left")) + onset_samples
        i1 = i0 + n_trial
        if i0 < 0 or i1 > raw.shape[1]:  # skip markers too close to the recording edge
            continue
        trials.append(raw[:, i0:i1])
        labels.append(int(label))
    if not trials:
        return np.zeros((0, raw.shape[0], n_trial)), np.zeros((0,), dtype=np.int64)
    return np.stack(trials), np.asarray(labels, dtype=np.int64)


def sliding_windows(
    trials: np.ndarray,
    labels: np.ndarray,
    window_samples: int,
    stride_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand each trial into overlapping windows.

    Returns ``(X, y, group)`` where ``group[i]`` is the source trial index for
    window i (used to keep windows from the same trial in the same CV fold).
    """
    n_trials, n_channels, n_samples = trials.shape
    n_per_trial = 1 + (n_samples - window_samples) // stride_samples
    starts = np.arange(n_per_trial) * stride_samples
    n_total = n_trials * n_per_trial
    X = np.empty((n_total, n_channels, window_samples), dtype=trials.dtype)
    y = np.empty(n_total, dtype=labels.dtype)
    group = np.empty(n_total, dtype=np.int64)
    for ti in range(n_trials):
        for wi, s in enumerate(starts):
            X[ti * n_per_trial + wi] = trials[ti, :, s : s + window_samples]
        y[ti * n_per_trial : (ti + 1) * n_per_trial] = labels[ti]
        group[ti * n_per_trial : (ti + 1) * n_per_trial] = ti
    return X, y, group


# === Dataset =============================================================
@dataclass
class Dataset:
    """Container for preprocessed, epoched MI data."""

    X: np.ndarray  # (n_trials, ch, samples) preprocessed MI epochs
    y: np.ndarray  # (n_trials,) 0=left, 1=right
    session: np.ndarray  # (n_trials,) session index (group key for LOSO evaluation)
    fs: float
    ch_names: list[str]
    session_ids: list[str]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    def summary(self) -> str:
        lines = [
            f"trials       : {self.X.shape[0]}",
            f"channels     : {self.n_channels}  {self.ch_names}",
            f"samples/trial: {self.X.shape[2]}  ({self.X.shape[2] / self.fs:.1f}s @ {self.fs:.0f}Hz)",
            f"sessions     : {len(self.session_ids)}",
        ]
        for s_idx, s_id in enumerate(self.session_ids):
            mask = self.session == s_idx
            counts = {CLASS_NAMES[c]: int((self.y[mask] == c).sum()) for c in CLASS_NAMES}
            lines.append(f"  [{s_idx}] {s_id}: {counts}")
        total = {CLASS_NAMES[c]: int((self.y == c).sum()) for c in CLASS_NAMES}
        lines.append(f"class totals : {total}")
        return "\n".join(lines)


def remove_bad_trials(dataset: Dataset, bad_trials: list[int]) -> Dataset:
    """Return a dataset without the specified original trial indices."""
    bad = np.unique(np.asarray(bad_trials, dtype=int))
    if np.any((bad < 0) | (bad >= len(dataset.X))):
        raise IndexError(f"bad_trials must be between 0 and {len(dataset.X) - 1}")
    if len(bad) == len(dataset.X):
        raise ValueError("bad_trials cannot remove every trial")
    keep = np.ones(len(dataset.X), dtype=bool)
    keep[bad] = False
    return Dataset(
        X=dataset.X[keep],
        y=dataset.y[keep],
        session=dataset.session[keep],
        fs=dataset.fs,
        ch_names=dataset.ch_names,
        session_ids=dataset.session_ids,
    )


@dataclass
class ERDEpochs:
    """Continuous epochs aligned to MI onset for descriptive ERD analysis."""

    epochs: np.ndarray  # (n_trials, ch, samples), from epoch_window_s[0] to [1]
    times: np.ndarray  # (samples,), seconds relative to MI onset
    baseline: np.ndarray  # (n_trials, ch, baseline_samples)
    mi: np.ndarray  # (n_trials, ch, mi_samples)
    y: np.ndarray
    session: np.ndarray
    original_trial: np.ndarray  # index in Dataset returned by load_dataset()
    fs: float
    ch_names: list[str]


def find_data_dir() -> Path:
    """Search for a data folder containing recording.npz files."""
    env = os.environ.get("MI_DATA_DIR")
    candidates = ([Path(env)] if env else []) + _DEFAULT_DATA_DIRS
    for d in candidates:
        if d.is_dir() and list(d.glob("*/recording.npz")):
            return d
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find a data folder containing recording.npz.\n"
        f"Searched:\n  {searched}\n"
        "Set DATA_DIR or the MI_DATA_DIR environment variable to the data location."
    )


@dataclass
class _RawSession:
    """One session's continuous EEG after preprocessing, ready for epoching."""

    s_idx: int  # position in the sorted session list (group key for LOSO)
    name: str
    filtered: np.ndarray  # (ch, T) after notch → bandpass → CAR
    timestamps: np.ndarray  # (T,) LSL timestamps, one per sample
    markers: list[tuple[float, int]]  # (timestamp_lsl_s, label)
    fs: float
    ch_names: list[str]
    mi_s: float  # MI trial length from metadata.json (default 4.0 s)


def _iter_preprocessed_sessions(
    data_dir: str | os.PathLike | None,
    bandpass: tuple[float, float],
    notch_hz: float | None,
    reference: str,
):
    """Load every ``*/recording.npz`` and yield it preprocessed, in session order.

    Centralises the per-session work shared by :func:`load_dataset` and
    :func:`load_erd_epochs`: read the npz, apply the same notch → bandpass → CAR
    :class:`Preprocessor` as the live service, and read the MI trial length from
    ``metadata.json``. All sessions must share the same sampling rate and channel
    order; a mismatch raises ``ValueError``.
    """
    root = Path(data_dir) if data_dir is not None else find_data_dir()
    session_dirs = sorted(p.parent for p in root.glob("*/recording.npz"))
    if not session_dirs:
        raise FileNotFoundError(f"No recording.npz found under {root}.")

    fs_ref: float | None = None
    ch_ref: list[str] | None = None
    for s_idx, sdir in enumerate(session_dirs):
        rec = np.load(sdir / "recording.npz", allow_pickle=True)
        fs = float(rec["fs"])
        ch_names = [str(c) for c in rec["channel_order"]]
        if fs_ref is None:
            fs_ref, ch_ref = fs, ch_names
        elif fs != fs_ref or ch_names != ch_ref:
            raise ValueError(f"Sampling rate or channel order differs in {sdir.name}")

        meta = sdir / "metadata.json"
        mi_s = float(json.loads(meta.read_text()).get("mi_s", 4.0)) if meta.exists() else 4.0
        filtered = Preprocessor(
            PreprocessingConfig(
                fs=fs, channel_order=ch_names, bandpass=bandpass,
                notch_hz=notch_hz, reference=reference,
            )
        ).transform(rec["raw"].astype(np.float64))
        markers = list(zip(
            rec["marker_times"].astype(np.float64).tolist(),
            rec["marker_labels"].astype(int).tolist(),
        ))
        yield _RawSession(
            s_idx=s_idx, name=sdir.name, filtered=filtered,
            timestamps=rec["timestamps"].astype(np.float64), markers=markers,
            fs=fs, ch_names=ch_names, mi_s=mi_s,
        )


def load_dataset(
    data_dir: str | os.PathLike | None = None,
    reference: str = "auto",
    bandpass: tuple[float, float] = (8.0, 30.0),
    notch_hz: float | None = 50.0,
) -> Dataset:
    """Load all sessions, preprocess, and extract MI epochs.

    Applies notch → bandpass → CAR per session (using the same
    :class:`Preprocessor` as the live service), then extracts ``mi_s``-second
    trials at each marker. Markers are placed at MI onset, so ``onset_s=0``.
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_sess: list[np.ndarray] = []
    session_ids: list[str] = []
    fs_ref: float | None = None
    ch_ref: list[str] | None = None

    for s in _iter_preprocessed_sessions(data_dir, bandpass, notch_hz, reference):
        fs_ref, ch_ref = s.fs, s.ch_names
        X, y = extract_trials(s.filtered, s.timestamps, s.markers, s.mi_s, s.fs, onset_s=0.0)
        if X.shape[0] == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        all_sess.append(np.full(len(y), s.s_idx, dtype=np.int64))
        session_ids.append(s.name)

    if not all_X:
        raise RuntimeError("No valid trials could be extracted.")
    return Dataset(
        X=np.concatenate(all_X),
        y=np.concatenate(all_y),
        session=np.concatenate(all_sess),
        fs=float(fs_ref),
        ch_names=list(ch_ref),  # type: ignore[arg-type]
        session_ids=session_ids,
    )


def load_erd_epochs(
    data_dir: str | os.PathLike | None = None,
    reference: str = "auto",
    bandpass: tuple[float, float] = (8.0, 30.0),
    notch_hz: float | None = 50.0,
    baseline_window_s: tuple[float, float] = (-3.0, 0.0),
    epoch_window_s: tuple[float, float] = (-3.0, 4.0),
    skip_first_trial_per_session: bool = True,
) -> ERDEpochs:
    """Load continuous baseline-to-MI epochs aligned to each MI-onset marker.

    The default epoch spans the three-second cue/rest period and four-second MI
    period (−3 to 4 s relative to MI onset). ``baseline`` and ``mi`` are also
    returned as convenient views of that continuous epoch. The first trial of each
    session is skipped by default because it is preceded by GET READY rather than a
    regular REST period.
    """
    b_start_s, b_stop_s = baseline_window_s
    if not b_start_s < b_stop_s <= 0:
        raise ValueError("baseline_window_s must satisfy start < stop <= 0")
    e_start_s, e_stop_s = epoch_window_s
    if not e_start_s <= b_start_s < b_stop_s <= 0 < e_stop_s:
        raise ValueError(
            "epoch_window_s must contain the baseline and span MI onset "
            "(epoch_start <= baseline_start < baseline_stop <= 0 < epoch_stop)"
        )

    baselines: list[np.ndarray] = []
    mi_epochs: list[np.ndarray] = []
    full_epochs: list[np.ndarray] = []
    labels: list[int] = []
    sessions: list[int] = []
    original_trials: list[int] = []
    fs_ref: float | None = None
    ch_ref: list[str] | None = None
    original_trial = 0

    for s in _iter_preprocessed_sessions(data_dir, bandpass, notch_hz, reference):
        fs_ref, ch_ref = s.fs, s.ch_names
        if e_stop_s > s.mi_s:
            raise ValueError(
                f"epoch_window_s ends at {e_stop_s:g} s, but {s.name} "
                f"contains only {s.mi_s:g} s of MI"
            )

        trial_in_session = 0
        for marker_time, label in s.markers:
            if int(label) not in _TRIAL_LABELS:
                continue
            this_original_trial = original_trial
            original_trial += 1
            is_first = trial_in_session == 0
            trial_in_session += 1
            if skip_first_trial_per_session and is_first:
                continue

            marker_i = int(np.searchsorted(s.timestamps, marker_time, side="left"))
            e0 = marker_i + int(round(e_start_s * s.fs))
            e1 = marker_i + int(round(e_stop_s * s.fs))
            b0 = marker_i + int(round(b_start_s * s.fs))
            b1 = marker_i + int(round(b_stop_s * s.fs))
            m0, m1 = marker_i, e1
            if e0 < 0 or e1 > s.filtered.shape[1]:
                continue
            full_epochs.append(s.filtered[:, e0:e1])
            baselines.append(s.filtered[:, b0:b1])
            mi_epochs.append(s.filtered[:, m0:m1])
            labels.append(int(label))
            sessions.append(s.s_idx)
            original_trials.append(this_original_trial)

    if not baselines:
        raise RuntimeError("No valid paired baseline/MI epochs could be extracted.")
    epochs = np.stack(full_epochs)
    times = np.arange(epochs.shape[-1], dtype=float) / float(fs_ref) + e_start_s
    return ERDEpochs(
        epochs=epochs,
        times=times,
        baseline=np.stack(baselines),
        mi=np.stack(mi_epochs),
        y=np.asarray(labels, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
        original_trial=np.asarray(original_trials, dtype=np.int64),
        fs=float(fs_ref),
        ch_names=list(ch_ref),  # type: ignore[arg-type]
    )


@dataclass
class ERDTimeCourses:
    """Baseline-normalized class-average ERD time courses."""

    times: np.ndarray
    values: dict[str, np.ndarray]  # band -> (class, channel, time)
    bands: dict[str, tuple[float, float]]
    channels: list[str]
    baseline_window_s: tuple[float, float]
    mi_window_s: tuple[float, float]
    n_trials: int


def compute_erd_time_courses(
    epochs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    fs: float,
    ch_names: list[str],
    bands: dict[str, tuple[float, float]],
    baseline_window_s: tuple[float, float],
    mi_window_s: tuple[float, float],
    channels: tuple[str, ...] = ("FC5", "FC6"),
    smooth_s: float = 0.25,
) -> ERDTimeCourses:
    """Compute μ/β ERD% after trial averaging and baseline normalization."""
    import mne
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import hilbert

    if not bands:
        raise ValueError("bands must contain at least one frequency range")
    plot_channels = [channel for channel in channels if channel in ch_names]
    if not plot_channels:
        raise ValueError(f"None of the requested channels are available: {channels}")

    baseline_mask = (
        (times >= baseline_window_s[0]) & (times < baseline_window_s[1])
    )
    if not baseline_mask.any():
        raise ValueError("baseline_window_s does not overlap the epoch")

    smooth_n = max(1, int(round(smooth_s * fs)))
    values: dict[str, np.ndarray] = {}
    for band_name, (low, high) in bands.items():
        if not 0 < low < high < fs / 2:
            raise ValueError(
                f"Invalid band {band_name}: {(low, high)} for sampling rate {fs:g} Hz"
            )
        signal = mne.filter.filter_data(
            epochs,
            sfreq=fs,
            l_freq=low,
            h_freq=high,
            method="fir",
            phase="zero",
            verbose=False,
        )
        power = np.abs(hilbert(signal, axis=-1)) ** 2
        power = uniform_filter1d(
            power, size=smooth_n, axis=-1, mode="reflect"
        )

        courses = np.empty((len(CLASS_NAMES), len(plot_channels), len(times)))
        for label in CLASS_NAMES:
            class_mask = labels == label
            if not class_mask.any():
                raise ValueError(f"No trials found for class {CLASS_NAMES[label]}")
            for channel_i, channel in enumerate(plot_channels):
                mean_power = power[class_mask, ch_names.index(channel)].mean(axis=0)
                reference = mean_power[baseline_mask].mean()
                courses[label, channel_i] = (
                    (mean_power - reference) / reference * 100.0
                )
        values[band_name] = courses

    return ERDTimeCourses(
        times=times,
        values=values,
        bands=dict(bands),
        channels=plot_channels,
        baseline_window_s=baseline_window_s,
        mi_window_s=mi_window_s,
        n_trials=len(labels),
    )


def load_erd_time_courses(
    data_dir: str | os.PathLike | None,
    bandpass: tuple[float, float],
    notch_hz: float | None,
    reference: str,
    baseline_window_s: tuple[float, float],
    mi_window_s: tuple[float, float],
    bands: dict[str, tuple[float, float]],
    excluded_trials: list[int] | None = None,
    skip_first_trial_per_session: bool = True,
    channels: tuple[str, ...] = ("FC5", "FC6"),
) -> ERDTimeCourses:
    """Load continuous epochs and compute ERD time courses for a notebook."""
    for band_name, (low, high) in bands.items():
        if bandpass[0] > low or bandpass[1] < high:
            raise ValueError(f"bandpass={bandpass} must include {band_name}")

    erd = load_erd_epochs(
        data_dir=data_dir,
        bandpass=bandpass,
        notch_hz=notch_hz,
        reference=reference,
        baseline_window_s=baseline_window_s,
        epoch_window_s=(baseline_window_s[0], mi_window_s[1]),
        skip_first_trial_per_session=skip_first_trial_per_session,
    )
    keep = ~np.isin(
        erd.original_trial,
        [] if excluded_trials is None else excluded_trials,
    )
    return compute_erd_time_courses(
        epochs=erd.epochs[keep],
        labels=erd.y[keep],
        times=erd.times,
        fs=erd.fs,
        ch_names=erd.ch_names,
        bands=bands,
        baseline_window_s=baseline_window_s,
        mi_window_s=mi_window_s,
        channels=channels,
    )


def plot_erd_time_courses(result: ERDTimeCourses):
    """Plot band-by-channel ERD time courses and return the figure and axes."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(result.bands),
        len(result.channels),
        figsize=(12, 3.5 * len(result.bands)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for row, band_name in enumerate(result.bands):
        courses = result.values[band_name]
        for col, channel in enumerate(result.channels):
            ax = axes[row, col]
            for label, color in (
                (LABEL_LEFT, "tab:blue"),
                (LABEL_RIGHT, "tab:red"),
            ):
                ax.plot(
                    result.times,
                    courses[label, col],
                    color=color,
                    lw=2,
                    label=f"{CLASS_NAMES[label]}-hand MI",
                )
            ax.axvspan(
                *result.baseline_window_s,
                color="0.9",
                label="baseline / cue",
            )
            ax.axvline(0, color="black", lw=1.2)
            ax.axhline(0, color="black", lw=1, ls="--")
            hemisphere = "left" if channel == "FC5" else "right"
            ax.set(
                title=f"{band_name} | {channel} ({hemisphere} hemisphere)",
                xlabel="Time from MI onset [s]",
                xlim=(
                    result.baseline_window_s[0],
                    result.mi_window_s[1],
                ),
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="best")
        axes[row, 0].set_ylabel("ERD [%]  (negative = ERD)")

    fig.suptitle("Band-power ERD Time Courses: FC5 vs. FC6")
    fig.tight_layout()
    return fig, axes
