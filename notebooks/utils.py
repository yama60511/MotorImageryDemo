from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np

LABEL_NAMES = {0: "left", 1: "right", 99: "rest"}


def _default_data_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        Path(os.environ["MI_DATA_DIR"]) if "MI_DATA_DIR" in os.environ else None,
        here.parent / "data",
        here / "data",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return candidates[1]


def _iter_recordings(data_root: Path) -> list[Path]:
    if (data_root / "recording.npz").exists():
        return [data_root / "recording.npz"]
    return sorted(data_root.rglob("recording.npz"))


def _load_recording(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        raw = data["raw"].astype(np.float64)
        timestamps = data["timestamps"].astype(np.float64)
        meta: dict[str, object] = {
            "marker_times": data["marker_times"].astype(np.float64),
            "marker_labels": data["marker_labels"].astype(np.int64),
            "fs": float(data["fs"]),
            "channel_order": [str(x) for x in data["channel_order"].tolist()],
        }
    return raw, timestamps, meta


def load_raw_eeg(data_dir: str | Path | None = None) -> mne.io.RawArray:
    """Load and concatenate all session recordings into one Raw object."""
    data_root = Path(data_dir) if data_dir is not None else _default_data_root()
    recording_paths = _iter_recordings(data_root)
    if not recording_paths:
        raise FileNotFoundError(f"No recording.npz files found under {data_root}")

    raws: list[np.ndarray] = []
    annotations: list[dict[str, float | str]] = []
    channel_order: list[str] | None = None
    fs: float | None = None
    sample_offset = 0

    for rec_path in recording_paths:
        raw_uV, timestamps, meta = _load_recording(rec_path)
        fs_i = float(meta["fs"])
        channels_i = list(meta["channel_order"])
        if fs is None:
            fs = fs_i
        elif not np.isclose(fs, fs_i):
            raise ValueError(f"Sampling rate mismatch in {rec_path}: {fs_i} vs {fs}")
        if channel_order is None:
            channel_order = channels_i
        elif channels_i != channel_order:
            raise ValueError(f"Channel order mismatch in {rec_path}")

        raws.append(raw_uV * 1e-6)

        marker_times = np.asarray(meta["marker_times"], dtype=np.float64)
        marker_labels = np.asarray(meta["marker_labels"], dtype=np.int64)
        if len(timestamps) == 0:
            continue
        t0 = float(timestamps[0])
        for t, label in zip(marker_times, marker_labels):
            onset_samples = int(np.searchsorted(timestamps, float(t), side="left"))
            onset = sample_offset / fs + onset_samples / fs
            duration = 0.0
            annotations.append(
                {
                    "onset": onset,
                    "duration": duration,
                    "description": LABEL_NAMES.get(int(label), str(int(label))),
                }
            )
        sample_offset += raw_uV.shape[1]

    assert fs is not None and channel_order is not None
    data = np.concatenate(raws, axis=1) if len(raws) > 1 else raws[0]
    info = mne.create_info(ch_names=channel_order, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if annotations:
        raw.set_annotations(
            mne.Annotations(
                onset=[a["onset"] for a in annotations],
                duration=[a["duration"] for a in annotations],
                description=[a["description"] for a in annotations],
            )
        )
    return raw


def print_raw_summary(raw: mne.io.BaseRaw) -> None:
    """Print a compact summary of the loaded recording."""
    n_samples = raw.n_times
    duration_s = n_samples / raw.info["sfreq"]
    print(
        f"Raw EEG: {len(raw.ch_names)} ch, {n_samples} samples, "
        f"{duration_s:.1f} s, fs={raw.info['sfreq']:.1f} Hz"
    )
    if raw.annotations is not None and len(raw.annotations) > 0:
        counts = {}
        for desc in raw.annotations.description:
            counts[desc] = counts.get(desc, 0) + 1
        print("Annotations:", counts)
