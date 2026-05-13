"""Pre-train a RiemannianRecenterDecoder on a single subject's recording.

The recording must have been saved by apps/service.py.  Accepts either
the new session-folder format or a legacy flat .npz:
  --npz data/Pau_20260513T055406/          (folder with recording.npz)
  --npz data/S01_20260511T103000Z.npz      (legacy flat file)

The script preprocesses the raw data identically to the calibration
pipeline, extracts fixed-length trials, and fits the Riemannian MDM on
the recentered covariances of that one subject.  The resulting decoder
has a trained source MDM but no recentering reference yet — call
decoder.fit(X_calib, y_calib) on a new subject to set that.

The pre-trained decoder is saved to:
  bci/models/riemannian_recenter/pretrained/<subject>.joblib

Usage (from repo root):
    python -m bci.models.riemannian_recenter.pretrain_single \\
        --npz data/Pau_20260513T055406/ --subject Pau
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bci.config import build_preprocessing_config, load_yaml  # noqa: E402
from bci.epoching import extract_trials, sliding_windows  # noqa: E402
from bci.logging_setup import setup_logging  # noqa: E402
from bci.models.riemannian_recenter.decoder import RiemannianRecenterDecoder  # noqa: E402
from bci.preprocessing import Preprocessor  # noqa: E402

logger = logging.getLogger("bci.riemannian_recenter.pretrain_single")

PRETRAINED_DIR = Path(__file__).resolve().parent / "pretrained"


def _resolve_npz(path: Path) -> Path:
    return path / "recording.npz" if path.is_dir() else path


def load_subject_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[float, int]], float, list[str]]:
    """Load raw recording saved by apps/service.py.

    Accepts either a session folder (containing recording.npz) or a
    direct path to a .npz file.

    Returns
    -------
    raw           (n_channels, T)
    timestamps    (T,)
    markers       list of (lsl_timestamp, label)
    fs            sampling rate
    channel_order list of channel names
    """
    data = np.load(_resolve_npz(path), allow_pickle=False)
    raw = data["raw"].astype(np.float64)
    timestamps = data["timestamps"].astype(np.float64)
    marker_times = data["marker_times"].astype(np.float64)
    marker_labels = data["marker_labels"].astype(np.int64)
    fs = float(data["fs"])
    channel_order = list(data["channel_order"].astype(str))
    markers = list(zip(marker_times.tolist(), marker_labels.tolist()))
    return raw, timestamps, markers, fs, channel_order


def pretrain_single(
    npz_path: Path,
    subject: str,
    config_path: Path,
    mi_s: float | None = None,
    out_dir: Path = PRETRAINED_DIR,
) -> Path:
    """Pre-train a RiemannianRecenterDecoder on one subject's npz file.

    Parameters
    ----------
    npz_path    : path to the .npz saved by apps/service.py
    subject     : subject ID used in the output filename
    config_path : path to the YAML config (for preprocessing params)
    mi_s        : trial length in seconds; if None uses calibration.mi_s from config
    out_dir     : directory to save the pre-trained .joblib file

    Returns
    -------
    Path to the saved .joblib file.
    """
    logger.info("Loading %s", npz_path)
    raw, timestamps, markers, fs, channel_order = load_subject_npz(npz_path)
    logger.info(
        "Loaded: raw=%s, fs=%g, channels=%s, markers=%d",
        raw.shape, fs, channel_order, len(markers),
    )

    cfg = load_yaml(str(config_path))
    onset_s = 1.0  # skip first second after cue onset (1s → 4s)
    full_mi_s = mi_s if mi_s is not None else float(cfg["calibration"]["mi_s"])
    trial_len_s = full_mi_s - onset_s  # epoch duration = 3s

    pre_cfg = build_preprocessing_config(cfg, fs=fs, channel_order=channel_order)
    pre = Preprocessor(pre_cfg)

    logger.info("Preprocessing (bandpass + notch + CAR)...")
    filtered = pre.transform(raw)

    logger.info("Extracting trials (onset=%.1fs, duration=%.1fs)...", onset_s, trial_len_s)
    trials, y_trials = extract_trials(filtered, timestamps, markers, trial_len_s, fs, onset_s=onset_s)
    logger.info(
        "Trials: shape=%s, class_counts=%s",
        trials.shape, np.bincount(y_trials).tolist(),
    )

    if trials.shape[0] == 0:
        raise RuntimeError("No trials extracted — check marker timestamps in the npz.")

    win_cfg = cfg["windowing"]
    window_samples = int(round(float(win_cfg["window_s"]) * fs))
    stride_samples = int(round(int(win_cfg["calib_stride_ms"]) / 1000.0 * fs))

    logger.info(
        "Sliding windows: window=%d samples (%.1fs), stride=%d samples",
        window_samples, float(win_cfg["window_s"]), stride_samples,
    )
    X, y, group = sliding_windows(trials, y_trials, window_samples, stride_samples)
    logger.info(
        "Windows: shape=%s, class_counts=%s, n_trials=%d",
        X.shape, np.bincount(y).tolist(), len(np.unique(group)),
    )

    logger.info("Fitting source MDM on recentered covariances...")
    decoder = RiemannianRecenterDecoder(metric="riemann", n_jobs=1)
    decoder.fit_source_subjects([X], [y])
    logger.info("Source MDM fitted. classes_=%s", decoder.classes_)

    # Training accuracy: set recentering reference to this subject's own mean,
    # then predict on the same data used for training.
    decoder.fit(X, y)
    preds = decoder.predict(X)
    train_acc = float((preds == y).mean())
    logger.info("----------------------------------------")
    logger.info("Training accuracy : %.3f  (%d/%d correct)",
                train_acc, int((preds == y).sum()), len(y))
    logger.info("Class counts      : left=%d  right=%d",
                int((y == 0).sum()), int((y == 1).sum()))
    logger.info("----------------------------------------")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}.joblib"
    joblib.dump(decoder, out_path)
    logger.info("Pre-trained decoder saved to %s", out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz", required=True,
        help="Path to the raw .npz file saved by apps/service.py",
    )
    parser.add_argument(
        "--subject", required=True,
        help="Subject ID used in the output filename (e.g. S01)",
    )
    parser.add_argument(
        "--mi_s", type=float, default=None,
        help="Trial length in seconds (default: read from config calibration.mi_s)",
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    setup_logging(REPO_ROOT / "logs", level="INFO")

    pretrain_single(
        npz_path=Path(args.npz),
        subject=args.subject,
        config_path=Path(args.config),
        mi_s=args.mi_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
