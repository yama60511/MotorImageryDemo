"""Evaluate ADCSPMFDecoder on a single subject with 5-fold cross-validation.

Loads a raw recording saved by apps/service.py, epochs from 1s to 4s after
each cue marker, then runs stratified 5-fold CV at trial level (sliding
windows are applied per fold, after splitting, to avoid data leakage).

Accepts either the new session-folder format or a legacy flat .npz:
  --npz data/Pau_20260513T055406/          (folder with recording.npz)
  --npz data/S01_20260511T103000Z.npz      (legacy flat file)

Usage (from repo root):
    python -m bci.models.adcsp_mf.pretrain_eval \\
        --npz data/Pau_20260513T055406/ --subject Pau
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bci.config import build_preprocessing_config, load_yaml  # noqa: E402
from bci.epoching import extract_trials, sliding_windows  # noqa: E402
from bci.logging_setup import setup_logging  # noqa: E402
from bci.models.adcsp_mf.decoder import ADCSPMFDecoder  # noqa: E402
from bci.preprocessing import Preprocessor  # noqa: E402

logger = logging.getLogger("bci.adcsp_mf.pretrain_eval")

ONSET_S = 1.0   # skip first second after cue (avoids early evoked response)
N_SPLITS = 5


def _resolve_npz(path: Path) -> Path:
    return path / "recording.npz" if path.is_dir() else path


def _load_npz(path: Path):
    data = np.load(_resolve_npz(path), allow_pickle=False)
    raw = data["raw"].astype(np.float64)
    timestamps = data["timestamps"].astype(np.float64)
    markers = list(zip(
        data["marker_times"].astype(np.float64).tolist(),
        data["marker_labels"].astype(np.int64).tolist(),
    ))
    fs = float(data["fs"])
    channel_order = list(data["channel_order"].astype(str))
    return raw, timestamps, markers, fs, channel_order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True,
                        help="Path to .npz saved by apps/service.py")
    parser.add_argument("--subject", required=True,
                        help="Subject ID (used in log output)")
    parser.add_argument("--n_splits", type=int, default=N_SPLITS,
                        help=f"Number of CV folds (default: {N_SPLITS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for StratifiedKFold (default: 42)")
    parser.add_argument("--config",
                        default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    setup_logging(REPO_ROOT / "logs", level="INFO")

    # ── Load ──────────────────────────────────────────────────────────
    logger.info("Loading %s", args.npz)
    raw, timestamps, markers, fs, channel_order = _load_npz(Path(args.npz))
    logger.info("raw=%s  fs=%g  channels=%s  markers=%d",
                raw.shape, fs, channel_order, len(markers))

    cfg = load_yaml(args.config)
    full_mi_s = float(cfg["calibration"]["mi_s"])
    trial_len_s = full_mi_s - ONSET_S  # 1s→4s = 3s duration

    # ── Preprocess ────────────────────────────────────────────────────
    pre_cfg = build_preprocessing_config(cfg, fs=fs, channel_order=channel_order)
    pre = Preprocessor(pre_cfg)
    logger.info("Preprocessing (bandpass + notch + CAR)...")
    filtered = pre.transform(raw)

    # ── Epoch (1s → 4s after cue) ─────────────────────────────────────
    logger.info("Extracting trials (onset=%.1fs, duration=%.1fs)...",
                ONSET_S, trial_len_s)
    trials, y_trials = extract_trials(
        filtered, timestamps, markers, trial_len_s, fs, onset_s=ONSET_S
    )
    logger.info("Trials: shape=%s  class_counts=%s",
                trials.shape, np.bincount(y_trials).tolist())

    if trials.shape[0] == 0:
        logger.error("No trials extracted — check marker timestamps.")
        return 1

    # ── Sliding window config ─────────────────────────────────────────
    win_cfg = cfg["windowing"]
    window_samples = int(round(float(win_cfg["window_s"]) * fs))
    stride_samples = int(round(int(win_cfg["calib_stride_ms"]) / 1000.0 * fs))
    logger.info("Sliding windows: window=%d samples (%.1fs), stride=%d samples",
                window_samples, float(win_cfg["window_s"]), stride_samples)

    # ── 5-fold stratified CV at trial level ───────────────────────────
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                          random_state=args.seed)
    idx = np.arange(len(y_trials))

    fold_accs: list[float] = []
    for fold, (idx_train, idx_test) in enumerate(skf.split(idx, y_trials), start=1):
        trials_train = trials[idx_train]
        y_train      = y_trials[idx_train]
        trials_test  = trials[idx_test]
        y_test       = y_trials[idx_test]

        X_train, y_train_w, _ = sliding_windows(
            trials_train, y_train, window_samples, stride_samples
        )
        X_test, y_test_w, _ = sliding_windows(
            trials_test, y_test, window_samples, stride_samples
        )

        decoder = ADCSPMFDecoder()
        decoder.fit(X_train, y_train_w)

        preds = decoder.predict(X_test)
        acc = float((preds == y_test_w).mean())
        fold_accs.append(acc)
        logger.info(
            "Fold %d/%d — test acc: %.3f  (%d/%d windows)  "
            "train trials=%d  test trials=%d",
            fold, args.n_splits, acc,
            int((preds == y_test_w).sum()), len(y_test_w),
            len(idx_train), len(idx_test),
        )

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))

    logger.info("═══════════════════════════════════════")
    logger.info("Subject  : %s", args.subject)
    logger.info("CV folds : %d", args.n_splits)
    logger.info("Per-fold : %s", [f"{a:.3f}" for a in fold_accs])
    logger.info("Mean acc : %.3f ± %.3f", mean_acc, std_acc)
    logger.info("═══════════════════════════════════════")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
