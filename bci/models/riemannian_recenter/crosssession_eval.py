"""Cross-session evaluation: pre-train on one session, evaluate on another.

Pre-trains the source MDM on a first session, then runs stratified 5-fold CV
on a second session.  For each fold the recentering reference (M_recenter_) is
estimated from that fold's training trials; the source MDM is never retrained.

Each fold also runs a no-pretrain baseline (fresh decoder fitted only on that
fold's session-2 training data) using the exact same split.  Comparing the two
controls for session difficulty: if pre-training helps, the pretrained accuracy
should be higher than the baseline on the same folds.

Accepts either session folders (new format) or flat .npz files (legacy):
  --source data/Pau_20260513T055406/
  --target data/Pau_20260513T061338/

Usage (from repo root):
    python -m bci.models.riemannian_recenter.crosssession_eval \\
        --source data/Pau_20260513T055406/ \\
        --target data/Pau_20260513T061338/ \\
        --subject Pau
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
from bci.models.riemannian_recenter.decoder import RiemannianRecenterDecoder  # noqa: E402
from bci.preprocessing import Preprocessor  # noqa: E402

logger = logging.getLogger("bci.riemannian_recenter.crosssession_eval")

ONSET_S = 1.0
N_SPLITS = 5


def _resolve_npz(path: Path) -> Path:
    return path / "recording.npz" if path.is_dir() else path


def _load_session(
    path: Path,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load, preprocess, and epoch one session.

    Returns
    -------
    trials    (n_trials, n_ch, n_samples)
    y_trials  (n_trials,)
    fs        sampling rate
    """
    data = np.load(_resolve_npz(path), allow_pickle=False)
    raw = data["raw"].astype(np.float64)
    timestamps = data["timestamps"].astype(np.float64)
    markers = list(zip(
        data["marker_times"].astype(np.float64).tolist(),
        data["marker_labels"].astype(np.int64).tolist(),
    ))
    fs = float(data["fs"])
    channel_order = list(data["channel_order"].astype(str))

    full_mi_s = float(cfg["calibration"]["mi_s"])
    trial_len_s = full_mi_s  # mi_s is the MI duration after the cue; onset_s is independent

    pre_cfg = build_preprocessing_config(cfg, fs=fs, channel_order=channel_order)
    filtered = Preprocessor(pre_cfg).transform(raw)

    trials, y_trials = extract_trials(
        filtered, timestamps, markers, trial_len_s, fs, onset_s=ONSET_S
    )
    return trials, y_trials, fs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="Session used for pre-training (folder or .npz)")
    parser.add_argument("--target", required=True,
                        help="Session used for evaluation (folder or .npz)")
    parser.add_argument("--subject", required=True,
                        help="Subject ID (used in log output)")
    parser.add_argument("--n_splits", type=int, default=N_SPLITS,
                        help=f"Number of CV folds on target session (default: {N_SPLITS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for StratifiedKFold (default: 42)")
    parser.add_argument("--config",
                        default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    setup_logging(REPO_ROOT / "logs", level="INFO")

    cfg = load_yaml(args.config)
    win_cfg = cfg["windowing"]

    # ── Load source session ───────────────────────────────────────────
    logger.info("Loading source session: %s", args.source)
    trials_src, y_src, fs_src = _load_session(Path(args.source), cfg)
    logger.info("Source: trials=%d  class_counts=%s  fs=%g",
                len(y_src), np.bincount(y_src).tolist(), fs_src)

    if trials_src.shape[0] == 0:
        logger.error("No trials extracted from source session.")
        return 1

    win_src = int(round(float(win_cfg["window_s"]) * fs_src))
    str_src = int(round(int(win_cfg["calib_stride_ms"]) / 1000.0 * fs_src))
    X_src, y_src_w, _ = sliding_windows(trials_src, y_src, win_src, str_src)
    logger.info("Source windows: %s", X_src.shape)

    # ── Load target session ───────────────────────────────────────────
    logger.info("Loading target session: %s", args.target)
    trials_tgt, y_tgt, fs_tgt = _load_session(Path(args.target), cfg)
    logger.info("Target: trials=%d  class_counts=%s  fs=%g",
                len(y_tgt), np.bincount(y_tgt).tolist(), fs_tgt)

    if trials_tgt.shape[0] == 0:
        logger.error("No trials extracted from target session.")
        return 1

    win_tgt = int(round(float(win_cfg["window_s"]) * fs_tgt))
    str_tgt = int(round(int(win_cfg["calib_stride_ms"]) / 1000.0 * fs_tgt))

    # ── Pre-train source MDM on session 1 (fixed for all folds) ──────
    logger.info("Pre-training source MDM on session 1...")
    decoder = RiemannianRecenterDecoder(metric="riemann", n_jobs=1)
    decoder.fit_source_subjects([X_src], [y_src_w])
    logger.info("Source MDM fitted. classes_=%s", decoder.classes_)

    # ── 5-fold stratified CV on target session ────────────────────────
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                          random_state=args.seed)
    idx = np.arange(len(y_tgt))

    fold_accs_pretrained: list[float] = []
    fold_accs_baseline:   list[float] = []

    for fold, (idx_train, idx_test) in enumerate(skf.split(idx, y_tgt), start=1):
        trials_train = trials_tgt[idx_train]
        y_train      = y_tgt[idx_train]
        trials_test  = trials_tgt[idx_test]
        y_test       = y_tgt[idx_test]

        X_train, y_train_w, _ = sliding_windows(
            trials_train, y_train, win_tgt, str_tgt
        )
        X_test, y_test_w, _ = sliding_windows(
            trials_test, y_test, win_tgt, str_tgt
        )

        # ── pretrained: source MDM fixed, only recenter to this fold ──
        decoder.fit(X_train, y_train_w)
        preds_pt = decoder.predict(X_test)
        acc_pt = float((preds_pt == y_test_w).mean())
        fold_accs_pretrained.append(acc_pt)

        # ── baseline: fresh decoder fitted only on session-2 fold data ─
        baseline = RiemannianRecenterDecoder(metric="riemann", n_jobs=1)
        baseline.fit(X_train, y_train_w)   # standalone mode: fits MDM + recenter
        preds_bl = baseline.predict(X_test)
        acc_bl = float((preds_bl == y_test_w).mean())
        fold_accs_baseline.append(acc_bl)

        logger.info(
            "Fold %d/%d — pretrained: %.3f  baseline: %.3f  "
            "(train trials=%d  test trials=%d)",
            fold, args.n_splits, acc_pt, acc_bl,
            len(idx_train), len(idx_test),
        )

    mean_pt = float(np.mean(fold_accs_pretrained))
    std_pt  = float(np.std(fold_accs_pretrained))
    mean_bl = float(np.mean(fold_accs_baseline))
    std_bl  = float(np.std(fold_accs_baseline))

    logger.info("═══════════════════════════════════════")
    logger.info("Subject   : %s", args.subject)
    logger.info("Source    : %s  (%d trials)", args.source, len(y_src))
    logger.info("Target    : %s  (%d trials)", args.target, len(y_tgt))
    logger.info("CV folds  : %d", args.n_splits)
    logger.info("───────────────────────────────────────")
    logger.info("              pretrained    baseline")
    for i, (ap, ab) in enumerate(zip(fold_accs_pretrained, fold_accs_baseline), 1):
        logger.info("  Fold %-2d     %.3f        %.3f", i, ap, ab)
    logger.info("───────────────────────────────────────")
    logger.info("  Mean        %.3f±%.3f   %.3f±%.3f",
                mean_pt, std_pt, mean_bl, std_bl)
    logger.info("═══════════════════════════════════════")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
