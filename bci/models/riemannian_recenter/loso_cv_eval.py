"""Leave-One-Subject-Out evaluation of RiemannianRecenterDecoder with inner K-fold CV.

For each subject as the held-out subject:
  1. Pool all sessions from each source subject, then pretrain a source MDM
     with one recentering operation per subject (not per session).
  2. Run K-fold stratified CV on the held-out subject's pooled trials:
       - For each fold: fine-tune the recentering reference on the fold-train
         trials, then evaluate on the fold-val trials.
  3. Report per-fold and mean ± std accuracy for that held-out subject.

Overall mean ± std across all held-out subjects is reported at the end.

Accepts either the new session-folder format or legacy flat .npz files.
Pass the same subject ID for multiple sessions of the same subject so they
are pooled before recentering.

Usage (from repo root):
    python -m bci.models.riemannian_recenter.loso_cv_eval \\
        --npz data/Pau_20260513T055406/ data/Pau_20260513T061338/ \\
               data/yamamura_20260513T073957/ data/yamamura_20260513T075942/ \\
               data/yoshioka_20260513T065501/ data/yoshioka_20260513T071233/ \\
        --subjects Pau Pau yamamura yamamura yoshioka yoshioka

Notes
-----
* Sessions from the same subject are pooled before recentering, giving a
  better Riemannian mean estimate when both sessions share the same subject
  characteristics (e.g. recorded in the same day).
* Sliding windows are applied per fold to prevent data leakage between
  train and val splits.
* Recentering (fit) uses only the fold-train windows; labels are not used
  by the recentering step itself (it estimates the Riemannian mean of X).
* Requires at least 2 distinct subjects.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
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

logger = logging.getLogger("bci.riemannian_recenter.loso_cv_eval")

ONSET_S = 1.0
DEFAULT_N_SPLITS = 5


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _load_session(
    npz_path: Path,
    cfg: dict,
    window_samples: int,
    stride_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw, timestamps, markers, fs, channel_order = _load_npz(npz_path)
    trial_len_s = float(cfg["calibration"]["mi_s"])
    pre_cfg = build_preprocessing_config(cfg, fs=fs, channel_order=channel_order)
    filtered = Preprocessor(pre_cfg).transform(raw)
    trials, y_trials = extract_trials(
        filtered, timestamps, markers, trial_len_s, fs, onset_s=ONSET_S
    )
    X_windows, y_windows, group = sliding_windows(
        trials, y_trials, window_samples, stride_samples
    )
    return trials, y_trials, X_windows, y_windows, group


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", nargs="+", required=True,
                        help="Paths to session folders / .npz files")
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Subject ID for each --npz entry (repeat ID for multi-session subjects)")
    parser.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS,
                        help=f"Number of inner CV folds (default: {DEFAULT_N_SPLITS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for StratifiedKFold (default: 42)")
    parser.add_argument("--config",
                        default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    if len(args.npz) != len(args.subjects):
        parser.error("--npz and --subjects must have the same number of entries.")

    unique_subjects = list(dict.fromkeys(args.subjects))
    if len(unique_subjects) < 2:
        parser.error("LOSO requires at least 2 distinct subjects.")

    setup_logging(REPO_ROOT / "logs", level="INFO")

    cfg = load_yaml(args.config)
    win_cfg = cfg["windowing"]

    sessions_by_subject: dict[str, list[Path]] = defaultdict(list)
    for npz_path, sid in zip(args.npz, args.subjects):
        sessions_by_subject[sid].append(Path(npz_path))

    logger.info("LOSO-CV evaluation: %d subjects, %d inner folds",
                len(unique_subjects), args.n_splits)
    for sid in unique_subjects:
        logger.info("  %s: %d session(s)", sid, len(sessions_by_subject[sid]))

    # ── Pre-load and pool sessions per subject ────────────────────────────────
    subjects: list[dict] = []

    for sid in unique_subjects:
        session_paths = sessions_by_subject[sid]
        logger.info("Loading subject %s (%d session(s))...", sid, len(session_paths))

        all_trials:    list[np.ndarray] = []
        all_y_trials:  list[np.ndarray] = []
        all_X_windows: list[np.ndarray] = []
        all_y_windows: list[np.ndarray] = []
        window_samples = stride_samples = None

        for sess_path in session_paths:
            data = np.load(_resolve_npz(sess_path), allow_pickle=False)
            fs = float(data["fs"])
            ws = int(round(float(win_cfg["window_s"]) * fs))
            ss = int(round(int(win_cfg["calib_stride_ms"]) / 1000.0 * fs))
            del data

            if window_samples is None:
                window_samples, stride_samples = ws, ss

            trials, y_trials, X_windows, y_windows, _ = _load_session(
                sess_path, cfg, ws, ss
            )
            if trials.shape[0] == 0:
                logger.warning("  Session %s has no valid trials — skipping.", sess_path)
                continue
            logger.info("  Session %s: trials=%d  class_counts=%s  windows=%d",
                        sess_path.name if sess_path.is_dir() else sess_path.stem,
                        len(y_trials), np.bincount(y_trials).tolist(), len(y_windows))
            all_trials.append(trials)
            all_y_trials.append(y_trials)
            all_X_windows.append(X_windows)
            all_y_windows.append(y_windows)

        if not all_trials:
            logger.error("Subject %s has no valid sessions — skipping.", sid)
            continue

        trials_all   = np.concatenate(all_trials,    axis=0)
        y_trials_all = np.concatenate(all_y_trials,  axis=0)
        X_windows_all = np.concatenate(all_X_windows, axis=0)
        y_windows_all = np.concatenate(all_y_windows, axis=0)

        logger.info("  %s total: trials=%d  windows=%d", sid,
                    len(y_trials_all), len(y_windows_all))

        if len(y_trials_all) < args.n_splits:
            logger.error(
                "Subject %s has only %d trials but --n_splits=%d — skipping.",
                sid, len(y_trials_all), args.n_splits,
            )
            continue

        subjects.append({
            "id": sid,
            "trials":       trials_all,
            "y_trials":     y_trials_all,
            "X_windows":    X_windows_all,
            "y_windows":    y_windows_all,
            "window_samples": window_samples,
            "stride_samples": stride_samples,
        })

    if len(subjects) < 2:
        logger.error("Fewer than 2 valid subjects after loading — cannot run LOSO.")
        return 1

    # ── LOSO loop ─────────────────────────────────────────────────────────────
    subject_results: list[tuple[str, float, float]] = []
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for i, held_out in enumerate(subjects):
        sid = held_out["id"]
        source_subjects = [s for j, s in enumerate(subjects) if j != i]

        logger.info("═" * 55)
        logger.info("Held-out: %s  |  Source: %s",
                    sid, [s["id"] for s in source_subjects])

        # ── 1. Pretrain: one recentering per source subject (sessions pooled) ─
        Xs_source = [s["X_windows"] for s in source_subjects]
        ys_source = [s["y_windows"] for s in source_subjects]

        pretrain_decoder = RiemannianRecenterDecoder(metric="riemann", n_jobs=1)
        pretrain_decoder.fit_source_subjects(Xs_source, ys_source)
        logger.info("  Pretrain: %d source subjects, %d total windows",
                    len(source_subjects), sum(len(y) for y in ys_source))

        # ── 2. Inner K-fold CV on held-out subject (all sessions pooled) ──────
        trials   = held_out["trials"]
        y_trials = held_out["y_trials"]
        win_s    = held_out["window_samples"]
        str_s    = held_out["stride_samples"]

        fold_accs: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.arange(len(y_trials)), y_trials), start=1
        ):
            decoder = RiemannianRecenterDecoder(metric="riemann", n_jobs=1)
            decoder.mdm_     = pretrain_decoder.mdm_
            decoder.classes_ = pretrain_decoder.classes_

            X_train, y_train_w, _ = sliding_windows(
                trials[train_idx], y_trials[train_idx], win_s, str_s
            )
            decoder.fit(X_train, y_train_w)

            X_val, y_val_w, _ = sliding_windows(
                trials[val_idx], y_trials[val_idx], win_s, str_s
            )
            preds = decoder.predict(X_val)
            acc = float((preds == y_val_w).mean())
            fold_accs.append(acc)

            logger.info(
                "  Fold %d/%d — val acc: %.3f  "
                "(train trials=%d → %d windows, val trials=%d → %d windows)",
                fold, args.n_splits, acc,
                len(train_idx), len(X_train),
                len(val_idx), len(X_val),
            )

        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs))
        subject_results.append((sid, mean_acc, std_acc))
        logger.info("  %s CV mean: %.3f ± %.3f  per-fold: %s",
                    sid, mean_acc, std_acc, [f"{a:.3f}" for a in fold_accs])

    # ── Summary ───────────────────────────────────────────────────────────────
    if not subject_results:
        logger.error("No subjects were successfully evaluated.")
        return 1

    all_means    = np.array([m for _, m, _ in subject_results])
    overall_mean = float(all_means.mean())
    overall_std  = float(all_means.std())

    logger.info("═" * 55)
    logger.info("LOSO-CV Summary  (%d-fold inner CV, sessions pooled per subject)",
                args.n_splits)
    logger.info("─" * 55)
    for s_id, mean_acc, std_acc in subject_results:
        logger.info("  %-14s  %.3f ± %.3f", s_id, mean_acc, std_acc)
    logger.info("─" * 55)
    logger.info("  Overall         %.3f ± %.3f  (mean of subject means)", overall_mean, overall_std)
    logger.info("═" * 55)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
