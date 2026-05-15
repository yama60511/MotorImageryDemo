"""Leave-One-Subject-Out evaluation of ADCSPMFDecoder.

For each subject as the held-out subject:
  1. Pretrain ADCSPMFDecoder on all N-1 source subjects (sessions pooled per subject).
  2. Fine-tune the MFACC classifier on --calib_trials calibration trials from the
     held-out subject (ADCSP spatial filters stay fixed).
  3. Test on the remaining held-out trials.

Multiple sessions per subject are supported: pass the same subject ID for each
session folder and the script groups them automatically.

Accepts either the new session-folder format or legacy flat .npz files:
  --npz data/Pau_20260513T055406/ data/Pau_20260513T061338/ ...

Usage (from repo root):
    python -m bci.models.adcsp_mf.loso_eval \\
        --npz data/Pau_20260513T055406/ data/Pau_20260513T061338/ \\
               data/yamamura_20260513T073957/ data/yamamura_20260513T075942/ \\
               data/yoshioka_20260513T065501/ data/yoshioka_20260513T071233/ \\
        --subjects Pau Pau yamamura yamamura yoshioka yoshioka

Notes
-----
* Calibration uses the chronologically first --calib_trials trials across the
  held-out subject's sessions (realistic BCI scenario).
* Sliding windows are applied per-split to prevent data leakage.
* Requires at least 2 distinct subjects.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bci.config import build_preprocessing_config, load_yaml  # noqa: E402
from bci.epoching import extract_trials, sliding_windows  # noqa: E402
from bci.logging_setup import setup_logging  # noqa: E402
from bci.models.adcsp_mf.decoder import ADCSPMFDecoder  # noqa: E402
from bci.preprocessing import Preprocessor  # noqa: E402

logger = logging.getLogger("bci.adcsp_mf.loso_eval")

ONSET_S = 1.0
DEFAULT_CALIB = 20


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
    """Load, preprocess, epoch, and window one recording session.

    Returns
    -------
    trials    (n_trials, n_ch, n_samples)
    y_trials  (n_trials,)
    X_windows (n_windows, n_ch, window_samples)
    y_windows (n_windows,)
    group     (n_windows,) — trial index per window
    """
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
    parser.add_argument("--calib_trials", type=int, default=DEFAULT_CALIB,
                        help=f"Calibration trials from the held-out subject (default: {DEFAULT_CALIB})")
    parser.add_argument("--config",
                        default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    if len(args.npz) != len(args.subjects):
        parser.error("--npz and --subjects must have the same number of entries.")

    unique_subjects = list(dict.fromkeys(args.subjects))  # preserve insertion order
    if len(unique_subjects) < 2:
        parser.error("LOSO requires at least 2 distinct subjects.")

    setup_logging(REPO_ROOT / "logs", level="INFO")

    cfg = load_yaml(args.config)
    win_cfg = cfg["windowing"]

    # Group session paths by subject
    sessions_by_subject: dict[str, list[Path]] = defaultdict(list)
    for npz_path, sid in zip(args.npz, args.subjects):
        sessions_by_subject[sid].append(Path(npz_path))

    logger.info("LOSO evaluation: %d subjects, calib_trials=%d",
                len(unique_subjects), args.calib_trials)
    for sid in unique_subjects:
        logger.info("  %s: %d session(s)", sid, len(sessions_by_subject[sid]))

    # ── Pre-load all subjects ─────────────────────────────────────────────────
    subjects: list[dict] = []

    for sid in unique_subjects:
        session_paths = sessions_by_subject[sid]
        logger.info("Loading subject %s (%d session(s))...", sid, len(session_paths))

        all_trials: list[np.ndarray] = []
        all_y_trials: list[np.ndarray] = []
        all_X_windows: list[np.ndarray] = []
        all_y_windows: list[np.ndarray] = []

        for sess_path in session_paths:
            data = np.load(_resolve_npz(sess_path), allow_pickle=False)
            fs = float(data["fs"])
            window_samples = int(round(float(win_cfg["window_s"]) * fs))
            stride_samples = int(round(int(win_cfg["calib_stride_ms"]) / 1000.0 * fs))
            del data

            trials, y_trials, X_windows, y_windows, _ = _load_session(
                sess_path, cfg, window_samples, stride_samples
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

        # Stack all sessions for this subject
        trials_all = np.concatenate(all_trials, axis=0)
        y_trials_all = np.concatenate(all_y_trials, axis=0)
        X_windows_all = np.concatenate(all_X_windows, axis=0)
        y_windows_all = np.concatenate(all_y_windows, axis=0)

        logger.info("  %s total: trials=%d  windows=%d", sid,
                    len(y_trials_all), len(y_windows_all))

        if len(y_trials_all) <= args.calib_trials:
            logger.error(
                "Subject %s has only %d trials but --calib_trials=%d — skipping.",
                sid, len(y_trials_all), args.calib_trials,
            )
            continue

        subjects.append({
            "id": sid,
            "trials": trials_all,
            "y_trials": y_trials_all,
            "X_windows": X_windows_all,
            "y_windows": y_windows_all,
            "window_samples": window_samples,
            "stride_samples": stride_samples,
        })

    if len(subjects) < 2:
        logger.error("Fewer than 2 valid subjects after loading — cannot run LOSO.")
        return 1

    # ── LOSO loop ─────────────────────────────────────────────────────────────
    subject_accs: list[tuple[str, float]] = []

    for i, held_out in enumerate(subjects):
        sid = held_out["id"]
        source_subjects = [s for j, s in enumerate(subjects) if j != i]

        logger.info("─" * 55)
        logger.info("Held-out: %s  |  Source: %s",
                    sid, [s["id"] for s in source_subjects])

        # ── 1. Pretrain on N-1 source subjects ────────────────────────────
        Xs_source = [s["X_windows"] for s in source_subjects]
        ys_source = [s["y_windows"] for s in source_subjects]

        decoder = ADCSPMFDecoder()
        decoder.fit_source_subjects(Xs_source, ys_source)
        logger.info("  Pretrain: %d source subjects, %d total windows",
                    len(source_subjects),
                    sum(len(y) for y in ys_source))

        # ── 2. Fine-tune MFACC on calibration trials from held-out subject ─
        n_calib = args.calib_trials
        trials_calib = held_out["trials"][:n_calib]
        y_calib_t    = held_out["y_trials"][:n_calib]

        X_calib, y_calib_w, _ = sliding_windows(
            trials_calib, y_calib_t,
            held_out["window_samples"], held_out["stride_samples"],
        )
        decoder.fit(X_calib, y_calib_w)
        logger.info("  Fine-tune: %d calib trials → %d windows  class_counts=%s",
                    n_calib, len(y_calib_w), np.bincount(y_calib_t).tolist())

        # ── 3. Test on remaining held-out trials ──────────────────────────
        trials_test = held_out["trials"][n_calib:]
        y_test_t    = held_out["y_trials"][n_calib:]

        X_test, y_test_w, _ = sliding_windows(
            trials_test, y_test_t,
            held_out["window_samples"], held_out["stride_samples"],
        )

        preds = decoder.predict(X_test)
        acc = float((preds == y_test_w).mean())
        subject_accs.append((sid, acc))

        logger.info("  Test: %d trials → %d windows  class_counts=%s",
                    len(y_test_t), len(y_test_w), np.bincount(y_test_t).tolist())
        logger.info("  Test acc: %.3f  (%d/%d windows correct)",
                    acc, int((preds == y_test_w).sum()), len(y_test_w))

    # ── Summary ───────────────────────────────────────────────────────────────
    if not subject_accs:
        logger.error("No subjects were successfully evaluated.")
        return 1

    accs = np.array([a for _, a in subject_accs])
    mean_acc = float(accs.mean())
    std_acc  = float(accs.std())

    logger.info("═" * 55)
    logger.info("LOSO Results  (calib_trials=%d)", args.calib_trials)
    logger.info("─" * 55)
    for s_id, acc in subject_accs:
        logger.info("  %-12s  %.3f", s_id, acc)
    logger.info("─" * 55)
    logger.info("  Mean          %.3f ± %.3f", mean_acc, std_acc)
    logger.info("═" * 55)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
