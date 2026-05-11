"""Leave-One-Subject-Out evaluation of RiemannianRecenterDecoder.

For each subject as the held-out subject:
  1. Pre-train a source MDM on the other N-1 subjects (covariances recentered
     per-subject to the identity before pooling).
  2. Estimate the held-out subject's recentering reference from a small
     calibration set (first --calib_trials trials, chronological order).
  3. Test on the remaining trials of the held-out subject.

Accuracy is reported per subject and as mean ± std across all subjects.

Usage (from repo root):
    python -m bci.models.riemannian_recenter.loso_eval \\
        --npz  data/S01_....npz data/S02_....npz data/S03_....npz \\
        --subjects S01 S02 S03

Notes
-----
* Calibration uses the chronologically first --calib_trials trials (realistic
  BCI scenario: collect calibration block, then run free online).
* Sliding windows are applied per-split so there is no leakage across subjects
  or between calibration and test portions.
* Requires at least 2 subjects.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bci.config import build_preprocessing_config, load_yaml  # noqa: E402
from bci.epoching import extract_trials, sliding_windows  # noqa: E402
from bci.logging_setup import setup_logging  # noqa: E402
from bci.models.riemannian_recenter.decoder import RiemannianRecenterDecoder  # noqa: E402
from bci.preprocessing import Preprocessor  # noqa: E402

logger = logging.getLogger("bci.riemannian_recenter.loso_eval")

ONSET_S = 1.0        # skip first second after cue
DEFAULT_CALIB = 10   # calibration trials for the held-out subject


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_npz(path: Path):
    data = np.load(path, allow_pickle=False)
    raw = data["raw"].astype(np.float64)
    timestamps = data["timestamps"].astype(np.float64)
    markers = list(zip(
        data["marker_times"].astype(np.float64).tolist(),
        data["marker_labels"].astype(np.int64).tolist(),
    ))
    fs = float(data["fs"])
    channel_order = list(data["channel_order"].astype(str))
    return raw, timestamps, markers, fs, channel_order


def _load_subject(
    npz_path: Path,
    cfg: dict,
    window_samples: int,
    stride_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, preprocess, epoch, and window one subject.

    Returns
    -------
    trials        (n_trials, n_ch, n_samples)
    y_trials      (n_trials,)
    X_windows     (n_windows, n_ch, window_samples)
    y_windows     (n_windows,)
    group         (n_windows,)  — trial index per window
    """
    raw, timestamps, markers, fs, channel_order = _load_npz(npz_path)

    full_mi_s = float(cfg["calibration"]["mi_s"])
    trial_len_s = full_mi_s - ONSET_S

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
                        help="Paths to per-subject .npz files (one per subject)")
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Subject IDs matching --npz order (e.g. S01 S02 S03)")
    parser.add_argument("--calib_trials", type=int, default=DEFAULT_CALIB,
                        help=(
                            f"Calibration trials taken from the held-out subject "
                            f"(default: {DEFAULT_CALIB})"
                        ))
    parser.add_argument("--config",
                        default=str(REPO_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    if len(args.npz) != len(args.subjects):
        parser.error("--npz and --subjects must have the same number of entries.")
    if len(args.npz) < 2:
        parser.error("LOSO requires at least 2 subjects.")

    setup_logging(REPO_ROOT / "logs", level="INFO")

    cfg = load_yaml(args.config)
    win_cfg = cfg["windowing"]
    fs_ref = None  # will be inferred per subject; logged for consistency check
    window_samples_ref: int | None = None
    stride_samples_ref: int | None = None

    n_subjects = len(args.subjects)
    logger.info("LOSO evaluation: %d subjects, calib_trials=%d",
                n_subjects, args.calib_trials)

    # ── Pre-load all subjects ─────────────────────────────────────────────────
    # We pre-load all so the inner LOSO loop doesn't re-read disk N² times.
    subjects: list[dict] = []
    for npz_path, sid in zip(args.npz, args.subjects):
        logger.info("Loading subject %s from %s ...", sid, npz_path)

        # Peek at fs to compute window/stride samples
        data = np.load(npz_path, allow_pickle=False)
        fs = float(data["fs"])
        window_samples = int(round(float(win_cfg["window_s"]) * fs))
        stride_samples = int(round(int(win_cfg["calib_stride_ms"]) / 1000.0 * fs))

        if fs_ref is None:
            fs_ref = fs
            window_samples_ref = window_samples
            stride_samples_ref = stride_samples
        elif (window_samples != window_samples_ref
              or stride_samples != stride_samples_ref):
            logger.warning(
                "Subject %s has fs=%g → window=%d stride=%d, "
                "which differs from the first subject (fs=%g → %d/%d). "
                "Proceeding anyway — subjects will be handled independently.",
                sid, fs, window_samples, stride_samples,
                fs_ref, window_samples_ref, stride_samples_ref,
            )

        del data  # free before full load

        trials, y_trials, X_windows, y_windows, group = _load_subject(
            Path(npz_path), cfg, window_samples, stride_samples
        )
        logger.info(
            "  %s: trials=%d  class_counts=%s  windows=%d",
            sid, len(y_trials), np.bincount(y_trials).tolist(), len(y_windows),
        )

        if trials.shape[0] == 0:
            logger.error("Subject %s has no valid trials — skipping.", sid)
            continue

        if trials.shape[0] <= args.calib_trials:
            logger.error(
                "Subject %s has only %d trials but --calib_trials=%d — skipping.",
                sid, trials.shape[0], args.calib_trials,
            )
            continue

        subjects.append({
            "id": sid,
            "trials": trials,
            "y_trials": y_trials,
            "X_windows": X_windows,
            "y_windows": y_windows,
            "group": group,
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

        logger.info("─" * 50)
        logger.info("Held-out: %s  |  Source subjects: %s",
                    sid, [s["id"] for s in source_subjects])

        # ── 1. Pre-train source MDM on N-1 subjects ────────────────────────
        Xs_source = [s["X_windows"] for s in source_subjects]
        ys_source = [s["y_windows"] for s in source_subjects]

        decoder = RiemannianRecenterDecoder(metric="riemann", n_jobs=1)
        decoder.fit_source_subjects(Xs_source, ys_source)

        # ── 2. Calibration: first calib_trials trials of held-out subject ──
        n_calib = args.calib_trials
        trials_calib = held_out["trials"][:n_calib]
        y_calib_t    = held_out["y_trials"][:n_calib]

        X_calib, y_calib_w, _ = sliding_windows(
            trials_calib, y_calib_t,
            held_out["window_samples"], held_out["stride_samples"],
        )
        decoder.fit(X_calib, y_calib_w)
        logger.info(
            "  Calibration: %d trials → %d windows  class_counts=%s",
            n_calib, len(y_calib_w), np.bincount(y_calib_t).tolist(),
        )

        # ── 3. Test: remaining trials of held-out subject ─────────────────
        trials_test = held_out["trials"][n_calib:]
        y_test_t    = held_out["y_trials"][n_calib:]

        X_test, y_test_w, _ = sliding_windows(
            trials_test, y_test_t,
            held_out["window_samples"], held_out["stride_samples"],
        )

        preds = decoder.predict(X_test)
        acc = float((preds == y_test_w).mean())
        subject_accs.append((sid, acc))

        logger.info(
            "  Test: %d trials → %d windows  class_counts=%s",
            len(y_test_t), len(y_test_w), np.bincount(y_test_t).tolist(),
        )
        logger.info("  Test acc: %.3f  (%d/%d windows correct)",
                    acc, int((preds == y_test_w).sum()), len(y_test_w))

    # ── Summary ───────────────────────────────────────────────────────────────
    if not subject_accs:
        logger.error("No subjects were successfully evaluated.")
        return 1

    accs = np.array([a for _, a in subject_accs])
    mean_acc = float(accs.mean())
    std_acc  = float(accs.std())

    logger.info("═" * 50)
    logger.info("LOSO Results  (calib_trials=%d)", args.calib_trials)
    logger.info("─" * 50)
    for sid, acc in subject_accs:
        logger.info("  %-6s  %.3f", sid, acc)
    logger.info("─" * 50)
    logger.info("  Mean    %.3f ± %.3f", mean_acc, std_acc)
    logger.info("═" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
