"""End-to-end calibration on synthetic data (Phase 2.6).

Generates a continuous 2ch EEG stream with embedded left/right MI
patterns, runs the full calibration pipeline (preprocess -> epoch ->
sliding windows -> trial-level GroupKFold CV -> fit), and saves both
the raw npz and the resulting bundle.

Run from the repo root:
    python scripts/dev_calibrate_synth.py [--subject A] [--seed 0]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bci.bundle import (  # noqa: E402
    Bundle,
    make_metadata,
    make_recording_metadata,
    make_session_id,
    save_bundle,
    save_session_recording,
)
from bci.config import (  # noqa: E402
    PreprocessingConfig,
    build_preprocessing_config,
    load_yaml,
)
from bci.epoching import extract_trials, sliding_windows  # noqa: E402
from bci.logging_setup import setup_logging  # noqa: E402
from bci.models.csp_lda import CSPLDADecoder  # noqa: E402
from bci.preprocessing import Preprocessor  # noqa: E402
from tests._synth import generate_continuous_mi  # noqa: E402

logger = logging.getLogger("bci.dev_calibrate_synth")


def _make_markers(
    n_trials_per_class: int,
    cue_s: float,
    trial_len_s: float,
    rest_s: float,
    pre_s: float,
    t0: float,
    seed: int,
) -> list[tuple[float, int]]:
    rng = np.random.default_rng(seed)
    labels = np.concatenate(
        [
            np.zeros(n_trials_per_class, dtype=np.int64),
            np.ones(n_trials_per_class, dtype=np.int64),
        ]
    )
    rng.shuffle(labels)
    interval = cue_s + trial_len_s + rest_s
    markers: list[tuple[float, int]] = []
    for i, label in enumerate(labels):
        t_marker = t0 + pre_s + cue_s + i * interval
        markers.append((float(t_marker), int(label)))
    return markers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="synth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    setup_logging(REPO_ROOT / "logs", level="INFO")
    logger.info("Loading config from %s", args.config)
    cfg = load_yaml(args.config)

    fs = 128.0
    n_channels = 2
    channel_order = ["C3", "C4"]
    pre_cfg: PreprocessingConfig = build_preprocessing_config(
        cfg, fs=fs, channel_order=channel_order
    )

    cal = cfg["calibration"]
    win_cfg = cfg["windowing"]
    cv_cfg = cfg["cv"]

    n_per_class = int(cal["trials_per_class"])
    cue_s = float(cal["cue_s"])
    mi_s = float(cal["mi_s"])
    rest_s = float(cal["rest_s"])
    window_s = float(win_cfg["window_s"])
    calib_stride_ms = int(win_cfg["calib_stride_ms"])
    n_splits = int(cv_cfg["n_splits"])

    window_samples = int(round(window_s * fs))
    stride_samples = int(round(calib_stride_ms / 1000.0 * fs))

    logger.info(
        "Generating synthetic continuous EEG: %d trials/class, fs=%g, "
        "n_channels=%d",
        n_per_class,
        fs,
        n_channels,
    )
    markers = _make_markers(
        n_trials_per_class=n_per_class,
        cue_s=cue_s,
        trial_len_s=mi_s,
        rest_s=rest_s,
        pre_s=1.0,
        t0=1000.0,
        seed=args.seed,
    )
    raw, timestamps = generate_continuous_mi(
        markers=markers,
        fs=fs,
        trial_len_s=mi_s,
        pre_s=1.0,
        post_s=2.0,
        n_channels=n_channels,
        seed=args.seed,
    )
    logger.info(
        "raw shape=%s, timestamps shape=%s, marker count=%d",
        raw.shape,
        timestamps.shape,
        len(markers),
    )

    session_id = make_session_id(args.subject)
    data_root = REPO_ROOT / str(cfg["paths"]["data_dir"])
    if data_root.name.lower() != "eeg":
        data_root = data_root / "eeg"
    recording_md = make_recording_metadata(
        session_id=session_id,
        subject=args.subject,
        recorded_at=datetime.now(timezone.utc),
        fs=fs,
        channel_order=channel_order,
        n_trials_per_class=n_per_class,
        n_total_trials=len(markers),
        cue_s=cue_s,
        mi_s=mi_s,
        rest_s=rest_s,
        extra={
            "source": "synthetic",
            "synth_seed": args.seed,
            "emotiv_total_channels": 7,
            "emotiv_channel_names": ["FC3", "FC4"],
        },
    )
    data_path = save_session_recording(
        data_root,
        session_id,
        raw=raw,
        timestamps=timestamps,
        marker_times=np.array([m[0] for m in markers], dtype=np.float64),
        marker_labels=np.array([m[1] for m in markers], dtype=np.int64),
        metadata=recording_md,
    )
    logger.info("Saved raw to %s", data_path)

    pre = Preprocessor(pre_cfg)
    filtered = pre.transform(raw)

    trials, labels = extract_trials(filtered, timestamps, markers, mi_s, fs)
    logger.info("Extracted trials: shape=%s, labels=%s", trials.shape, np.bincount(labels))

    X, y, group = sliding_windows(
        trials, labels, window_samples=window_samples, stride_samples=stride_samples
    )
    logger.info(
        "Sliding windows: X=%s, y=%s, n_groups=%d",
        X.shape,
        y.shape,
        len(np.unique(group)),
    )

    splitter = GroupKFold(n_splits=n_splits)
    fold_accs: list[float] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups=group), start=1):
        groups_train = set(group[tr].tolist())
        groups_test = set(group[te].tolist())
        assert groups_train.isdisjoint(groups_test), (
            "GroupKFold leaked: same trial id in train and test"
        )
        decoder_k = CSPLDADecoder(n_components=4).fit(X[tr], y[tr])
        pred = decoder_k.predict(X[te])
        acc = float((pred == y[te]).mean())
        fold_accs.append(acc)
        all_true.extend(y[te].astype(int).tolist())
        all_pred.extend(pred.astype(int).tolist())
        logger.info("fold %d: acc=%.3f", k, acc)

    cv_acc = float(np.mean(fold_accs))
    cv_confusion = confusion_matrix(all_true, all_pred, labels=[0, 1]).astype(int)
    logger.info("trial-level GroupKFold acc = %.3f (folds: %s)", cv_acc, fold_accs)

    final_decoder = CSPLDADecoder(n_components=4).fit(X, y)
    bundle = Bundle(
        decoder=final_decoder,
        decoder_name="csp_lda",
        decoder_params=final_decoder.get_params(),
        preprocessing=pre_cfg,
        label_map={0: "left", 1: "right"},
        window_samples=window_samples,
        stride_samples=stride_samples,
        smoother_name="passthrough",
        smoother_params={},
        metadata=make_metadata(
            subject=args.subject,
            cal_acc=cv_acc,
            raw_data_path=data_path,
            trial_len_s=mi_s,
            cue_s=cue_s,
            mi_s=mi_s,
            rest_s=rest_s,
            n_trials_per_class=n_per_class,
            fs=fs,
            emotiv_n_channels=n_channels,
            emotiv_total_channels=7,
            emotiv_channel_names=["FC3", "FC4"],
            physical_electrodes=["FC3", "FC4"],
            logical_channels=channel_order,
            extra={
                "session_id": session_id,
                "decoder_name": "csp_lda",
                "decoder_params": final_decoder.get_params(),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "cv_acc": cv_acc,
                "cv_folds": fold_accs,
                "cv_confusion": cv_confusion.tolist(),
                "trials_used": int(trials.shape[0]),
                "source": "synthetic",
                "synth_seed": args.seed,
            },
        ),
    )
    bundle_dir = REPO_ROOT / str(cfg["paths"]["bundles_dir"]) / session_id / "csp_lda"
    bundle_path = save_bundle(bundle, bundle_dir)
    logger.info("Saved bundle to %s", bundle_path)

    if cv_acc < 0.65:
        logger.error("CV acc %.3f below 0.65 threshold", cv_acc)
        return 1
    logger.info("OK: CV acc %.3f >= 0.65", cv_acc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
