from __future__ import annotations

import numpy as np

# Only motor-imagery onsets are trials. Every other paradigm marker
# (10/11 = left/right cue, 98 = fixation, 99 = rest) is silently dropped.
_TRIAL_LABELS: frozenset[int] = frozenset({0, 1})


def extract_trials(
    raw: np.ndarray,
    timestamps: np.ndarray,
    markers: list[tuple[float, int]],
    trial_len_s: float,
    fs: float,
    onset_s: float = 0.0,
    reject_max_abs_uv: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice fixed-length trials out of a continuous EEG record.

    Parameters
    ----------
    raw : (n_channels, T) array
        Continuous, ideally already-filtered EEG.
    timestamps : (T,) array
        Per-sample timestamps in LSL clock seconds.
    markers : list of (t_lsl_seconds, label)
        Paradigm markers; only motor-imagery onsets (labels in {0, 1})
        are kept. Cue (10/11), fixation (98), and rest (99) markers are
        ignored.
    trial_len_s : float
        Trial duration in seconds.
    fs : float
        Sampling rate (Hz). Sanity-checked against ``timestamps``.
    onset_s : float, optional
        Seconds after the marker to start the epoch (default 0.0).
        E.g. onset_s=1.0 skips the first second after the cue.

    Returns
    -------
    X : (n_trials, n_channels, n_trial_samples) float array
    y : (n_trials,) int array of class labels
    """
    if raw.ndim != 2:
        raise ValueError(f"raw must be (n_channels, T); got shape {raw.shape}")
    if timestamps.ndim != 1 or len(timestamps) != raw.shape[1]:
        raise ValueError(
            "timestamps must be 1D and match raw.shape[1] "
            f"(raw={raw.shape}, ts={timestamps.shape})"
        )
    n_trial = int(round(trial_len_s * fs))
    if n_trial <= 0:
        raise ValueError("trial_len_s * fs must yield a positive sample count")
    onset_samples = int(round(onset_s * fs))

    trials: list[np.ndarray] = []
    labels: list[int] = []
    for t_lsl, label in markers:
        if label not in _TRIAL_LABELS:
            continue
        i0 = int(np.searchsorted(timestamps, t_lsl, side="left")) + onset_samples
        i1 = i0 + n_trial
        if i0 < 0 or i1 > raw.shape[1]:
            # Marker too close to either end: skip silently.
            continue
        trials.append(raw[:, i0:i1])
        labels.append(int(label))

    if not trials:
        return (
            np.zeros((0, raw.shape[0], n_trial), dtype=raw.dtype),
            np.zeros((0,), dtype=np.int64),
        )

    X = np.stack(trials, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def sliding_windows(
    trials: np.ndarray,
    labels: np.ndarray,
    window_samples: int,
    stride_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand each trial into overlapping windows.

    Returns ``(X, y, group)`` where ``group[i]`` is the trial index that
    produced window ``i``. The group array is what GroupKFold needs to
    keep windows from the same trial in the same fold.
    """
    if trials.ndim != 3:
        raise ValueError(
            f"trials must be (n_trials, n_channels, n_samples); got {trials.shape}"
        )
    if len(labels) != trials.shape[0]:
        raise ValueError("labels length must match trials.shape[0]")
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_samples and stride_samples must be positive")
    if window_samples > trials.shape[2]:
        raise ValueError(
            f"window_samples ({window_samples}) exceeds trial length "
            f"({trials.shape[2]})"
        )

    n_trials, n_channels, n_samples = trials.shape
    n_per_trial = 1 + (n_samples - window_samples) // stride_samples
    if n_per_trial <= 0:
        return (
            np.zeros((0, n_channels, window_samples), dtype=trials.dtype),
            np.zeros((0,), dtype=labels.dtype),
            np.zeros((0,), dtype=np.int64),
        )

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
