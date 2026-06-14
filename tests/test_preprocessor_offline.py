from __future__ import annotations

import numpy as np
from scipy.signal import welch

from bci.config import PreprocessingConfig
from bci.preprocessing import Preprocessor


def _band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 4)))
    mask = (f >= lo) & (f <= hi)
    return float(np.trapezoid(p[mask], f[mask]))


def _composite_signal(fs: float = 128.0, dur_s: float = 8.0) -> np.ndarray:
    t = np.arange(int(dur_s * fs)) / fs
    # 10 Hz signal in passband + 50 Hz powerline noise (Tokyo / East Japan)
    return np.sin(2 * np.pi * 10 * t) + 0.8 * np.sin(2 * np.pi * 50 * t)


def test_notch_attenuates_50hz_in_2ch() -> None:
    fs = 128.0
    cfg = PreprocessingConfig(fs=fs, channel_order=["C3", "C4"])
    pre = Preprocessor(cfg)
    sig = _composite_signal(fs=fs)
    X = np.stack([sig, sig], axis=0)  # (2, T)
    out = pre.transform(X)

    p_50_in = _band_power(X[0], fs, 48, 52)
    p_50_out = _band_power(out[0], fs, 48, 52)
    assert p_50_out < 0.05 * p_50_in, (
        f"50Hz power not sufficiently attenuated: in={p_50_in:.3g} "
        f"out={p_50_out:.3g}"
    )

    # 10 Hz lies inside the 8-30 Hz passband and should remain.
    p_10_in = _band_power(X[0], fs, 8, 12)
    p_10_out = _band_power(out[0], fs, 8, 12)
    assert p_10_out > 0.3 * p_10_in


def test_apply_car_off_for_2ch_auto_preserves_channel_mean() -> None:
    fs = 128.0
    cfg = PreprocessingConfig(fs=fs, channel_order=["C3", "C4"])
    pre = Preprocessor(cfg)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, int(8 * fs)))
    # Add a deliberate offset that, if CAR were on, would zero-out per-sample
    X[0] += 1.5
    out = pre.transform(X)
    # Per-sample mean across channels should NOT be ~0 (CAR is off).
    sample_means = out.mean(axis=0)
    assert np.std(sample_means) > 1e-3


def test_apply_car_on_for_4ch_auto_zeros_per_sample_mean() -> None:
    fs = 128.0
    cfg = PreprocessingConfig(
        fs=fs, channel_order=["C3", "C4", "Cz", "C5"]
    )
    pre = Preprocessor(cfg)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, int(8 * fs)))
    out = pre.transform(X)
    sample_means = out.mean(axis=0)
    np.testing.assert_allclose(sample_means, 0.0, atol=1e-9)


def test_fit_is_noop() -> None:
    cfg = PreprocessingConfig(fs=128.0, channel_order=["C3", "C4"])
    pre = Preprocessor(cfg)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 256))
    sos_bp_before = pre.sos_bp.copy()
    sos_notch_before = pre.sos_notch.copy() if pre.sos_notch is not None else None
    result = pre.fit(X)
    assert result is None
    np.testing.assert_array_equal(pre.sos_bp, sos_bp_before)
    if sos_notch_before is not None:
        np.testing.assert_array_equal(pre.sos_notch, sos_notch_before)


def test_transform_suppresses_dc_offset_startup_transient() -> None:
    """A large DC offset (Emotiv-style) must not create a start-up spike.

    With CAR off (2ch auto) a zero-state bandpass over a ~4180 uV DC
    offset rings to ~2000 uV in the first samples; steady-state seeding
    keeps the whole filtered signal at the in-band scale, including the
    very first sample. This is the artefact that was falsely tripping the
    calibration amplitude sanity check on real 2-channel recordings.
    """
    fs = 128.0
    n = int(8 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    sig = 15.0 * np.sin(2 * np.pi * 10 * t) + rng.standard_normal((2, n)) * 5.0
    sig = sig + 4180.0  # realistic Emotiv DC offset; CAR off for 2ch
    cfg = PreprocessingConfig(fs=fs, channel_order=["C3", "C4"])
    out = Preprocessor(cfg).transform(sig)
    assert cfg.apply_car is False  # guard: this is the no-CAR path
    # No DC transient anywhere, including the leading samples.
    assert np.abs(out[:, : int(0.1 * fs)]).max() < 100.0
    assert np.abs(out).max() < 100.0


def test_transform_preserves_3d_shape() -> None:
    cfg = PreprocessingConfig(fs=128.0, channel_order=["C3", "C4"])
    pre = Preprocessor(cfg)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 2, 256))
    out = pre.transform(X)
    assert out.shape == X.shape


def test_explicit_reference_overrides_auto() -> None:
    fs = 128.0
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, int(4 * fs)))

    # Force CAR on for 2ch
    cfg_car = PreprocessingConfig(
        fs=fs, channel_order=["C3", "C4"], reference="car"
    )
    out_car = Preprocessor(cfg_car).transform(X)
    np.testing.assert_allclose(out_car.mean(axis=0), 0.0, atol=1e-9)

    # Force CAR off for 4ch
    cfg_none = PreprocessingConfig(
        fs=fs,
        channel_order=["C3", "C4", "Cz", "C5"],
        reference="none",
    )
    X4 = rng.standard_normal((4, int(4 * fs)))
    out_none = Preprocessor(cfg_none).transform(X4)
    assert np.std(out_none.mean(axis=0)) > 1e-3
