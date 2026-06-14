from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from .config import PreprocessingConfig


class Preprocessor:
    """Offline + online preprocessing with shared filter coefficients.

    The same `sos_notch` and `sos_bp` are used by `transform` (zi-less,
    one-shot) and `transform_chunk` (zi state carried across chunks),
    so calibration-time and realtime distributions match aside from the
    leading transient.
    """

    def __init__(self, cfg: PreprocessingConfig) -> None:
        self.cfg = cfg
        if cfg.notch_hz is not None:
            b, a = iirnotch(w0=cfg.notch_hz, Q=cfg.notch_q, fs=cfg.fs)
            self.sos_notch: np.ndarray | None = tf2sos(b, a)
        else:
            self.sos_notch = None
        self.sos_bp: np.ndarray = butter(
            cfg.bandpass_order,
            list(cfg.bandpass),
            btype="band",
            fs=cfg.fs,
            output="sos",
        )
        self._zi_notch: np.ndarray | None = None
        self._zi_bp: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """No-op hook reserved for future bad-channel removal or
        standardization. CSP handles spatial covariance on its own."""
        return None

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Offline one-shot filtering for calibration data.

        Accepts arrays with channels on the second-to-last axis and
        samples on the last axis (e.g. (ch, T) or (n_trials, ch, T)).

        Each cascade is seeded with steady-state initial conditions
        scaled to its first sample (``sosfilt_zi * x0``) so a large DC
        offset on the input does not produce a start-up transient. A
        zero-state ``sosfilt`` over EEG carrying an Emotiv-style ~4000 uV
        DC offset rings to hundreds of uV for the first ~0.1 s — far above
        the in-band signal scale — which a downstream amplitude sanity
        check would mistake for a bad recording. This matters most for the
        2-channel case, where CAR is off and cannot cancel the common-mode
        transient. After the warm-up the response is identical to a
        zero-state run, so the trials extracted for training are
        unaffected.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim < 2:
            raise ValueError(
                f"X must have at least 2 dims (..., ch, samples); got {arr.shape}"
            )
        if self.sos_notch is not None:
            arr = self._sosfilt_steady(self.sos_notch, arr)
        arr = self._sosfilt_steady(self.sos_bp, arr)
        if self.cfg.apply_car:
            arr = arr - arr.mean(axis=-2, keepdims=True)
        return arr

    @staticmethod
    def _sosfilt_steady(sos: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """``sosfilt`` along the last axis, seeded to the steady state for
        each lane's first sample so a DC offset yields no start-up spike."""
        orig_shape = arr.shape
        if orig_shape[-1] == 0:
            return arr
        x2 = arr.reshape(-1, orig_shape[-1])  # (n_lanes, T)
        # zi: (n_sections, n_lanes, 2) = sosfilt_zi(sos) scaled per lane by x0.
        zi = sosfilt_zi(sos)[:, None, :] * x2[None, :, 0, None]
        y2, _ = sosfilt(sos, x2, axis=-1, zi=zi)
        return y2.reshape(orig_shape)

    def reset_filter_state(self, n_channels: int) -> None:
        """Reinitialise online zi state. Call before realtime starts."""
        if self.sos_notch is not None:
            zi_n = sosfilt_zi(self.sos_notch)
            self._zi_notch = np.repeat(zi_n[None, :, :], n_channels, axis=0)
        else:
            self._zi_notch = None
        zi_b = sosfilt_zi(self.sos_bp)
        self._zi_bp = np.repeat(zi_b[None, :, :], n_channels, axis=0)

    def transform_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Online chunk filtering. chunk shape: (n_samples, n_channels)."""
        if self._zi_bp is None:
            raise RuntimeError(
                "reset_filter_state must be called before transform_chunk."
            )
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"chunk must be (n_samples, n_channels); got shape {arr.shape}"
            )
        n_channels = arr.shape[1]
        out = np.empty_like(arr)
        for ch in range(n_channels):
            x = arr[:, ch]
            if self.sos_notch is not None:
                x, self._zi_notch[ch] = sosfilt(
                    self.sos_notch, x, zi=self._zi_notch[ch]
                )
            x, self._zi_bp[ch] = sosfilt(self.sos_bp, x, zi=self._zi_bp[ch])
            out[:, ch] = x
        if self.cfg.apply_car:
            out = out - out.mean(axis=1, keepdims=True)
        return out
