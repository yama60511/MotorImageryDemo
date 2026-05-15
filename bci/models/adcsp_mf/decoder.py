from __future__ import annotations

import sys
from typing import Any, Iterable
from pathlib import Path

import numpy as np

from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline, Pipeline

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from bci.models.base import BaseDecoder
    from bci.models.registry import register
else:
    from ..base import BaseDecoder
    from ..registry import register

try:
    from .mean_field_acc import MFACC
    from .mean_field_acc_utils import ADCSP
except Exception:  # pragma: no cover - optional dependency
    MFACC = None  # type: ignore
    ADCSP = None  # type: ignore


'''
Adaptive Common Spatial Pattern (ACSP) + Means Field Accumulator (MFACC)
decoder for motor imagery BCI.
'''

@register("adcsp_mf")
class ADCSPMFDecoder(BaseDecoder):

    def __init__(
        self,
        cov_estimator: str = "oas", # algorithm for covariance estimation (e.g. "oas", "lwf", "scm")
        adcsp_modes: Iterable[str] | None = None, # list of ADCSP modes to apply (e.g. ["high_electrodes_count", "low_electrodes_count"])
        mfacc_kwargs: dict[str, Any] | None = None, # keyword arguments for MFACC (e.g. {"method_label": "lda", "n_jobs": 1, "rpme_enabled": False})
    ) -> None:
        self.cov_estimator = cov_estimator
        self.adcsp_modes = (
            tuple(adcsp_modes) if adcsp_modes is not None else ("high_electrodes_count",)
        )
        self.mfacc_kwargs = dict(
            mfacc_kwargs or {"method_label": "lda", "n_jobs": 1, "rpme_enabled": False}
        )
        # Standalone mode (single-subject fit)
        self.pipeline = None
        self.classes_ = None
        # Pretrain/finetune mode (cross-subject transfer)
        # Covariances and ADCSP+MFACC are kept separate so that covariances
        # can be computed per-subject independently (subjects may have different
        # sampling rates, yielding different window lengths).
        self._cov_est_: Covariances | None = None
        self._adcsp_mf_: Pipeline | None = None
        self._source_covs_: np.ndarray | None = None
        self._source_y_: np.ndarray | None = None

    # internal method to build the sklearn pipeline based on current parameters
    def _build_pipeline(self):
        if MFACC is None or ADCSP is None:  # pragma: no cover - runtime error if missing
            raise RuntimeError("mfacc package is required for ADCSP+MF decoder")
        steps = [Covariances(estimator=self.cov_estimator)] # add covariance estimation step
        for mode in self.adcsp_modes: # add ADCSP stages for each specified mode
            steps.append(ADCSP(mode=mode))
        steps.append(MFACC(**self.mfacc_kwargs)) # add final MFACC classifier stage
        return make_pipeline(*steps)

    def _build_adcsp_mf_pipeline(self) -> Pipeline:
        """Build ADCSP+MFACC pipeline that operates on covariance matrices."""
        if MFACC is None or ADCSP is None:  # pragma: no cover
            raise RuntimeError("mfacc package is required for ADCSP+MF decoder")
        steps = [(f"adcsp_{mode}", ADCSP(mode=mode)) for mode in self.adcsp_modes]
        steps.append(("mfacc", MFACC(**self.mfacc_kwargs)))
        return Pipeline(steps)

    def _covs_through_adcsp(self, C: np.ndarray) -> np.ndarray:
        """Transform covariance matrices through all ADCSP steps (not MFACC)."""
        out = C
        for _, step in self._adcsp_mf_.steps[:-1]:
            out = step.transform(out)
        return out

    def fit_source_subjects(
        self,
        Xs: list[np.ndarray],
        ys: list[np.ndarray],
    ) -> "ADCSPMFDecoder":
        """Pre-train ADCSP+MF on pooled covariances from source subjects.

        Each subject's covariances are computed independently (so subjects with
        different sampling rates / window lengths are handled correctly).  The
        ADCSP spatial filters and MFACC are then fitted on the pooled covariances.

        A subsequent fit() call fine-tunes only the MFACC using the held-out
        subject's calibration data, keeping the ADCSP filters fixed.

        Parameters
        ----------
        Xs : list of (n_windows, n_ch, n_samples) arrays, one per subject.
        ys : list of (n_windows,) label arrays, one per subject.
        """
        if MFACC is None or ADCSP is None:  # pragma: no cover
            raise RuntimeError("mfacc package is required for ADCSP+MF decoder")

        cov_est = Covariances(estimator=self.cov_estimator)
        all_covs: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for X_s, y_s in zip(Xs, ys):
            if X_s.ndim != 3:
                raise ValueError(
                    f"Each element of Xs must be (n_windows, n_ch, n_samples); got {X_s.shape}"
                )
            C_s = cov_est.fit_transform(X_s.astype(np.float64))
            all_covs.append(C_s)
            all_y.append(y_s)

        C_pool = np.concatenate(all_covs, axis=0)
        y_pool = np.concatenate(all_y, axis=0)

        adcsp_mf = self._build_adcsp_mf_pipeline()
        adcsp_mf.fit(C_pool, y_pool)

        self._cov_est_ = cov_est
        self._adcsp_mf_ = adcsp_mf
        self.classes_ = getattr(adcsp_mf[-1], "classes_", None)

        # Cache covariances after ADCSP (before MFACC) for MFACC fine-tuning
        self._source_covs_ = self._covs_through_adcsp(C_pool)
        self._source_y_ = y_pool
        return self

    # fit the model to training data
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ADCSPMFDecoder":
        if X.ndim != 3:
            raise ValueError(
                f"X must be (n_trials, n_channels, n_samples); got shape {X.shape}"
            )
        X64 = X.astype(np.float64)

        if self._source_covs_ is not None and self._adcsp_mf_ is not None:
            # Fine-tune mode: ADCSP filters stay fixed, only MFACC is retrained
            # on combined source covariances + target calibration covariances.
            C_calib = self._cov_est_.fit_transform(X64)
            calib_covs = self._covs_through_adcsp(C_calib)
            combined_covs = np.concatenate([self._source_covs_, calib_covs], axis=0)
            combined_y = np.concatenate([self._source_y_, y], axis=0)
            new_mfacc = MFACC(**self.mfacc_kwargs)
            new_mfacc.fit(combined_covs, combined_y)
            old_steps = list(self._adcsp_mf_.steps)
            self._adcsp_mf_ = Pipeline(old_steps[:-1] + [(old_steps[-1][0], new_mfacc)])
            self.classes_ = getattr(new_mfacc, "classes_", None)
        else:
            # Standalone mode: train the full pipeline from scratch
            pipe = self._build_pipeline()
            pipe.fit(X64, y) # fit the pipeline (many pyriemann/mfacc components expect float64)
            self.pipeline = pipe
            try:
                final = pipe[-1] # capture classes from final estimator if available
                self.classes_ = getattr(final, "classes_", None) # capture classes from final estimator if available
            except Exception:
                self.classes_ = None
        return self

    # predict class labels for new data
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(
                f"X must be (n_trials, n_channels, n_samples); got shape {X.shape}"
            )
        X64 = X.astype(np.float64)
        if self._adcsp_mf_ is not None and self._cov_est_ is not None:
            # Pretrain/finetune mode
            C = self._cov_est_.fit_transform(X64)
            return np.asarray(self._adcsp_mf_.predict(C))
        if self.pipeline is None:
            raise RuntimeError("ADCSPMFDecoder.fit must be called before predict.")
        return np.asarray(self.pipeline.predict(X64))

    # predict class probabilities for new data. If the final estimator does not
    # support predict_proba, fallback to one-hot encoding of predictions.
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(
                f"X must be (n_trials, n_channels, n_samples); got shape {X.shape}"
            )
        X64 = X.astype(np.float64)

        if self._adcsp_mf_ is not None and self._cov_est_ is not None:
            # Pretrain/finetune mode
            C = self._cov_est_.fit_transform(X64)
            final = self._adcsp_mf_[-1]
            if hasattr(final, "predict_proba"):
                return np.asarray(self._adcsp_mf_.predict_proba(C))
            preds = self._adcsp_mf_.predict(C)
        elif self.pipeline is not None:
            # Standalone mode
            final = self.pipeline[-1]
            if hasattr(final, "predict_proba"):
                return np.asarray(self.pipeline.predict_proba(X64))
            preds = self.pipeline.predict(X64)
        else:
            raise RuntimeError("ADCSPMFDecoder.fit must be called before predict_proba.")

        # fallback: one-hot encode predictions when estimator has no predict_proba
        classes = self.classes_ if self.classes_ is not None else np.unique(preds)
        proba = np.zeros((len(preds), len(classes)), dtype=np.float64)
        for i, p in enumerate(preds):
            idx = int(np.where(classes == p)[0][0])
            proba[i, idx] = 1.0
        return proba

    # get current parameters as a dictionary
    def get_params(self) -> dict[str, Any]:
        return {
            "cov_estimator": self.cov_estimator,
            "adcsp_modes": list(self.adcsp_modes),
            "mfacc_kwargs": dict(self.mfacc_kwargs),
        }

    # set parameters from a dictionary and reset pipeline and classes
    def set_params(self, params: dict[str, Any]) -> None:
        if "cov_estimator" in params:
            self.cov_estimator = params["cov_estimator"]
        if "adcsp_modes" in params:
            self.adcsp_modes = tuple(params["adcsp_modes"])
        if "mfacc_kwargs" in params:
            self.mfacc_kwargs = dict(params["mfacc_kwargs"])
        self.pipeline = None
        self.classes_ = None
        self._cov_est_ = None
        self._adcsp_mf_ = None
        self._source_covs_ = None
        self._source_y_ = None
