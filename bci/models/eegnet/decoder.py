from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

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
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


if nn is not None:

    class _EEGNetModule(nn.Module):
        """Compact EEGNet-style network for motor imagery windows."""

        def __init__(
            self,
            n_channels: int,
            n_samples: int,
            n_classes: int,
            *,
            F1: int,
            D: int,
            F2: int,
            kernel_length: int,
            dropout: float,
        ) -> None:
            super().__init__()
            temporal_padding = kernel_length // 2
            separable_kernel = 16
            separable_padding = separable_kernel // 2

            self.features = nn.Sequential(
                nn.Conv2d(
                    1,
                    F1,
                    kernel_size=(1, kernel_length),
                    padding=(0, temporal_padding),
                    bias=False,
                ),
                nn.BatchNorm2d(F1),
                nn.Conv2d(
                    F1,
                    F1 * D,
                    kernel_size=(n_channels, 1),
                    groups=F1,
                    bias=False,
                ),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(
                    F1 * D,
                    F1 * D,
                    kernel_size=(1, separable_kernel),
                    padding=(0, separable_padding),
                    groups=F1 * D,
                    bias=False,
                ),
                nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(dropout),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_samples)
                n_features = int(np.prod(self.features(dummy).shape[1:]))
            self.classifier = nn.Linear(n_features, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = torch.flatten(x, start_dim=1)
            return self.classifier(x)


@register("eegnet")
class EEGNetDecoder(BaseDecoder):
    """EEGNet decoder with the project BaseDecoder interface.

    Input arrays use the same shape as the other decoders:
    (n_trials, n_channels, n_samples).
    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,
        device: str = "auto",
        random_state: int | None = 0,
        verbose: bool = False,
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.F1 = int(F1)
        self.D = int(D)
        self.F2 = int(F2) if F2 is not None else int(F1) * int(D)
        self.kernel_length = int(kernel_length)
        self.device = device
        self.random_state = random_state
        self.verbose = bool(verbose)

        self.model_: Any | None = None
        self.classes_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_channels_: int | None = None
        self.n_samples_: int | None = None

    def _require_torch(self) -> None:
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise RuntimeError(
                "EEGNetDecoder requires PyTorch. Install torch or recreate the "
                "conda environment from environment.yml."
            )

    def _resolved_device(self) -> Any:
        self._require_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @staticmethod
    def _check_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(
                f"X must be (n_trials, n_channels, n_samples); got shape {X.shape}"
            )
        return X.astype(np.float32, copy=False)

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True)
        self.std_[self.std_ < 1e-6] = 1.0
        return (X - self.mean_) / self.std_

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("EEGNetDecoder.fit must be called before inference.")
        return (X - self.mean_) / self.std_

    def _as_tensor(self, X: np.ndarray, device: Any) -> Any:
        return torch.from_numpy(X[:, np.newaxis, :, :]).to(device=device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetDecoder":
        self._require_torch()
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y must be a 1D array with {X.shape[0]} labels; got shape {y.shape}"
            )
        classes, y_encoded = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("EEGNetDecoder requires at least two classes.")

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            np.random.seed(int(self.random_state))

        n_trials, n_channels, n_samples = X.shape
        self.classes_ = classes
        self.n_channels_ = n_channels
        self.n_samples_ = n_samples
        X = self._standardize_fit(X)

        device = self._resolved_device()
        model = _EEGNetModule(
            n_channels,
            n_samples,
            len(classes),
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            kernel_length=min(self.kernel_length, n_samples),
            dropout=self.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        dataset = TensorDataset(
            self._as_tensor(X, device=torch.device("cpu")),
            torch.from_numpy(y_encoded.astype(np.int64)),
        )
        loader = DataLoader(
            dataset,
            batch_size=max(1, min(self.batch_size, n_trials)),
            shuffle=True,
        )

        model.train()
        for epoch in range(max(1, self.epochs)):
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(device=device)
                yb = yb.to(device=device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * xb.shape[0]
            if self.verbose:
                mean_loss = total_loss / n_trials
                print(f"EEGNet epoch {epoch + 1}/{self.epochs}: loss={mean_loss:.4f}")

        self.model_ = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_torch()
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("EEGNetDecoder.fit must be called before predict_proba.")
        X = self._check_X(X)
        if X.shape[1:] != (self.n_channels_, self.n_samples_):
            raise ValueError(
                "X must have the same channel/sample dimensions used in fit; "
                f"got {X.shape[1:]}, expected {(self.n_channels_, self.n_samples_)}"
            )
        X = self._standardize(X)
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(self._as_tensor(X, device=device))
            proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return proba.astype(np.float64, copy=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("EEGNetDecoder.fit must be called before predict.")
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    def get_params(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "F1": self.F1,
            "D": self.D,
            "F2": self.F2,
            "kernel_length": self.kernel_length,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            if key not in self.get_params():
                raise KeyError(f"unknown EEGNetDecoder parameter {key!r}")
            setattr(self, key, value)
        self.epochs = int(self.epochs)
        self.batch_size = int(self.batch_size)
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.dropout = float(self.dropout)
        self.F1 = int(self.F1)
        self.D = int(self.D)
        self.F2 = int(self.F2)
        self.kernel_length = int(self.kernel_length)
        self.verbose = bool(self.verbose)
        self.model_ = None
        self.classes_ = None
        self.mean_ = None
        self.std_ = None
        self.n_channels_ = None
        self.n_samples_ = None
