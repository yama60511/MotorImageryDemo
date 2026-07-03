from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PreprocessingConfig:
    fs: float
    channel_order: list[str]
    bandpass: tuple[float, float] = (8.0, 30.0)
    bandpass_order: int = 4
    notch_hz: float | None = 50.0
    notch_q: float = 30.0
    reference: str = "auto"  # "auto" | "car" | "none"

    def __post_init__(self) -> None:
        if self.reference not in {"auto", "car", "none"}:
            raise ValueError(f"reference must be auto/car/none, got {self.reference!r}")
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        if not self.channel_order:
            raise ValueError("channel_order must be non-empty")
        lo, hi = self.bandpass
        if not (0 < lo < hi < self.fs / 2):
            raise ValueError(f"bandpass {self.bandpass} invalid for fs={self.fs}")

    @property
    def n_channels(self) -> int:
        return len(self.channel_order)

    @property
    def apply_car(self) -> bool:
        if self.reference == "car":
            return True
        if self.reference == "none":
            return False
        # "auto": CAR only when we have >= 4 channels
        return self.n_channels >= 4


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping at top level")
    return data


def build_preprocessing_config(
    d: dict[str, Any],
    fs: float,
    channel_order: list[str],
) -> PreprocessingConfig:
    pre = d.get("preprocessing", {})
    bp = pre.get("bandpass", [8.0, 30.0])
    return PreprocessingConfig(
        fs=fs,
        channel_order=list(channel_order),
        bandpass=(float(bp[0]), float(bp[1])),
        bandpass_order=int(pre.get("bandpass_order", 4)),
        notch_hz=pre.get("notch_hz", 50.0),
        notch_q=float(pre.get("notch_q", 30.0)),
        reference=str(pre.get("reference", "auto")),
    )
