from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

from .config import PreprocessingConfig

if TYPE_CHECKING:
    from .models.base import BaseDecoder

_TRACKED_PACKAGES = ("numpy", "scipy", "scikit-learn", "mne", "joblib", "pylsl")


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def collect_package_versions() -> dict[str, str]:
    versions = {pkg: _pkg_version(pkg) for pkg in _TRACKED_PACKAGES}
    versions["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return versions


def _git_sha(cwd: str | Path | None = None) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except (FileNotFoundError, OSError):
        pass
    return "unknown"


@dataclass
class Bundle:
    decoder: BaseDecoder
    decoder_name: str
    decoder_params: dict[str, Any]
    preprocessing: PreprocessingConfig
    label_map: dict[int, str]
    window_samples: int
    stride_samples: int
    smoother_name: str
    smoother_params: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def make_session_id(subject: str, now: datetime | None = None) -> str:
    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{subject}_{ts.strftime('%Y%m%dT%H%M%S')}"


def _read_json(path: Path) -> dict[str, Any]:
    """Return the JSON object stored at ``path``, or ``{}`` if it is absent."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def make_metadata(
    *,
    subject: str,
    cal_acc: float,
    raw_data_path: str | Path | None,
    trial_len_s: float,
    cue_s: float,
    mi_s: float,
    rest_s: float,
    n_trials_per_class: int,
    fs: float,
    emotiv_n_channels: int,
    emotiv_total_channels: int,
    emotiv_channel_names: list[str],
    physical_electrodes: list[str],
    logical_channels: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    md: dict[str, Any] = {
        "subject": subject,
        "date": datetime.now(timezone.utc).isoformat(),
        "cal_acc": float(cal_acc),
        "raw_data_path": str(raw_data_path) if raw_data_path is not None else None,
        "trial_len_s": float(trial_len_s),
        "cue_s": float(cue_s),
        "mi_s": float(mi_s),
        "rest_s": float(rest_s),
        "n_trials_per_class": int(n_trials_per_class),
        "fs": float(fs),
        "emotiv_n_channels": int(emotiv_n_channels),
        "emotiv_total_channels": int(emotiv_total_channels),
        "emotiv_channel_names": list(emotiv_channel_names),
        "physical_electrodes": list(physical_electrodes),
        "logical_channels": list(logical_channels),
        "package_versions": collect_package_versions(),
        "git_sha": _git_sha(),
    }
    if extra:
        md.update(extra)
    return md


def make_recording_metadata(
    *,
    session_id: str,
    subject: str,
    recorded_at: datetime | None,
    fs: float,
    channel_order: list[str],
    n_trials_per_class: int,
    n_total_trials: int,
    cue_s: float,
    mi_s: float,
    rest_s: float,
    fixation_s: float = 0.0,
    marker_left: int = 0,
    marker_right: int = 1,
    marker_cue_left: int = 10,
    marker_cue_right: int = 11,
    marker_fixation: int = 98,
    marker_rest: int = 99,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recorded = (recorded_at or datetime.now(timezone.utc)).astimezone(
        timezone.utc
    )
    md: dict[str, Any] = {
        "session_id": session_id,
        "subject": subject,
        "recorded_at": recorded.isoformat(),
        "fs": float(fs),
        "channel_order": list(channel_order),
        "n_trials_per_class": int(n_trials_per_class),
        "n_total_trials": int(n_total_trials),
        "fixation_s": float(fixation_s),
        "cue_s": float(cue_s),
        "mi_s": float(mi_s),
        "rest_s": float(rest_s),
        "marker_left": int(marker_left),
        "marker_right": int(marker_right),
        "marker_cue_left": int(marker_cue_left),
        "marker_cue_right": int(marker_cue_right),
        "marker_fixation": int(marker_fixation),
        "marker_rest": int(marker_rest),
        "package_versions": collect_package_versions(),
        "git_sha": _git_sha(),
    }
    if extra:
        md.update(extra)
    return md


def save_session_recording(
    root: str | Path,
    session_id: str,
    *,
    raw: np.ndarray,
    timestamps: np.ndarray,
    marker_times: np.ndarray,
    marker_labels: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    session_dir = Path(root) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    recording_path = session_dir / "recording.npz"
    np.savez(
        recording_path,
        raw=raw,
        timestamps=timestamps,
        marker_times=marker_times,
        marker_labels=marker_labels,
        fs=float(metadata["fs"]),
        channel_order=np.array(metadata["channel_order"]),
    )
    with (session_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2, sort_keys=True)
        f.write("\n")
    return recording_path


def load_session_recording(root: str | Path, session_id: str | None = None) -> dict[str, Any]:
    session_dir = Path(root) if session_id is None else Path(root) / session_id
    recording_path = session_dir / "recording.npz"
    if not recording_path.exists():
        raise FileNotFoundError(recording_path)
    metadata = _read_json(session_dir / "metadata.json")
    with np.load(recording_path, allow_pickle=False) as data:
        return {
            "session_id": metadata.get("session_id", session_dir.name),
            "session_dir": session_dir,
            "recording_path": recording_path,
            "metadata": metadata,
            "raw": data["raw"],
            "timestamps": data["timestamps"],
            "marker_times": data["marker_times"],
            "marker_labels": data["marker_labels"],
            "fs": float(data["fs"]),
            "channel_order": [str(x) for x in data["channel_order"].tolist()],
        }


def find_sessions(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.exists():
        return []
    sessions: list[dict[str, Any]] = []
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue
        if not (session_dir / "recording.npz").exists():
            continue
        metadata = _read_json(session_dir / "metadata.json")
        sessions.append(
            {
                "session_id": metadata.get("session_id", session_dir.name),
                "subject": metadata.get("subject", session_dir.name.split("_", 1)[0]),
                "path": session_dir,
                "metadata": metadata,
            }
        )
    return sorted(sessions, key=lambda item: str(item["session_id"]))


def save_bundle(bundle: Bundle, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix != ".joblib":
        path.mkdir(parents=True, exist_ok=True)
        metadata_path = path / "metadata.json"
        path = path / "bundle.joblib"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = path.with_suffix(".metadata.json")
    payload = {
        "decoder": bundle.decoder,
        "decoder_name": bundle.decoder_name,
        "decoder_params": bundle.decoder_params,
        "preprocessing": asdict(bundle.preprocessing),
        "label_map": bundle.label_map,
        "window_samples": bundle.window_samples,
        "stride_samples": bundle.stride_samples,
        "smoother_name": bundle.smoother_name,
        # Init kwargs only — never the smoother instance, so internal
        # state (e.g. EMA running average) cannot accidentally be
        # persisted across runs.
        "smoother_params": bundle.smoother_params,
        "metadata": bundle.metadata,
    }
    joblib.dump(payload, path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(bundle.metadata), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_bundle(path: str | Path) -> Bundle:
    path = Path(path)
    if path.is_dir():
        path = path / "bundle.joblib"
    payload = joblib.load(path)
    pre_dict = payload["preprocessing"]
    preprocessing = PreprocessingConfig(
        fs=pre_dict["fs"],
        channel_order=list(pre_dict["channel_order"]),
        bandpass=tuple(pre_dict["bandpass"]),
        bandpass_order=int(pre_dict["bandpass_order"]),
        notch_hz=pre_dict["notch_hz"],
        notch_q=float(pre_dict["notch_q"]),
        reference=str(pre_dict["reference"]),
    )
    return Bundle(
        decoder=payload["decoder"],
        decoder_name=payload["decoder_name"],
        decoder_params=dict(payload["decoder_params"]),
        preprocessing=preprocessing,
        label_map=dict(payload["label_map"]),
        window_samples=int(payload["window_samples"]),
        stride_samples=int(payload["stride_samples"]),
        smoother_name=str(payload["smoother_name"]),
        smoother_params=dict(payload["smoother_params"]),
        metadata=dict(payload["metadata"]),
    )


def find_bundles(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.exists():
        return []
    bundles: list[dict[str, Any]] = []
    for bundle_path in sorted(root.glob("*/*/bundle.joblib")):
        bundle_dir = bundle_path.parent
        decoder_name = bundle_dir.name
        session_id = bundle_dir.parent.name
        metadata = _read_json(bundle_dir / "metadata.json")
        bundles.append(
            {
                "session_id": metadata.get("session_id", session_id),
                "decoder_name": metadata.get("decoder_name", decoder_name),
                "path": bundle_dir,
                "bundle_path": bundle_path,
                "metadata": metadata,
                "acc": metadata.get("cv_acc", metadata.get("cal_acc")),
            }
        )
    return sorted(
        bundles,
        key=lambda item: (str(item["session_id"]), str(item["decoder_name"])),
    )
