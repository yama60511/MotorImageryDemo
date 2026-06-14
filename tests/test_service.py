"""Phase 5 integration test: drive Service through every state.

Brings up a fake EPOC X EEG outlet, a Markers outlet, a Commands
outlet, plus Status / BCI_Proba inlets. Runs the Service in a worker
thread and walks it through:

  IDLE -> CALIBRATING -> TRAINING -> READY -> RUNNING -> READY -> shutdown

This is slower than other unit tests (calibration timing is bounded by
real wall-clock so the marker timestamps stay coherent with EEG).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pylsl import StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_byprop

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import apps.service as service_mod  # noqa: E402
from apps.service import Service, State, parse_command  # noqa: E402
from bci.bundle import Bundle, save_bundle  # noqa: E402
from bci.config import PreprocessingConfig  # noqa: E402
from bci.models.base import BaseDecoder  # noqa: E402
from tests._synth import make_mi_trial  # noqa: E402

FS = 128.0
N_EEG = 14
TOTAL_CHANNELS = 3 + N_EEG + 2
EPOCX_LABELS: tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)


def _test_cfg(trials_per_class: int = 2) -> dict:
    """Compact config that runs in a few seconds.

    Calibration uses 2 trials/class (4 total) at 0.6s MI duration so
    the test wall-clock stays small, while still exercising the full
    extract_trials -> sliding_windows -> CV path.
    """
    return {
        "lsl": {
            "eeg_stream_type": "EEG",
            "markers_stream": "Markers",
            "commands_stream": "Commands",
            "stall_timeout_s": 2.0,
            "emotiv": {
                "channel_offset": 3,
                "expected_total_channels": [19],
            },
        },
        "preprocessing": {
            "bandpass": [8.0, 30.0],
            "bandpass_order": 4,
            "notch_hz": 50.0,
            "notch_q": 30.0,
            "reference": "auto",
        },
        "calibration": {
            "trials_per_class": trials_per_class,
            "cue_s": 0.2,
            "mi_s": 1.5,
            "rest_s": 0.4,
            "sanity": {"max_abs_uv": 5000.0, "nan_inf_check": True},
        },
        "windowing": {
            "window_s": 1.0,
            "realtime_stride_ms": 250,
            "calib_stride_ms": 250,
        },
        "cv": {"splitter": "group_kfold", "n_splits": 2},
        "realtime": {
            "warmup_s": 0.4,
            "smoother": {"name": "ema", "alpha": 0.5},
            "fallback_acc_threshold": 0.6,
        },
        "channels": {"selection": None},
        "paths": {
            "bundles_dir": "bundles/",
            "data_dir": "data/",
            "logs_dir": "logs/",
        },
    }


def _make_eeg_outlet(source_id: str, stream_type: str = "EEG") -> StreamOutlet:
    info = StreamInfo(
        name=f"SvcTestEEG_{source_id}",
        type=stream_type,
        channel_count=TOTAL_CHANNELS,
        nominal_srate=FS,
        channel_format="float32",
        source_id=source_id,
    )
    desc = info.desc()
    desc.append_child_value("manufacturer", "Emotiv")
    chans = desc.append_child("channels")
    labels = (
        ("counter", "interpolated", "signal_quality")
        + EPOCX_LABELS
        + ("aux0", "aux1")
    )
    for name in labels:
        ch = chans.append_child("channel")
        ch.append_child_value("label", name)
    return StreamOutlet(info, chunk_size=16, max_buffered=360)


def _push_eeg_chunk(
    outlet: StreamOutlet,
    n_samples: int,
    label: int | None,
    sample_idx: int,
    rng: np.random.Generator,
    motor_indices: tuple[int, int] = (3, 10),  # FC5, FC6
) -> int:
    block = np.zeros((n_samples, TOTAL_CHANNELS), dtype=np.float32)
    block[:, 0] = (np.arange(sample_idx, sample_idx + n_samples) % 256).astype(
        np.float32
    )
    block[:, 2] = 1.0
    eeg = rng.standard_normal((n_samples, N_EEG)).astype(np.float32) * 1.5
    if label in (0, 1):
        pair = make_mi_trial(
            label=label,
            n_samples=n_samples,
            fs=FS,
            n_channels=2,
            rng=rng,
        )
        eeg[:, motor_indices[0]] = pair[0]
        eeg[:, motor_indices[1]] = pair[1]
    block[:, 3 : 3 + N_EEG] = eeg
    outlet.push_chunk(block.tolist(), local_clock())
    return sample_idx + n_samples


def _make_string_outlet(stream_type: str, source_id: str) -> StreamOutlet:
    info = StreamInfo(
        name=f"SvcTest_{stream_type}_{source_id}",
        type=stream_type,
        channel_count=1,
        nominal_srate=0.0,
        channel_format="string",
        source_id=source_id,
    )
    return StreamOutlet(info)


def _make_int_outlet(stream_type: str, source_id: str) -> StreamOutlet:
    info = StreamInfo(
        name=f"SvcTest_{stream_type}_{source_id}",
        type=stream_type,
        channel_count=1,
        nominal_srate=0.0,
        channel_format="int32",
        source_id=source_id,
    )
    return StreamOutlet(info)


def _resolve_inlet(stream_type: str, source_id: str) -> StreamInlet:
    streams = resolve_byprop("type", stream_type, timeout=5.0)
    info = next(s for s in streams if s.source_id() == source_id)
    inlet = StreamInlet(info)
    inlet.open_stream(timeout=2.0)
    return inlet


def _drain_string_inlet(inlet: StreamInlet) -> list[str]:
    samples, _ts = inlet.pull_chunk(timeout=0.0, max_samples=128)
    return [s[0] for s in samples]


class _RecordingStatusPublisher:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def push(self, msg: str) -> None:
        self.messages.append(msg)


class _RecordingProbaPublisher:
    def __init__(self) -> None:
        self.samples: list[np.ndarray] = []

    def push(self, proba: np.ndarray) -> None:
        self.samples.append(np.asarray(proba, dtype=np.float64))


class _StaticDecoder(BaseDecoder):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_StaticDecoder":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(np.asarray(X).shape[0], dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = np.asarray(X).shape[0]
        return np.tile(np.array([[0.25, 0.75]], dtype=np.float64), (n, 1))

    def get_params(self) -> dict[str, Any]:
        return {}

    def set_params(self, params: dict[str, Any]) -> None:
        return None


class _FakeEEGReceiver:
    def __init__(
        self,
        *,
        n_channels: int = 2,
        n_filled: int = 128,
        stalled: bool = False,
    ) -> None:
        self._info = {
            "n_channels": n_channels,
            "total_channels": n_channels,
            "channel_names": ["C3", "C4"][:n_channels],
        }
        self._n_filled = n_filled
        self._stalled = stalled
        self.attached = False
        self.detached = False

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    def attach_preprocessor(self, pre: object) -> None:
        self.attached = True

    def detach_preprocessor(self) -> None:
        self.detached = True

    def is_stalled(self) -> bool:
        return self._stalled

    def n_filled_filtered(self) -> int:
        return self._n_filled

    def get_last_filtered(self, n_samples: int) -> np.ndarray:
        return np.zeros((self._info["n_channels"], n_samples), dtype=np.float64)


class _BadCalibrationEEGReceiver(_FakeEEGReceiver):
    def stop_recording(self) -> tuple[np.ndarray, np.ndarray]:
        # A 15 Hz oscillation lives in the 8-30 Hz passband, so its
        # amplitude survives Preprocessor.transform almost unchanged and
        # reliably trips the amp_exceeded check on the *filtered* signal.
        fs = 128.0
        n = 256
        t = np.arange(n) / fs
        sig = 10000.0 * np.sin(2 * np.pi * 15.0 * t)
        raw = np.vstack([sig, -sig]).astype(np.float64)
        ts = t.astype(np.float64)
        return raw, ts


class _NormalCalibrationEEGReceiver(_FakeEEGReceiver):
    def stop_recording(self) -> tuple[np.ndarray, np.ndarray]:
        # A realistic Emotiv 2-channel recording: a ~4180 uV DC offset plus
        # a small in-band (15 Hz) oscillation of a few tens of uV. A
        # zero-state bandpass turns the DC step into a brief multi-hundred-uV
        # start-up transient, but the actual signal is tiny. This is exactly
        # the recording the amplitude check used to falsely reject.
        fs = 128.0
        n = 256
        t = np.arange(n) / fs
        sig = 4180.0 + 15.0 * np.sin(2 * np.pi * 15.0 * t)
        raw = np.vstack([sig, sig * 0.98]).astype(np.float64)
        ts = t.astype(np.float64)
        return raw, ts


def _make_test_bundle() -> Bundle:
    pre_cfg = PreprocessingConfig(fs=128.0, channel_order=["C3", "C4"])
    return Bundle(
        decoder=_StaticDecoder(),
        decoder_name="static",
        decoder_params={},
        preprocessing=pre_cfg,
        label_map={0: "left", 1: "right"},
        window_samples=128,
        stride_samples=32,
        smoother_name="passthrough",
        smoother_params={},
        metadata={"subject": "alice"},
    )


def test_parse_command_extracts_kwargs() -> None:
    name, kw = parse_command("start_calibration:subject=A")
    assert name == "start_calibration"
    assert kw == {"subject": "A"}

    name, kw = parse_command("save_bundle:subject=Bob,note=foo")
    assert name == "save_bundle"
    assert kw == {"subject": "Bob", "note": "foo"}

    name, kw = parse_command("shutdown")
    assert name == "shutdown"
    assert kw == {}


def test_finalize_calibration_rejects_amplitude_outlier(tmp_path: Path) -> None:
    cfg = _test_cfg(trials_per_class=1)
    cfg["paths"]["data_dir"] = str(tmp_path)
    svc = Service(cfg=cfg)
    status = _RecordingStatusPublisher()
    svc.status_pub = status
    svc.eeg_rx = _BadCalibrationEEGReceiver()
    svc.preprocessing = PreprocessingConfig(fs=128.0, channel_order=["C3", "C4"])
    svc.state = State.CALIBRATING

    svc._finalize_calibration()

    assert svc.state == State.IDLE
    joined = " | ".join(status.messages)
    assert "error:bad_calibration_data:reason=amp_exceeded" in joined
    assert "state:IDLE" in joined
    # The raw npz must be persisted even though the sanity check failed —
    # losing the recording is worse than skipping training.
    assert svc.cal_data_path is not None and svc.cal_data_path.exists()
    with np.load(svc.cal_data_path) as data:
        assert data["raw"].shape == (2, 256)


def test_finalize_calibration_accepts_normal_dc_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal recording with a large Emotiv DC offset must be accepted.

    The bandpass start-up transient that DC offset produces is a filter
    edge artefact, not signal, so the amplitude sanity check must not
    reject the recording. Uses the real 500 uV threshold (the transient
    spikes to ~2000 uV, so without the warm-up exclusion this would fail).
    Training is stubbed to isolate the sanity-check behaviour.
    """
    cfg = _test_cfg(trials_per_class=1)
    cfg["paths"]["data_dir"] = str(tmp_path)
    cfg["calibration"]["sanity"]["max_abs_uv"] = 500.0
    svc = Service(cfg=cfg)
    status = _RecordingStatusPublisher()
    svc.status_pub = status
    svc.eeg_rx = _NormalCalibrationEEGReceiver()
    svc.preprocessing = PreprocessingConfig(fs=128.0, channel_order=["C3", "C4"])
    svc.state = State.CALIBRATING
    # Isolate the sanity-check path from decoder training.
    monkeypatch.setattr(svc, "_fit_decoder", lambda filtered, ts: 0.9)

    svc._finalize_calibration()

    joined = " | ".join(status.messages)
    assert "amp_exceeded" not in joined, joined
    assert svc.state == State.READY, joined
    assert svc.cal_data_path is not None and svc.cal_data_path.exists()


def test_load_bundle_command_moves_service_to_ready(tmp_path: Path) -> None:
    bundle_path = save_bundle(_make_test_bundle(), tmp_path / "alice.joblib")
    svc = Service(cfg=_test_cfg())
    status = _RecordingStatusPublisher()
    svc.status_pub = status

    svc._cmd_load_bundle(str(bundle_path))

    assert svc.state == State.READY
    assert svc.bundle is not None
    assert svc.bundle.metadata["subject"] == "alice"
    assert svc.window_samples == 128
    assert svc.preprocessing is not None
    assert svc.preprocessing.channel_order == ["C3", "C4"]
    joined = " | ".join(status.messages)
    assert "bundle_loaded:subject=alice" in joined
    assert "state:READY" in joined


def test_realtime_warmup_suppresses_proba_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _test_cfg()
    cfg["realtime"]["warmup_s"] = 0.4
    svc = Service(cfg=cfg)
    status = _RecordingStatusPublisher()
    proba = _RecordingProbaPublisher()
    eeg = _FakeEEGReceiver(n_channels=2, n_filled=128)
    svc.status_pub = status
    svc.proba_pub = proba
    svc.eeg_rx = eeg
    svc.bundle = _make_test_bundle()
    svc.preprocessing = svc.bundle.preprocessing
    svc.window_samples = svc.bundle.window_samples
    svc.realtime_stride_samples = 32  # 250ms at 128Hz
    svc.state = State.READY

    now = [100.0]
    monkeypatch.setattr(service_mod.time, "time", lambda: now[0])

    svc._cmd_start_realtime()
    assert eeg.attached is True
    assert svc.state == State.RUNNING

    now[0] = 100.39
    svc._tick_running()
    assert proba.samples == []
    assert "realtime_ready" not in status.messages

    now[0] = 100.40
    svc._tick_running()
    assert proba.samples == []
    assert "realtime_ready" in status.messages

    now[0] = 100.65
    svc._tick_running()
    assert len(proba.samples) == 1
    np.testing.assert_allclose(proba.samples[0], [0.25, 0.75])


def test_realtime_stall_returns_to_ready() -> None:
    svc = Service(cfg=_test_cfg())
    status = _RecordingStatusPublisher()
    eeg = _FakeEEGReceiver(stalled=True)
    svc.status_pub = status
    svc.proba_pub = _RecordingProbaPublisher()
    svc.eeg_rx = eeg
    svc.pipeline = object()  # Only presence matters before stop_realtime clears it.
    svc.state = State.RUNNING

    svc._tick_running()

    assert svc.state == State.READY
    assert svc.pipeline is None
    assert eeg.detached is True
    joined = " | ".join(status.messages)
    assert "error:lsl_eeg_stalled" in joined
    assert "state:READY" in joined


def test_service_full_lifecycle(tmp_path: pytest.TempPathFactory) -> None:
    cfg = _test_cfg(trials_per_class=2)
    cfg["lsl"]["eeg_stream_type"] = "SvcTestEEG_lifecycle"
    cfg["lsl"]["markers_stream"] = "SvcTestMarkers_lifecycle"
    cfg["lsl"]["commands_stream"] = "SvcTestCommands_lifecycle"
    cfg["paths"] = {
        "bundles_dir": str(tmp_path / "bundles"),
        "data_dir": str(tmp_path / "data"),
        "logs_dir": str(tmp_path / "logs"),
    }

    eeg_out = _make_eeg_outlet(
        "svc_lifecycle", stream_type=cfg["lsl"]["eeg_stream_type"]
    )
    marker_out = _make_int_outlet(cfg["lsl"]["markers_stream"], "svc_marker")
    cmd_out = _make_string_outlet(cfg["lsl"]["commands_stream"], "svc_cmd")

    # Wait so receivers can resolve.
    time.sleep(0.5)

    svc = Service(cfg=cfg, channels=None)

    # Pump EEG continuously in a background thread so the receiver
    # always has fresh samples and never trips the stall detector.
    stop_eeg = threading.Event()
    eeg_label = {"label": None}  # mutable container shared with main thread
    rng = np.random.default_rng(0)

    def eeg_pump() -> None:
        idx = 0
        chunk_n = 16
        period = chunk_n / FS
        next_t = time.time()
        while not stop_eeg.is_set():
            idx = _push_eeg_chunk(
                eeg_out, chunk_n, eeg_label["label"], idx, rng
            )
            next_t += period
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.time()

    eeg_thread = threading.Thread(target=eeg_pump, daemon=True)
    eeg_thread.start()

    svc_thread = threading.Thread(target=svc.run, daemon=True)
    svc_thread.start()

    try:
        # Resolve Status / Proba once the service has published them.
        time.sleep(1.0)
        status_inlet = _resolve_inlet("Status", "bci_status")
        proba_inlet = _resolve_inlet("BCI_Proba", "bci_proba")
        time.sleep(0.3)

        # Drain whatever the service published before the inlet attached.
        _drain_string_inlet(status_inlet)

        # ---- start_calibration ----
        cmd_out.push_sample(["start_calibration:subject=alice"])
        time.sleep(0.6)
        cal_msgs = _drain_string_inlet(status_inlet)
        assert any("state:CALIBRATING" in m for m in cal_msgs), cal_msgs

        # ---- emit cue + rest markers paced like Unity ----
        n_per_class = cfg["calibration"]["trials_per_class"]
        labels = [0] * n_per_class + [1] * n_per_class
        rng_lbl = np.random.default_rng(1)
        rng_lbl.shuffle(labels)
        cue_s = cfg["calibration"]["cue_s"]
        mi_s = cfg["calibration"]["mi_s"]
        rest_s = cfg["calibration"]["rest_s"]
        for cls in labels:
            t_marker = local_clock()
            eeg_label["label"] = cls
            marker_out.push_sample([cls], t_marker)
            time.sleep(mi_s)
            eeg_label["label"] = None  # rest period, just noise
            marker_out.push_sample([99], local_clock())
            time.sleep(rest_s + cue_s)

        # Allow finalization (mi+rest wait inside service) + training.
        time.sleep(mi_s + rest_s + 1.5)
        post_train = _drain_string_inlet(status_inlet)
        joined = " | ".join(post_train)
        assert "calibration_done:acc=" in joined, joined
        assert "state:READY" in joined, joined
        assert svc.state == State.READY

        # ---- save_bundle ----
        cmd_out.push_sample(["save_bundle:subject=alice"])
        time.sleep(0.5)
        save_msgs = _drain_string_inlet(status_inlet)
        save_joined = " | ".join(save_msgs)
        assert "bundle_saved:path=" in save_joined, save_joined
        bundles = list(Path(cfg["paths"]["bundles_dir"]).glob("alice_*.joblib"))
        assert len(bundles) == 1

        # ---- start_realtime ----
        cmd_out.push_sample(["start_realtime"])
        time.sleep(cfg["realtime"]["warmup_s"] + 0.6)
        rt_msgs = _drain_string_inlet(status_inlet)
        rt_joined = " | ".join(rt_msgs)
        assert "realtime_warming_up" in rt_joined, rt_joined
        assert "realtime_ready" in rt_joined, rt_joined

        # Drain any pending proba samples published before resolve.
        proba_samples, _ = proba_inlet.pull_chunk(timeout=0.0, max_samples=128)

        # Hold a known label and verify proba follows.
        eeg_label["label"] = 1
        time.sleep(2.0)
        proba_samples, _ = proba_inlet.pull_chunk(timeout=0.5, max_samples=128)
        assert len(proba_samples) >= 4, f"only got {len(proba_samples)} proba"

        # ---- stop_realtime ----
        cmd_out.push_sample(["stop_realtime"])
        time.sleep(0.4)
        assert svc.state == State.READY

        # ---- shutdown ----
        cmd_out.push_sample(["shutdown"])
        svc_thread.join(timeout=3.0)
        assert not svc_thread.is_alive()
    finally:
        stop_eeg.set()
        eeg_thread.join(timeout=2.0)
        svc.shutdown_requested = True
        svc_thread.join(timeout=2.0)
        del eeg_out
        del marker_out
        del cmd_out
