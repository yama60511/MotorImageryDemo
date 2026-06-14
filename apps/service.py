"""BCI inference service (Phase 5).

State machine:
    IDLE -> CALIBRATING -> TRAINING -> READY -> RUNNING -> READY ...

The service is fully headless and Unity-agnostic. All control flows
through the LSL Commands stream and all visible feedback flows out via
the Status / BCI_Proba streams.

Run from the repo root:
    python -m apps.service
"""

from __future__ import annotations

import argparse
import enum
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bci.bundle import Bundle, load_bundle, make_metadata, save_bundle  # noqa: E402
from bci.config import (  # noqa: E402
    PreprocessingConfig,
    build_preprocessing_config,
    load_yaml,
)
from bci.epoching import extract_trials, sliding_windows  # noqa: E402
from bci.logging_setup import setup_logging  # noqa: E402
from bci.lsl_io import (  # noqa: E402
    CommandReceiver,
    EEGReceiver,
    MarkerReceiver,
    ProbaPublisher,
    StatusPublisher,
)
from bci.models.csp_lda.decoder import CSPLDADecoder  # noqa: E402
from bci.models.registry import (  # noqa: E402
    build as build_decoder,
    list_available,
)
from bci.pipeline import InferencePipeline  # noqa: E402
from bci.preprocessing import Preprocessor  # noqa: E402
from bci.smoother import make_smoother  # noqa: E402

import bci.models  # noqa: E402,F401  -- registers decoders

logger = logging.getLogger("bci.service")


class State(str, enum.Enum):
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    TRAINING = "TRAINING"
    READY = "READY"
    RUNNING = "RUNNING"


def parse_command(text: str) -> tuple[str, dict[str, str]]:
    text = text.strip()
    if ":" in text:
        name, rest = text.split(":", 1)
        kwargs: dict[str, str] = {}
        for kv in rest.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if "=" in kv:
                k, v = kv.split("=", 1)
                kwargs[k.strip()] = v.strip()
            else:
                kwargs[kv] = ""
        return name.strip(), kwargs
    return text, {}


class Service:
    def __init__(
        self,
        cfg: dict[str, Any],
        channels: list[str] | None = None,
        decoder_name: str = "csp_lda",
        decoder_params: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.channels = channels
        self.decoder_name = decoder_name
        self.decoder_params = dict(decoder_params or {"n_components": 4})

        # Logical state
        self.state: State = State.IDLE
        self.shutdown_requested = False

        # LSL plumbing
        self.eeg_rx: EEGReceiver | None = None
        self.marker_rx: MarkerReceiver | None = None
        self.cmd_rx: CommandReceiver | None = None
        self.status_pub: StatusPublisher | None = None
        self.proba_pub: ProbaPublisher | None = None
        self.state_heartbeat_s: float = 1.0
        self.next_state_heartbeat_t: float = 0.0

        # Calibration state (reset per session)
        self.cal_subject: str = "unknown"
        self.cal_trial_onsets: list[tuple[float, int]] = []
        self.cal_cue_count: int = 0
        self.cal_expected_cues: int = (
            int(cfg["calibration"]["trials_per_class"]) * 2
        )
        self.cal_last_cue_t: float | None = None
        self.cal_data_path: Path | None = None

        # Trained artefact (post-TRAINING) — referenced by save_bundle.
        self.bundle: Bundle | None = None
        self.preprocessing: PreprocessingConfig | None = None
        self.window_samples: int = 0
        self.realtime_stride_samples: int = 0
        self.calib_stride_samples: int = 0

        # Realtime state
        self.pipeline: InferencePipeline | None = None
        self.realtime_warmup_until: float = 0.0
        self.realtime_warmup_done: bool = False
        self.next_proba_t: float = 0.0

        # Sanity-check cfg
        self.sanity_max_abs_uv: float = float(
            cfg["calibration"].get("sanity", {}).get("max_abs_uv", 500.0)
        )
        self.sanity_nan_inf: bool = bool(
            cfg["calibration"].get("sanity", {}).get("nan_inf_check", True)
        )

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------
    def connect_streams(self) -> None:
        self.status_pub = StatusPublisher()
        self.proba_pub = ProbaPublisher(
            n_classes=2,
        )
        self.eeg_rx = EEGReceiver(
            stream_type=self.cfg["lsl"]["eeg_stream_type"],
            stall_timeout_s=float(self.cfg["lsl"]["stall_timeout_s"]),
            channel_selection=self.channels,
        )
        info = self.eeg_rx.connect()
        fs = float(info["fs"])
        # Now that we know fs, finalise window/stride parameters.
        win_cfg = self.cfg["windowing"]
        self.window_samples = int(round(float(win_cfg["window_s"]) * fs))
        self.realtime_stride_samples = int(
            round(int(win_cfg["realtime_stride_ms"]) / 1000.0 * fs)
        )
        self.calib_stride_samples = int(
            round(int(win_cfg["calib_stride_ms"]) / 1000.0 * fs)
        )
        self.preprocessing = build_preprocessing_config(
            self.cfg, fs=fs, channel_order=info["channel_names"]
        )

        self.marker_rx = MarkerReceiver(
            stream_type=self.cfg["lsl"]["markers_stream"]
        )
        self.marker_rx.connect()
        self.cmd_rx = CommandReceiver(
            stream_type=self.cfg["lsl"]["commands_stream"]
        )
        self.cmd_rx.connect()

        self.eeg_rx.start_streaming()
        self._publish_state()
        self._push_status(
            f"decoders_available:{','.join(list_available())}"
        )
        self.next_state_heartbeat_t = time.time() + self.state_heartbeat_s

    def shutdown(self) -> None:
        if self.eeg_rx is not None:
            self.eeg_rx.stop_streaming()
        logger.info("service stopped")

    # ------------------------------------------------------------------
    # Public main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.connect_streams()
        try:
            while not self.shutdown_requested:
                self.tick()
                time.sleep(0.02)
        finally:
            self.shutdown()

    def tick(self) -> None:
        assert self.cmd_rx is not None
        self._tick_state_heartbeat()
        for raw_cmd in self.cmd_rx.pull():
            self._handle_command(raw_cmd)
        if self.state == State.CALIBRATING:
            self._tick_calibrating()
        elif self.state == State.RUNNING:
            self._tick_running()

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------
    def _handle_command(self, raw_cmd: str) -> None:
        name, kwargs = parse_command(raw_cmd)
        logger.info("cmd: %s %s", name, kwargs)
        try:
            if name == "shutdown":
                self.shutdown_requested = True
            elif name == "start_calibration":
                self._cmd_start_calibration(
                    kwargs.get("subject", "unknown"),
                    decoder=kwargs.get("decoder"),
                )
            elif name == "abort_calibration":
                self._cmd_abort_calibration()
            elif name == "save_bundle":
                self._cmd_save_bundle(kwargs.get("subject", self.cal_subject))
            elif name == "load_bundle":
                self._cmd_load_bundle(kwargs.get("path", ""))
            elif name == "start_realtime":
                self._cmd_start_realtime()
            elif name == "stop_realtime":
                self._cmd_stop_realtime()
            elif name == "query_config":
                self._publish_service_config()
            else:
                logger.warning("unknown command: %s", raw_cmd)
                self._push_status(f"error:unknown_command:{name}")
        except Exception as exc:  # pragma: no cover -- defensive
            logger.exception("command %s failed: %s", name, exc)
            self._push_status(f"error:command_failed:{name}")

    # ------------------------------------------------------------------
    # CALIBRATING
    # ------------------------------------------------------------------
    def _cmd_start_calibration(
        self, subject: str, decoder: str | None = None
    ) -> None:
        if self.state != State.IDLE:
            self._push_status(f"error:bad_state:{self.state.value}")
            return
        if decoder is not None:
            if decoder not in list_available():
                self._push_status(f"error:unknown_decoder:{decoder}")
                return
            self.decoder_name = decoder
        assert self.eeg_rx is not None
        self.cal_subject = subject
        self.cal_trial_onsets = []
        self.cal_cue_count = 0
        self.cal_last_cue_t = None
        self.eeg_rx.detach_preprocessor()
        self.eeg_rx.start_recording()
        self.state = State.CALIBRATING
        self._publish_state()
        # Re-publish service_config so Unity (or any client that joined
        # late) has up-to-date timing parameters before the marker loop
        # starts. Idempotent ? same payload as the connect-time response
        # to query_config.
        self._publish_service_config()
        self._push_status(
            f"calibration_progress:0/{self.cal_expected_cues}"
        )

    def _cmd_abort_calibration(self) -> None:
        if self.state != State.CALIBRATING:
            return
        assert self.eeg_rx is not None
        self.eeg_rx.stop_recording()
        self.cal_trial_onsets = []
        self.cal_cue_count = 0
        self.state = State.IDLE
        self._publish_state()

    def _tick_calibrating(self) -> None:
        assert self.marker_rx is not None
        assert self.eeg_rx is not None

        if self.eeg_rx.is_stalled():
            self._push_status("error:lsl_eeg_stalled")
            self.eeg_rx.stop_recording()
            self.state = State.IDLE
            self._publish_state()
            return

        for t_lsl, label in self.marker_rx.pull():
            if label in (0, 1):
                self.cal_trial_onsets.append((t_lsl, label))
                self.cal_cue_count += 1
                self.cal_last_cue_t = time.time()
                self._push_status(
                    f"calibration_progress:{self.cal_cue_count}/{self.cal_expected_cues}"
                )

        if self.cal_cue_count < self.cal_expected_cues:
            return

        # All cues received; wait one more (mi + rest) period before stopping
        # so the last trial's MI is fully captured.
        if self.cal_last_cue_t is None:
            return
        wait_s = (
            float(self.cfg["calibration"]["mi_s"])
            + float(self.cfg["calibration"]["rest_s"])
        )
        if time.time() - self.cal_last_cue_t < wait_s:
            return

        self._finalize_calibration()

    def _finalize_calibration(self) -> None:
        assert self.eeg_rx is not None
        assert self.preprocessing is not None
        raw, ts = self.eeg_rx.stop_recording()
        logger.info(
            "Calibration recording complete: raw shape=%s ts shape=%s",
            raw.shape,
            ts.shape,
        )

        # Persist the recording unconditionally — the raw EEG is the
        # ground-truth artefact and must survive even when downstream
        # filtering or training fails. Sanity checks are intentionally
        # NOT applied to raw data: pre-filter EEG carries large DC
        # offsets that would mask a meaningful amplitude threshold.
        self._save_calibration_npz(raw, ts)

        pre = Preprocessor(self.preprocessing)
        filtered = pre.transform(raw)
        if not self._sanity_check(filtered):
            self.state = State.IDLE
            self._publish_state()
            return

        self.state = State.TRAINING
        self._publish_state()

        try:
            cv_acc = self._fit_decoder(filtered, ts)
        except Exception as exc:
            logger.exception("training failed: %s", exc)
            self._push_status(f"error:training_failed:{exc}")
            self.state = State.IDLE
            self._publish_state()
            return

        self._push_status(f"calibration_done:acc={cv_acc:.3f}")
        self.state = State.READY
        self._publish_state()

    def _save_calibration_npz(self, raw: np.ndarray, ts: np.ndarray) -> None:
        assert self.preprocessing is not None
        ts_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.cal_data_path = (
            REPO_ROOT
            / self.cfg["paths"]["data_dir"]
            / f"{self.cal_subject}_{ts_label}.npz"
        )
        self.cal_data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.cal_data_path,
            raw=raw,
            timestamps=ts,
            marker_times=np.array(
                [t for t, _ in self.cal_trial_onsets], dtype=np.float64
            ),
            marker_labels=np.array(
                [lbl for _, lbl in self.cal_trial_onsets], dtype=np.int64
            ),
            fs=self.preprocessing.fs,
            channel_order=np.array(self.preprocessing.channel_order),
        )
        self._push_status(f"calibration_saved:path={self.cal_data_path}")

    def _sanity_check(self, data: np.ndarray) -> bool:
        if data.size == 0 or data.shape[1] == 0:
            self._push_status("error:bad_calibration_data:reason=empty")
            return False
        if self.sanity_nan_inf and not np.isfinite(data).all():
            self._push_status("error:bad_calibration_data:reason=non_finite")
            return False
        # Exclude the filter warm-up region from the amplitude check. A
        # bandpass run from a zero state over EEG with a large DC offset
        # produces a brief start-up transient that is a filter edge
        # artefact, not signal — and on the 2-channel path (CAR off) it can
        # spike to thousands of uV. Trials are extracted well after this
        # region, so it must not fail an otherwise-good recording.
        assert self.preprocessing is not None
        warmup = int(round(self.preprocessing.fs))
        amp_region = data[:, warmup:] if data.shape[1] > warmup else data
        max_abs = float(np.abs(amp_region).max())
        if max_abs > self.sanity_max_abs_uv:
            self._push_status(
                f"error:bad_calibration_data:reason=amp_exceeded:peak={max_abs:.1f}"
            )
            return False
        return True

    def _fit_decoder(
        self, filtered: np.ndarray, timestamps: np.ndarray
    ) -> float:
        assert self.preprocessing is not None
        trials, labels = extract_trials(
            filtered,
            timestamps,
            self.cal_trial_onsets,
            float(self.cfg["calibration"]["mi_s"]),
            self.preprocessing.fs,
        )
        if trials.shape[0] == 0:
            raise RuntimeError(
                "extract_trials returned 0 trials; check marker timestamps"
            )
        X, y, group = sliding_windows(
            trials,
            labels,
            window_samples=self.window_samples,
            stride_samples=self.calib_stride_samples,
        )
        n_splits = min(int(self.cfg["cv"]["n_splits"]), len(np.unique(group)))
        n_splits = max(2, n_splits)
        splitter = GroupKFold(n_splits=n_splits)
        accs: list[float] = []
        for tr, te in splitter.split(X, y, groups=group):
            assert set(group[tr]).isdisjoint(set(group[te]))
            d = build_decoder(self.decoder_name, **self.decoder_params)
            d.fit(X[tr], y[tr])
            accs.append(float((d.predict(X[te]) == y[te]).mean()))
        cv_acc = float(np.mean(accs))
        logger.info("trial-level GroupKFold acc=%.3f folds=%s", cv_acc, accs)

        # Final fit on all data; this becomes the in-memory model the
        # save_bundle command will persist.
        final = build_decoder(self.decoder_name, **self.decoder_params)
        final.fit(X, y)

        smoother_cfg = self.cfg["realtime"]["smoother"]
        self.bundle = Bundle(
            decoder=final,
            decoder_name=self.decoder_name,
            decoder_params=final.get_params(),
            preprocessing=self.preprocessing,
            label_map={0: "left", 1: "right"},
            window_samples=self.window_samples,
            stride_samples=self.calib_stride_samples,
            smoother_name=str(smoother_cfg["name"]),
            smoother_params={
                k: v for k, v in smoother_cfg.items() if k != "name"
            },
            metadata=make_metadata(
                subject=self.cal_subject,
                cal_acc=cv_acc,
                raw_data_path=self.cal_data_path,
                trial_len_s=float(self.cfg["calibration"]["mi_s"]),
                cue_s=float(self.cfg["calibration"]["cue_s"]),
                mi_s=float(self.cfg["calibration"]["mi_s"]),
                rest_s=float(self.cfg["calibration"]["rest_s"]),
                n_trials_per_class=int(
                    self.cfg["calibration"]["trials_per_class"]
                ),
                fs=self.preprocessing.fs,
                emotiv_n_channels=self.preprocessing.n_channels,
                emotiv_total_channels=self.eeg_rx.info["total_channels"],
                emotiv_channel_names=self.eeg_rx.info["channel_names"],
                physical_electrodes=list(self.preprocessing.channel_order),
                logical_channels=list(self.preprocessing.channel_order),
                extra={"cv_folds": accs},
            ),
        )
        return cv_acc

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def _cmd_save_bundle(self, subject: str) -> None:
        if self.bundle is None:
            self._push_status("error:no_bundle")
            return
        ts_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = (
            REPO_ROOT
            / self.cfg["paths"]["bundles_dir"]
            / f"{subject}_{ts_label}.joblib"
        )
        save_bundle(self.bundle, path)
        self._push_status(f"bundle_saved:path={path}")

    def _cmd_load_bundle(self, path: str) -> None:
        if not path:
            self._push_status("error:load_bundle_path_missing")
            return
        bundle_path = Path(path)
        if not bundle_path.is_absolute():
            bundle_path = REPO_ROOT / bundle_path
        if not bundle_path.exists():
            self._push_status(f"error:bundle_not_found:{bundle_path}")
            return
        self.bundle = load_bundle(bundle_path)
        # Re-derive the in-memory window/stride from the bundle so
        # realtime predictions use exactly what the bundle was trained
        # for, regardless of what configs/default.yaml says now.
        self.window_samples = self.bundle.window_samples
        self.preprocessing = self.bundle.preprocessing
        self._push_status(
            f"bundle_loaded:subject={self.bundle.metadata.get('subject', '?')}"
        )
        self.state = State.READY
        self._publish_state()

    # ------------------------------------------------------------------
    # RUNNING
    # ------------------------------------------------------------------
    def _cmd_start_realtime(self) -> None:
        if self.state != State.READY or self.bundle is None:
            self._push_status(f"error:bad_state:{self.state.value}")
            return
        assert self.eeg_rx is not None
        assert self.preprocessing is not None

        if self.preprocessing.n_channels != self.eeg_rx.info["n_channels"]:
            self._push_status(
                "error:channel_mismatch:"
                f"bundle={self.preprocessing.n_channels}/"
                f"stream={self.eeg_rx.info['n_channels']}"
            )
            return

        pre = Preprocessor(self.preprocessing)
        self.eeg_rx.attach_preprocessor(pre)
        smoother = make_smoother(
            self.bundle.smoother_name, self.bundle.smoother_params
        )
        self.pipeline = InferencePipeline(pre, self.bundle.decoder, smoother)
        self.pipeline.reset_smoother()

        self.state = State.RUNNING
        self._publish_state()
        self._push_status("realtime_warming_up")
        warmup_s = float(self.cfg["realtime"]["warmup_s"])
        self.realtime_warmup_until = time.time() + warmup_s
        self.realtime_warmup_done = False
        fs = self.preprocessing.fs
        self.next_proba_t = self.realtime_warmup_until + (
            self.realtime_stride_samples / fs
        )

    def _cmd_stop_realtime(self) -> None:
        if self.state != State.RUNNING:
            return
        assert self.eeg_rx is not None
        self.eeg_rx.detach_preprocessor()
        self.pipeline = None
        self.state = State.READY
        self._publish_state()

    def _tick_running(self) -> None:
        assert self.eeg_rx is not None
        assert self.proba_pub is not None
        if self.eeg_rx.is_stalled():
            self._push_status("error:lsl_eeg_stalled")
            self._cmd_stop_realtime()
            return

        now = time.time()
        if now < self.realtime_warmup_until:
            return
        if not self.realtime_warmup_done:
            self._push_status("realtime_ready")
            self.realtime_warmup_done = True

        if now < self.next_proba_t:
            return
        if self.eeg_rx.n_filled_filtered() < self.window_samples:
            return
        assert self.pipeline is not None
        window = self.eeg_rx.get_last_filtered(self.window_samples)
        proba = self.pipeline.predict_window(window)
        self.proba_pub.push(proba)
        stride_s = self.realtime_stride_samples / self.preprocessing.fs
        self.next_proba_t = max(self.next_proba_t + stride_s, now + 0.001)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _push_status(self, msg: str) -> None:
        if self.status_pub is not None:
            self.status_pub.push(msg)
        logger.info("status: %s", msg)

    def _publish_state(self) -> None:
        self._push_status(f"state:{self.state.value}")

    def _publish_service_config(self) -> None:
        """Broadcast the timing parameters Unity needs.

        Sent in response to a ``query_config`` command (Unity issues this
        as soon as its Status inlet is up) and re-sent on
        ``start_calibration`` so late joiners stay in sync. Single source
        of truth is the YAML loaded by the service.
        """
        cal = self.cfg["calibration"]
        win = self.cfg["windowing"]
        self._push_status(
            "service_config:"
            f"cue_s={float(cal['cue_s'])},"
            f"mi_s={float(cal['mi_s'])},"
            f"rest_s={float(cal['rest_s'])},"
            f"trials_per_class={int(cal['trials_per_class'])},"
            f"realtime_stride_ms={int(win['realtime_stride_ms'])},"
            f"window_s={float(win['window_s'])}"
        )

    def _tick_state_heartbeat(self) -> None:
        now = time.time()
        if now < self.next_state_heartbeat_t:
            return
        self._publish_state()
        self.next_state_heartbeat_t = now + self.state_heartbeat_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "default.yaml")
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="EEG channel names to subset (default: all 14)",
    )
    parser.add_argument(
        "--decoder",
        default=None,
        help=(
            "Decoder name (overrides config). "
            f"Available: {','.join(list_available())}"
        ),
    )
    args = parser.parse_args()

    setup_logging(REPO_ROOT / "logs", level="INFO")
    cfg = load_yaml(args.config)
    channels = args.channels or cfg.get("channels", {}).get("selection")
    if channels in (None, [], "null"):
        channels = None

    dec_cfg = cfg.get("decoder", {}) or {}
    decoder_name = args.decoder or dec_cfg.get("name", "csp_lda")
    decoder_params = dec_cfg.get("params", {"n_components": 4})
    if decoder_name not in list_available():
        parser.error(
            f"unknown decoder: {decoder_name!r}. "
            f"available: {list_available()}"
        )

    svc = Service(
        cfg=cfg,
        channels=channels,
        decoder_name=decoder_name,
        decoder_params=decoder_params,
    )

    def _on_signal(signum, frame):  # noqa: ANN001
        logger.info("signal %d -> shutdown", signum)
        svc.shutdown_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    svc.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
