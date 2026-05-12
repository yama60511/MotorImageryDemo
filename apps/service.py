"""BCI inference service (Phase 5).

State machine:
    IDLE -> COLLECTING -> IDLE
    IDLE -> TRAINING -> IDLE
    IDLE -> READY -> RUNNING -> READY ...

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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bci.bundle import (  # noqa: E402
    Bundle,
    find_bundles,
    find_sessions,
    load_bundle,
    load_session_recording,
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
from bci.lsl_io import (  # noqa: E402
    CommandReceiver,
    EEGReceiver,
    MarkerReceiver,
    ProbaPublisher,
    StatusPublisher,
)
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
    COLLECTING = "COLLECTING"
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

        # Collection state (reset per recording session)
        self.collection_subject: str = "unknown"
        self.collection_session_id: str | None = None
        self.collection_auto_train: str | None = None
        self.collection_trial_onsets: list[tuple[float, int]] = []
        self.collection_cue_count: int = 0
        self.collection_expected_cues: int = (
            int(cfg["calibration"]["trials_per_class"]) * 2
        )
        self.collection_last_cue_t: float | None = None
        self.collection_recording_path: Path | None = None

        # Trained artefact loaded for realtime.
        self.bundle: Bundle | None = None
        self.bundle_session_id: str | None = None
        self.bundle_decoder_name: str | None = None
        self.bundle_path: Path | None = None
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
        self._publish_client_snapshot()
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
        if self.state == State.COLLECTING:
            self._tick_collecting()
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
            elif name == "start_collection":
                self._cmd_start_collection(
                    kwargs.get("subject", "unknown"),
                    auto_train=kwargs.get("auto_train"),
                )
            elif name == "abort_collection":
                self._cmd_abort_collection()
            elif name == "train":
                self._cmd_train(
                    kwargs.get("session", ""),
                    kwargs.get("decoder", self.decoder_name),
                )
            elif name == "train_many":
                self._cmd_train_many(
                    kwargs.get("session", ""),
                    kwargs.get("decoders", ""),
                )
            elif name == "query_performance":
                self._cmd_query_performance(kwargs.get("subject", ""))
            elif name == "load_bundle":
                self._cmd_load_bundle(
                    session=kwargs.get("session"),
                    decoder=kwargs.get("decoder"),
                    path=kwargs.get("path"),
                )
            elif name == "start_calibration":
                # Transitional alias: old Unity builds still work, while
                # new clients should prefer start_collection + train.
                self._cmd_start_collection(
                    kwargs.get("subject", "unknown"),
                    auto_train=kwargs.get("decoder", self.decoder_name),
                )
            elif name == "abort_calibration":
                self._cmd_abort_collection()
            elif name == "save_bundle":
                self._push_status("error:deprecated_command:save_bundle")
            elif name == "start_realtime":
                self._cmd_start_realtime()
            elif name == "stop_realtime":
                self._cmd_stop_realtime()
            elif name == "query_config":
                self._publish_client_snapshot()
            else:
                logger.warning("unknown command: %s", raw_cmd)
                self._push_status(f"error:unknown_command:{name}")
        except Exception as exc:  # pragma: no cover -- defensive
            logger.exception("command %s failed: %s", name, exc)
            self._push_status(f"error:command_failed:{name}")

    # ------------------------------------------------------------------
    # COLLECTING
    # ------------------------------------------------------------------
    def _cmd_start_collection(
        self, subject: str, auto_train: str | None = None
    ) -> None:
        if self.state != State.IDLE:
            self._push_status(f"error:bad_state:{self.state.value}")
            return
        if auto_train is not None:
            if auto_train not in list_available():
                self._push_status(f"error:unknown_decoder:{auto_train}")
                return
            self.decoder_name = auto_train
        assert self.eeg_rx is not None
        self.collection_subject = subject
        self.collection_session_id = make_session_id(subject)
        self.collection_auto_train = auto_train
        self.collection_trial_onsets = []
        self.collection_cue_count = 0
        self.collection_last_cue_t = None
        self.collection_recording_path = None
        self.eeg_rx.detach_preprocessor()
        self.eeg_rx.start_recording()
        self.state = State.COLLECTING
        self._publish_state()
        # Re-publish service_config so Unity (or any client that joined
        # late) has up-to-date timing parameters before the marker loop
        # starts. Idempotent: same payload as the connect-time response
        # to query_config.
        self._publish_service_config()
        self._push_status(
            f"collection_progress:0/{self.collection_expected_cues}"
        )

    def _cmd_abort_collection(self) -> None:
        if self.state != State.COLLECTING:
            return
        assert self.eeg_rx is not None
        self.eeg_rx.stop_recording()
        self.collection_trial_onsets = []
        self.collection_cue_count = 0
        self.collection_auto_train = None
        self.state = State.IDLE
        self._publish_state()

    def _tick_collecting(self) -> None:
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
                self.collection_trial_onsets.append((t_lsl, label))
                self.collection_cue_count += 1
                self.collection_last_cue_t = time.time()
                self._push_status(
                    "collection_progress:"
                    f"{self.collection_cue_count}/{self.collection_expected_cues}"
                )

        if self.collection_cue_count < self.collection_expected_cues:
            return

        # All cues received; wait one more (mi + rest) period before stopping
        # so the last trial's MI is fully captured.
        if self.collection_last_cue_t is None:
            return
        wait_s = (
            float(self.cfg["calibration"]["mi_s"])
            + float(self.cfg["calibration"]["rest_s"])
        )
        if time.time() - self.collection_last_cue_t < wait_s:
            return

        self._finalize_collection()

    def _finalize_collection(self) -> None:
        assert self.eeg_rx is not None
        assert self.preprocessing is not None
        raw, ts = self.eeg_rx.stop_recording()
        logger.info(
            "Collection recording complete: raw shape=%s ts shape=%s",
            raw.shape,
            ts.shape,
        )

        # Persist the recording unconditionally: the raw EEG is the
        # ground-truth artefact and must survive even when downstream
        # filtering or training fails. Sanity checks are intentionally
        # NOT applied to raw data: pre-filter EEG carries large DC
        # offsets that would mask a meaningful amplitude threshold.
        self._save_collection_recording(raw, ts)

        pre = Preprocessor(self.preprocessing)
        filtered = pre.transform(raw)
        if not self._sanity_check(filtered, error_name="bad_collection_data"):
            self.state = State.IDLE
            self._publish_state()
            return

        session_id = self.collection_session_id
        assert session_id is not None
        self._push_status(f"collection_done:session={session_id}")
        self._publish_available_sessions()

        auto_train = self.collection_auto_train
        self.collection_auto_train = None
        if auto_train is None:
            self.state = State.IDLE
            self._publish_state()
            return

        self._run_training(session_id, auto_train)

    def _save_collection_recording(self, raw: np.ndarray, ts: np.ndarray) -> None:
        assert self.preprocessing is not None
        assert self.collection_session_id is not None

        markers = self.collection_trial_onsets
        eeg_info = self.eeg_rx.info if self.eeg_rx is not None else {}
        metadata = make_recording_metadata(
            session_id=self.collection_session_id,
            subject=self.collection_subject,
            recorded_at=datetime.now(timezone.utc),
            fs=self.preprocessing.fs,
            channel_order=list(self.preprocessing.channel_order),
            n_trials_per_class=int(self.cfg["calibration"]["trials_per_class"]),
            n_total_trials=len(markers),
            cue_s=float(self.cfg["calibration"]["cue_s"]),
            mi_s=float(self.cfg["calibration"]["mi_s"]),
            rest_s=float(self.cfg["calibration"]["rest_s"]),
            extra={
                "emotiv_total_channels": eeg_info.get(
                    "total_channels", self.preprocessing.n_channels
                ),
                "emotiv_channel_names": eeg_info.get(
                    "channel_names", list(self.preprocessing.channel_order)
                ),
            },
        )
        self.collection_recording_path = save_session_recording(
            self._session_root(),
            self.collection_session_id,
            raw=raw,
            timestamps=ts,
            marker_times=np.array(
                [t for t, _ in markers], dtype=np.float64
            ),
            marker_labels=np.array(
                [lbl for _, lbl in markers], dtype=np.int64
            ),
            metadata=metadata,
        )
        self._push_status(
            f"collection_saved:path={self.collection_recording_path}"
        )

    def _sanity_check(self, data: np.ndarray, *, error_name: str) -> bool:
        if data.size == 0 or data.shape[1] == 0:
            self._push_status(f"error:{error_name}:reason=empty")
            return False
        if self.sanity_nan_inf and not np.isfinite(data).all():
            self._push_status(f"error:{error_name}:reason=non_finite")
            return False
        max_abs = float(np.abs(data).max())
        if max_abs > self.sanity_max_abs_uv:
            self._push_status(
                f"error:{error_name}:reason=amp_exceeded:peak={max_abs:.1f}"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def _cmd_train(self, session_id: str, decoder_name: str) -> None:
        if self.state != State.IDLE:
            self._push_status(f"error:bad_state:{self.state.value}")
            return
        if not self._validate_training_request(session_id, [decoder_name]):
            return

        self._run_training(session_id, decoder_name)

    def _cmd_train_many(self, session_id: str, decoders_text: str) -> None:
        if self.state != State.IDLE:
            self._push_status(f"error:bad_state:{self.state.value}")
            return
        decoder_names = [
            name.strip()
            for name in decoders_text.replace(";", "|").split("|")
            if name.strip()
        ]
        if not self._validate_training_request(session_id, decoder_names):
            return

        self.state = State.TRAINING
        self._publish_state()
        try:
            for decoder_name in decoder_names:
                self._push_status(
                    f"training_started:session={session_id},decoder={decoder_name}"
                )
                try:
                    cv_acc, _bundle, bundle_path = self._train_one(
                        session_id, decoder_name
                    )
                    self._push_status(
                        "training_done:"
                        f"session={session_id},decoder={decoder_name},acc={cv_acc:.3f}"
                    )
                    logger.info(
                        "saved trained bundle for %s/%s to %s",
                        session_id,
                        decoder_name,
                        bundle_path,
                    )
                except Exception as exc:
                    logger.exception(
                        "training failed for %s/%s: %s",
                        session_id,
                        decoder_name,
                        exc,
                    )
                    reason = str(exc).replace(",", ";")
                    self._push_status(
                        "training_failed:"
                        f"session={session_id},decoder={decoder_name},reason={reason}"
                    )
            self._publish_available_bundles()
        finally:
            self.state = State.IDLE
            self._publish_state()

    def _validate_training_request(
        self, session_id: str, decoder_names: list[str]
    ) -> bool:
        if not session_id:
            self._push_status("error:train_session_missing")
            return False
        if not decoder_names:
            self._push_status("error:train_decoders_missing")
            return False
        for decoder_name in decoder_names:
            if decoder_name not in list_available():
                self._push_status(f"error:unknown_decoder:{decoder_name}")
                return False
        if not (self._session_root() / session_id / "recording.npz").exists():
            self._push_status(f"error:unknown_session:{session_id}")
            return False
        return True

    def _run_training(self, session_id: str, decoder_name: str) -> None:
        self.state = State.TRAINING
        self._publish_state()

        try:
            cv_acc, _bundle, bundle_path = self._train_one(
                session_id, decoder_name
            )
            logger.info("saved trained bundle to %s", bundle_path)
            self._push_status(
                "training_done:"
                f"session={session_id},decoder={decoder_name},acc={cv_acc:.3f}"
            )
            self._publish_available_bundles()
        except Exception as exc:
            logger.exception("training failed: %s", exc)
            self._push_status(f"error:training_failed:{exc}")
        finally:
            self.state = State.IDLE
            self._publish_state()

    def _train_one(
        self, session_id: str, decoder_name: str
    ) -> tuple[float, Bundle, Path]:
        recording = load_session_recording(self._session_root(), session_id)
        pre_cfg = build_preprocessing_config(
            self.cfg,
            fs=float(recording["fs"]),
            channel_order=list(recording["channel_order"]),
        )
        self._ensure_window_params(pre_cfg.fs)
        filtered = Preprocessor(pre_cfg).transform(recording["raw"])
        if not self._sanity_check(filtered, error_name="bad_collection_data"):
            raise RuntimeError("bad collection data")

        marker_pairs = list(
            zip(
                recording["marker_times"].astype(float).tolist(),
                recording["marker_labels"].astype(int).tolist(),
            )
        )
        cv_acc, bundle = self._fit_decoder(
            filtered=filtered,
            timestamps=recording["timestamps"],
            marker_pairs=marker_pairs,
            preprocessing=pre_cfg,
            session_id=session_id,
            recording_path=recording["recording_path"],
            session_metadata=recording["metadata"],
            decoder_name=decoder_name,
        )
        bundle_dir = self._bundle_root() / session_id / decoder_name
        bundle_path = save_bundle(bundle, bundle_dir)
        self.bundle = bundle
        self.bundle_path = bundle_path
        return cv_acc, bundle, bundle_path

    def _fit_decoder(
        self,
        *,
        filtered: np.ndarray,
        timestamps: np.ndarray,
        marker_pairs: list[tuple[float, int]],
        preprocessing: PreprocessingConfig,
        session_id: str,
        recording_path: Path,
        session_metadata: dict[str, Any],
        decoder_name: str,
    ) -> tuple[float, Bundle]:
        trials, labels = extract_trials(
            filtered,
            timestamps,
            marker_pairs,
            float(self.cfg["calibration"]["mi_s"]),
            preprocessing.fs,
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
        unique_groups = np.unique(group)
        if len(unique_groups) < 2:
            raise RuntimeError("need at least two trials for GroupKFold")
        n_splits = min(int(self.cfg["cv"]["n_splits"]), len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        accs: list[float] = []
        all_true: list[int] = []
        all_pred: list[int] = []
        decoder_params = self._decoder_params_for(decoder_name)
        for tr, te in splitter.split(X, y, groups=group):
            assert set(group[tr]).isdisjoint(set(group[te]))
            d = build_decoder(decoder_name, **decoder_params)
            d.fit(X[tr], y[tr])
            pred = d.predict(X[te])
            accs.append(float((pred == y[te]).mean()))
            all_true.extend(y[te].astype(int).tolist())
            all_pred.extend(pred.astype(int).tolist())
        cv_acc = float(np.mean(accs))
        cv_confusion = confusion_matrix(
            all_true, all_pred, labels=[0, 1]
        ).astype(int)
        logger.info("trial-level GroupKFold acc=%.3f folds=%s", cv_acc, accs)

        final = build_decoder(decoder_name, **decoder_params)
        final.fit(X, y)

        smoother_cfg = self.cfg["realtime"]["smoother"]
        subject = str(
            session_metadata.get("subject", session_id.split("_", 1)[0])
        )
        metadata = make_metadata(
            subject=subject,
            cal_acc=cv_acc,
            raw_data_path=recording_path,
            trial_len_s=float(self.cfg["calibration"]["mi_s"]),
            cue_s=float(self.cfg["calibration"]["cue_s"]),
            mi_s=float(self.cfg["calibration"]["mi_s"]),
            rest_s=float(self.cfg["calibration"]["rest_s"]),
            n_trials_per_class=int(
                session_metadata.get(
                    "n_trials_per_class",
                    self.cfg["calibration"]["trials_per_class"],
                )
            ),
            fs=preprocessing.fs,
            emotiv_n_channels=preprocessing.n_channels,
            emotiv_total_channels=int(
                session_metadata.get(
                    "emotiv_total_channels", preprocessing.n_channels
                )
            ),
            emotiv_channel_names=list(
                session_metadata.get(
                    "emotiv_channel_names", preprocessing.channel_order
                )
            ),
            physical_electrodes=list(preprocessing.channel_order),
            logical_channels=list(preprocessing.channel_order),
            extra={
                "session_id": session_id,
                "decoder_name": decoder_name,
                "decoder_params": final.get_params(),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "cv_acc": cv_acc,
                "cv_folds": accs,
                "cv_confusion": cv_confusion.tolist(),
                "trials_used": int(trials.shape[0]),
                "window_samples": self.window_samples,
                "stride_samples": self.calib_stride_samples,
                "preprocessing": {
                    "bandpass": list(preprocessing.bandpass),
                    "bandpass_order": preprocessing.bandpass_order,
                    "notch_hz": preprocessing.notch_hz,
                    "notch_q": preprocessing.notch_q,
                    "reference": preprocessing.reference,
                },
                "bandpass": (
                    f"{preprocessing.bandpass[0]:g}-"
                    f"{preprocessing.bandpass[1]:g}"
                ),
            },
        )
        return cv_acc, Bundle(
            decoder=final,
            decoder_name=decoder_name,
            decoder_params=final.get_params(),
            preprocessing=preprocessing,
            label_map={0: "left", 1: "right"},
            window_samples=self.window_samples,
            stride_samples=self.calib_stride_samples,
            smoother_name=str(smoother_cfg["name"]),
            smoother_params={
                k: v for k, v in smoother_cfg.items() if k != "name"
            },
            metadata=metadata,
        )

    def _decoder_params_for(self, decoder_name: str) -> dict[str, Any]:
        decoders_cfg = self.cfg.get("decoders", {}) or {}
        if decoder_name in decoders_cfg:
            return dict(decoders_cfg[decoder_name].get("params", {}) or {})
        dec_cfg = self.cfg.get("decoder", {}) or {}
        if decoder_name == dec_cfg.get("name", self.decoder_name):
            return dict(dec_cfg.get("params", self.decoder_params) or {})
        if decoder_name == self.decoder_name:
            return dict(self.decoder_params)
        return {}

    # ------------------------------------------------------------------
    # performance query
    # ------------------------------------------------------------------
    def _cmd_query_performance(self, subject: str) -> None:
        subject = subject.strip()
        if not subject:
            self._push_status("error:query_performance_subject_missing")
            return

        records = [
            item
            for item in find_bundles(self._bundle_root())
            if self._bundle_matches_subject(item, subject)
        ]
        records.sort(
            key=lambda item: float(item.get("acc") or -1.0),
            reverse=True,
        )

        if records:
            best = records[0]
            acc = best.get("acc")
            acc_part = "" if acc is None else f",acc={float(acc):.3f}"
            self._push_status(
                "performance_best:"
                f"subject={subject},session={best['session_id']},"
                f"decoder={best['decoder_name']}{acc_part}"
            )

        for item in records:
            self._push_status(self._format_performance_status(item))
        self._push_status(
            f"performance_done:subject={subject},count={len(records)}"
        )

    def _bundle_matches_subject(self, item: dict[str, Any], subject: str) -> bool:
        metadata = item.get("metadata", {})
        md_subject = str(metadata.get("subject", ""))
        session_id = str(item.get("session_id", ""))
        return md_subject == subject or session_id.startswith(f"{subject}_")

    def _format_performance_status(self, item: dict[str, Any]) -> str:
        def clean(value: Any) -> str:
            return str(value).replace(",", ";").replace("\n", " ").strip()

        metadata = item.get("metadata", {})
        acc = metadata.get("cv_acc", metadata.get("cal_acc", item.get("acc")))
        folds = [float(v) for v in metadata.get("cv_folds", [])]
        fold_min = min(folds) if folds else None
        fold_max = max(folds) if folds else None
        fold_text = "|".join(f"{v:.3f}" for v in folds)
        confusion = metadata.get("cv_confusion", [])
        confusion_text = ";".join(
            "|".join(str(int(cell)) for cell in row) for row in confusion
        )
        bandpass = metadata.get("bandpass")
        if isinstance(bandpass, (list, tuple)) and len(bandpass) >= 2:
            bandpass = f"{float(bandpass[0]):g}-{float(bandpass[1]):g}"
        if bandpass is None:
            bandpass = ""
        channel_text = "|".join(
            str(ch) for ch in metadata.get("logical_channels", [])
        )
        parts = [
            "performance:"
            f"session={item['session_id']}",
            f"decoder={item['decoder_name']}",
        ]
        if acc is not None:
            parts.append(f"acc={float(acc):.3f}")
        if fold_min is not None and fold_max is not None:
            parts.append(f"fold_min={fold_min:.3f}")
            parts.append(f"fold_max={fold_max:.3f}")
        if fold_text:
            parts.append(f"folds={fold_text}")
        parts.extend(
            [
                f"trials={int(metadata.get('trials_used', 0))}",
                f"trained_at={clean(metadata.get('trained_at', metadata.get('date', '')))}",
                f"confusion={confusion_text}",
                f"channels={channel_text}",
                f"window_samples={clean(metadata.get('window_samples', ''))}",
                f"raw_data_path={clean(metadata.get('raw_data_path', ''))}",
                f"bundle_path={clean(item.get('bundle_path', ''))}",
            ]
        )
        if bandpass:
            parts.append(f"bandpass={clean(bandpass)}")
        return ",".join(parts)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def _cmd_save_bundle(self, subject: str) -> None:
        if self.bundle is None:
            self._push_status("error:no_bundle")
            return
        session_id = str(
            self.bundle.metadata.get("session_id") or make_session_id(subject)
        )
        decoder_name = self.bundle.decoder_name
        bundle_path = save_bundle(
            self.bundle, self._bundle_root() / session_id / decoder_name
        )
        self._push_status(f"bundle_saved:path={bundle_path}")
        self._publish_available_bundles()

    def _cmd_load_bundle(
        self,
        session: str | None = None,
        decoder: str | None = None,
        path: str | None = None,
    ) -> None:
        if path is None and session and decoder is None:
            maybe_path = Path(session)
            if maybe_path.suffix == ".joblib" or maybe_path.exists():
                path = session
                session = None

        if path:
            bundle_ref = Path(path)
            if not bundle_ref.is_absolute():
                bundle_ref = REPO_ROOT / bundle_ref
        else:
            if not session:
                self._push_status("error:load_bundle_session_missing")
                return
            if not decoder:
                self._push_status("error:load_bundle_decoder_missing")
                return
            bundle_ref = self._bundle_root() / session / decoder

        bundle_file = (
            bundle_ref / "bundle.joblib" if bundle_ref.is_dir() else bundle_ref
        )
        if not bundle_file.exists():
            if session and decoder:
                self._push_status(f"error:unknown_bundle:{session}/{decoder}")
            else:
                self._push_status(f"error:bundle_not_found:{bundle_file}")
            return
        self.bundle = load_bundle(bundle_ref)
        # Re-derive the in-memory window/stride from the bundle so
        # realtime predictions use exactly what the bundle was trained
        # for, regardless of what configs/default.yaml says now.
        self.window_samples = self.bundle.window_samples
        self.preprocessing = self.bundle.preprocessing
        self.bundle_session_id = str(
            session or self.bundle.metadata.get("session_id", "?")
        )
        self.bundle_decoder_name = str(
            decoder or self.bundle.metadata.get(
                "decoder_name", self.bundle.decoder_name
            )
        )
        self.bundle_path = bundle_file
        self._push_status(
            "bundle_loaded:"
            f"session={self.bundle_session_id},decoder={self.bundle_decoder_name}"
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
        self._publish_now_running()
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
    def _resolve_repo_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return REPO_ROOT / path

    def _session_root(self) -> Path:
        data_dir = self._resolve_repo_path(self.cfg["paths"]["data_dir"])
        if data_dir.name.lower() == "eeg":
            return data_dir
        return data_dir / "eeg"

    def _bundle_root(self) -> Path:
        return self._resolve_repo_path(self.cfg["paths"]["bundles_dir"])

    def _ensure_window_params(self, fs: float) -> None:
        win_cfg = self.cfg["windowing"]
        if self.window_samples <= 0:
            self.window_samples = int(round(float(win_cfg["window_s"]) * fs))
        if self.realtime_stride_samples <= 0:
            self.realtime_stride_samples = int(
                round(int(win_cfg["realtime_stride_ms"]) / 1000.0 * fs)
            )
        if self.calib_stride_samples <= 0:
            self.calib_stride_samples = int(
                round(int(win_cfg["calib_stride_ms"]) / 1000.0 * fs)
            )

    def _publish_client_snapshot(self) -> None:
        self._publish_state()
        self._publish_service_config()
        self._push_status(f"decoders_available:{','.join(list_available())}")
        self._publish_available_sessions()
        self._publish_available_bundles()

    def _publish_available_sessions(self) -> None:
        session_ids = [
            str(item["session_id"]) for item in find_sessions(self._session_root())
        ]
        self._push_status(f"sessions_available:{','.join(session_ids)}")

    def _publish_available_bundles(self) -> None:
        entries: list[str] = []
        for item in find_bundles(self._bundle_root()):
            acc = item.get("acc")
            suffix = "" if acc is None else f":{float(acc):.3f}"
            entries.append(
                f"{item['session_id']}/{item['decoder_name']}{suffix}"
            )
        self._push_status(f"bundles_available:{','.join(entries)}")

    def _publish_now_running(self) -> None:
        if self.bundle is None:
            return
        session_id = self.bundle_session_id or str(
            self.bundle.metadata.get("session_id", "?")
        )
        decoder_name = self.bundle_decoder_name or self.bundle.decoder_name
        acc = self.bundle.metadata.get(
            "cv_acc", self.bundle.metadata.get("cal_acc")
        )
        acc_part = "" if acc is None else f",acc={float(acc):.3f}"
        self._push_status(
            f"now_running:session={session_id},decoder={decoder_name}{acc_part}"
        )

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
        ``start_collection`` so late joiners stay in sync. Single source
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
        if self.state == State.RUNNING:
            self._publish_now_running()
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
