from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop

from .preprocessing import Preprocessor

logger = logging.getLogger(__name__)

# Emotiv puts (counter, interpolated, signal_quality) ahead of EEG.
EMOTIV_CHANNEL_OFFSET = 3
# Only EPOC X is supported in this project. EPOC Flex (7-channel total
# stream) is intentionally rejected.
EMOTIV_LAYOUTS: dict[int, int] = {
    19: 14,  # EPOC X (14 EEG channels)
}
EMOTIV_X_MONTAGE: tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)


class RingBuffer:
    """Thread-safe circular buffer for streaming EEG samples.

    Storage layout is ``(n_channels, capacity)``. ``push`` accepts the
    pylsl-style ``(n_samples, n_channels)`` chunk shape and transposes
    internally. All public methods take an internal lock so concurrent
    push/get operations cannot tear a partial chunk read.
    """

    def __init__(self, n_channels: int, capacity_samples: int) -> None:
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.n_channels = int(n_channels)
        self.capacity = int(capacity_samples)
        self._buf = np.zeros((self.n_channels, self.capacity), dtype=np.float64)
        self._write_pos = 0  # next index to write
        self._filled = 0     # number of valid samples (<= capacity)
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"push expects (n_samples, n_channels); got shape {arr.shape}"
            )
        if arr.shape[1] != self.n_channels:
            raise ValueError(
                f"chunk has {arr.shape[1]} channels, buffer has {self.n_channels}"
            )
        n = arr.shape[0]
        if n == 0:
            return
        # Drop any oldest excess if a single chunk would overflow capacity;
        # only the trailing `capacity` samples can survive anyway.
        if n > self.capacity:
            arr = arr[-self.capacity :]
            n = self.capacity
        chunk_T = arr.T  # (n_channels, n)

        with self._lock:
            wp = self._write_pos
            tail = self.capacity - wp
            if n <= tail:
                self._buf[:, wp : wp + n] = chunk_T
            else:
                self._buf[:, wp:] = chunk_T[:, :tail]
                self._buf[:, : n - tail] = chunk_T[:, tail:]
            self._write_pos = (wp + n) % self.capacity
            self._filled = min(self._filled + n, self.capacity)

    def get_last(self, n_samples: int) -> np.ndarray:
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if n_samples == 0:
            return np.zeros((self.n_channels, 0), dtype=self._buf.dtype)
        with self._lock:
            if n_samples > self._filled:
                raise ValueError(
                    f"requested {n_samples} samples but only {self._filled} available"
                )
            end = self._write_pos
            start = (end - n_samples) % self.capacity
            if start < end:
                return self._buf[:, start:end].copy()
            return np.concatenate(
                [self._buf[:, start:], self._buf[:, :end]], axis=1
            )

    def n_filled(self) -> int:
        with self._lock:
            return self._filled

    def clear(self) -> None:
        with self._lock:
            self._filled = 0
            self._write_pos = 0
            self._buf.fill(0.0)


class EEGReceiver:
    """Background thread that pulls EEG chunks from an LSL inlet.

    Maintains two ring buffers:
      - raw_ring_buffer: post-offset EEG channels, no filtering.
      - filtered_ring_buffer: same chunks pushed through an attached
        Preprocessor with persistent zi state.

    The "filter on the receiver thread" pattern is what avoids per-window
    transients during realtime: the filter state stays continuous across
    chunk boundaries, so callers can grab any window from
    filtered_ring_buffer and feed it to the decoder unchanged.
    """

    def __init__(
        self,
        stream_type: str = "EEG",
        capacity_seconds: float = 5.0,
        stall_timeout_s: float = 2.0,
        resolve_timeout_s: float = 5.0,
        pull_timeout_s: float = 0.1,
        max_samples_per_pull: int = 64,
        channel_selection: list[str] | None = None,
    ) -> None:
        self.stream_type = stream_type
        self.capacity_seconds = float(capacity_seconds)
        self.stall_timeout_s = float(stall_timeout_s)
        self.resolve_timeout_s = float(resolve_timeout_s)
        self.pull_timeout_s = float(pull_timeout_s)
        self.max_samples_per_pull = int(max_samples_per_pull)
        # `channel_selection`: subset of EEG channel names to keep. When
        # set, downstream callers see only those channels (in the given
        # order); when None, all 14 EPOC X channels pass through.
        self.channel_selection = (
            list(channel_selection) if channel_selection else None
        )

        self._inlet: StreamInlet | None = None
        self._info: dict[str, Any] = {}
        self._raw_rb: RingBuffer | None = None
        self._filtered_rb: RingBuffer | None = None
        # Indices into the post-offset EEG block that survive selection.
        self._sel_indices: np.ndarray | None = None

        self._preprocessor: Preprocessor | None = None
        self._pre_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        self._stall_lock = threading.Lock()
        self._last_chunk_t: float | None = None  # set once streaming starts

        self._record_lock = threading.Lock()
        self._recording = False
        self._record_chunks: list[np.ndarray] = []
        self._record_ts: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # connection / setup
    # ------------------------------------------------------------------
    def connect(self) -> dict[str, Any]:
        streams = resolve_byprop(
            "type", self.stream_type, timeout=self.resolve_timeout_s
        )
        if not streams:
            raise RuntimeError(
                f"No LSL stream of type={self.stream_type!r} found within "
                f"{self.resolve_timeout_s}s"
            )
        self._inlet = StreamInlet(streams[0])
        info = self._inlet.info()
        total_channels = info.channel_count()
        fs = float(info.nominal_srate())
        manufacturer = info.desc().child_value("manufacturer") or ""

        if manufacturer == "Emotiv":
            if total_channels not in EMOTIV_LAYOUTS:
                raise RuntimeError(
                    f"Emotiv stream has unexpected channel_count="
                    f"{total_channels}; expected one of "
                    f"{sorted(EMOTIV_LAYOUTS)} (EPOC X only)"
                )
            n_eeg_full = EMOTIV_LAYOUTS[total_channels]
            offset = EMOTIV_CHANNEL_OFFSET
        else:
            n_eeg_full = total_channels
            offset = 0

        all_labels: list[str] = []
        ch = info.desc().child("channels").child("channel")
        while not ch.empty():
            all_labels.append(ch.child_value("label"))
            ch = ch.next_sibling("channel")
        eeg_labels_full = all_labels[offset : offset + n_eeg_full]

        if self.channel_selection is not None:
            sel_idx: list[int] = []
            for name in self.channel_selection:
                if name not in eeg_labels_full:
                    raise RuntimeError(
                        f"requested channel {name!r} not in stream EEG labels "
                        f"{eeg_labels_full}"
                    )
                sel_idx.append(eeg_labels_full.index(name))
            self._sel_indices = np.asarray(sel_idx, dtype=np.int64)
            channel_names = list(self.channel_selection)
            n_channels = len(channel_names)
        else:
            self._sel_indices = None
            channel_names = list(eeg_labels_full)
            n_channels = n_eeg_full

        self._info = {
            "fs": fs,
            "n_channels": n_channels,
            "channel_names": channel_names,
            "all_eeg_channel_names": eeg_labels_full,
            "manufacturer": manufacturer,
            "total_channels": total_channels,
            "channel_offset": offset,
            "stream_name": info.name(),
        }
        capacity = max(1, int(round(self.capacity_seconds * fs)))
        self._raw_rb = RingBuffer(n_channels=n_channels, capacity_samples=capacity)
        self._filtered_rb = RingBuffer(
            n_channels=n_channels, capacity_samples=capacity
        )
        logger.info("EEGReceiver connected: %s", self._info)
        return dict(self._info)

    @property
    def info(self) -> dict[str, Any]:
        if not self._info:
            raise RuntimeError("connect() must be called first")
        return dict(self._info)

    # ------------------------------------------------------------------
    # preprocessor attachment
    # ------------------------------------------------------------------
    def attach_preprocessor(self, pre: Preprocessor) -> None:
        if not self._info:
            raise RuntimeError("connect() first")
        pre.reset_filter_state(self._info["n_channels"])
        with self._pre_lock:
            self._preprocessor = pre

    def detach_preprocessor(self) -> None:
        with self._pre_lock:
            self._preprocessor = None

    # ------------------------------------------------------------------
    # streaming
    # ------------------------------------------------------------------
    def start_streaming(self) -> None:
        if self._inlet is None:
            raise RuntimeError("connect() first")
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("already streaming")
        self._stop_evt.clear()
        with self._stall_lock:
            self._last_chunk_t = time.time()
        self._thread = threading.Thread(
            target=self._run, name="EEGReceiver", daemon=True
        )
        self._thread.start()

    def stop_streaming(self, timeout_s: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stop_evt.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None

    def _run(self) -> None:
        offset = self._info["channel_offset"]
        n_eeg_full = len(self._info["all_eeg_channel_names"])
        eeg_end = offset + n_eeg_full
        sel = self._sel_indices  # None or 1D index array
        assert self._inlet is not None
        assert self._raw_rb is not None
        assert self._filtered_rb is not None

        while not self._stop_evt.is_set():
            try:
                samples, timestamps = self._inlet.pull_chunk(
                    timeout=self.pull_timeout_s,
                    max_samples=self.max_samples_per_pull,
                )
            except Exception as exc:  # pragma: no cover -- pylsl runtime issue
                logger.exception("pull_chunk failed: %s", exc)
                continue
            if not samples:
                continue
            arr = np.asarray(samples, dtype=np.float64)
            ts = np.asarray(timestamps, dtype=np.float64)
            eeg = arr[:, offset:eeg_end]
            if sel is not None:
                eeg = eeg[:, sel]

            with self._stall_lock:
                self._last_chunk_t = time.time()

            self._raw_rb.push(eeg)
            with self._pre_lock:
                pre = self._preprocessor
            if pre is not None:
                filtered = pre.transform_chunk(eeg)
                self._filtered_rb.push(filtered)

            if self._recording:
                with self._record_lock:
                    if self._recording:
                        self._record_chunks.append(eeg.copy())
                        self._record_ts.append(ts.copy())

    # ------------------------------------------------------------------
    # stall detection
    # ------------------------------------------------------------------
    def is_stalled(self) -> bool:
        with self._stall_lock:
            t = self._last_chunk_t
        if t is None:
            return False
        return (time.time() - t) > self.stall_timeout_s

    # ------------------------------------------------------------------
    # realtime accessors
    # ------------------------------------------------------------------
    def get_last_raw(self, n_samples: int) -> np.ndarray:
        if self._raw_rb is None:
            raise RuntimeError("connect() first")
        return self._raw_rb.get_last(n_samples)

    def get_last_filtered(self, n_samples: int) -> np.ndarray:
        if self._filtered_rb is None:
            raise RuntimeError("connect() first")
        return self._filtered_rb.get_last(n_samples)

    def n_filled_raw(self) -> int:
        if self._raw_rb is None:
            return 0
        return self._raw_rb.n_filled()

    def n_filled_filtered(self) -> int:
        if self._filtered_rb is None:
            return 0
        return self._filtered_rb.n_filled()

    # ------------------------------------------------------------------
    # calibration recording
    # ------------------------------------------------------------------
    def start_recording(self) -> None:
        with self._record_lock:
            self._record_chunks.clear()
            self._record_ts.clear()
            self._recording = True

    def stop_recording(self) -> tuple[np.ndarray, np.ndarray]:
        with self._record_lock:
            self._recording = False
            if not self._record_chunks:
                n_eeg = self._info.get("n_channels", 0)
                return (
                    np.zeros((n_eeg, 0), dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                )
            chunks = self._record_chunks
            tss = self._record_ts
            self._record_chunks = []
            self._record_ts = []
        # Concatenate outside the lock so the worker thread is unblocked.
        full_eeg = np.concatenate(chunks, axis=0).T  # (n_channels, T)
        full_ts = np.concatenate(tss, axis=0)
        return full_eeg, full_ts


def _resolve_one(
    stream_type: str, timeout_s: float, open_timeout_s: float = 2.0
) -> StreamInlet:
    streams = resolve_byprop("type", stream_type, timeout=timeout_s)
    if not streams:
        raise RuntimeError(
            f"No LSL stream of type={stream_type!r} found within {timeout_s}s"
        )
    inlet = StreamInlet(streams[0])
    # Block until the inlet has actually negotiated with the outlet so
    # subsequent pull_chunk calls don't silently lose pre-handshake
    # samples. Without this, in-process loopback tests are racy.
    inlet.open_stream(timeout=open_timeout_s)
    return inlet


class MarkerReceiver:
    """Pulls integer markers from an LSL stream.

    The publisher (Unity) is expected to push (timestamp_lsl, label)
    pairs where label encodes left/right MI cue or rest. We keep the
    LSL clock timestamp because it lives on the same clock as the EEG
    timestamps once pylsl applies its drift correction.
    """

    def __init__(
        self,
        stream_type: str = "Markers",
        resolve_timeout_s: float = 5.0,
        pull_timeout_s: float = 0.0,
        max_samples_per_pull: int = 64,
    ) -> None:
        self.stream_type = stream_type
        self.resolve_timeout_s = float(resolve_timeout_s)
        self.pull_timeout_s = float(pull_timeout_s)
        self.max_samples_per_pull = int(max_samples_per_pull)
        self._inlet: StreamInlet | None = None

    def connect(self) -> None:
        self._inlet = _resolve_one(self.stream_type, self.resolve_timeout_s)
        logger.info("MarkerReceiver connected to type=%s", self.stream_type)

    def pull(self) -> list[tuple[float, int]]:
        if self._inlet is None:
            raise RuntimeError("connect() first")
        samples, timestamps = self._inlet.pull_chunk(
            timeout=self.pull_timeout_s, max_samples=self.max_samples_per_pull
        )
        if not samples:
            return []
        out: list[tuple[float, int]] = []
        for sample, t in zip(samples, timestamps):
            out.append((float(t), int(sample[0])))
        return out


class CommandReceiver:
    """Pulls string commands from Unity (start_collection, train, etc.)."""

    def __init__(
        self,
        stream_type: str = "Commands",
        resolve_timeout_s: float = 5.0,
        pull_timeout_s: float = 0.0,
        max_samples_per_pull: int = 16,
    ) -> None:
        self.stream_type = stream_type
        self.resolve_timeout_s = float(resolve_timeout_s)
        self.pull_timeout_s = float(pull_timeout_s)
        self.max_samples_per_pull = int(max_samples_per_pull)
        self._inlet: StreamInlet | None = None

    def connect(self) -> None:
        self._inlet = _resolve_one(self.stream_type, self.resolve_timeout_s)
        logger.info("CommandReceiver connected to type=%s", self.stream_type)

    def pull(self) -> list[str]:
        if self._inlet is None:
            raise RuntimeError("connect() first")
        samples, _ts = self._inlet.pull_chunk(
            timeout=self.pull_timeout_s, max_samples=self.max_samples_per_pull
        )
        if not samples:
            return []
        return [str(s[0]) for s in samples]


class StatusPublisher:
    """Publishes service-state strings for Unity to display."""

    def __init__(
        self,
        name: str = "BCI_Status",
        source_id: str = "bci_status",
    ) -> None:
        info = StreamInfo(
            name=name,
            type="Status",
            channel_count=1,
            nominal_srate=0.0,
            channel_format="string",
            source_id=source_id,
        )
        self._outlet = StreamOutlet(info)

    def push(self, msg: str) -> None:
        self._outlet.push_sample([msg])


class ProbaPublisher:
    """Publishes smoothed class probabilities (one frame per chunk)."""

    def __init__(
        self,
        n_classes: int = 2,
        name: str = "BCI_Proba",
        source_id: str = "bci_proba",
    ) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        info = StreamInfo(
            name=name,
            type="BCI_Proba",
            channel_count=n_classes,
            nominal_srate=0.0,
            channel_format="float32",
            source_id=source_id,
        )
        self._outlet = StreamOutlet(info)
        self.n_classes = n_classes

    def push(self, proba: np.ndarray) -> None:
        arr = np.asarray(proba, dtype=np.float32).reshape(-1)
        if arr.size != self.n_classes:
            raise ValueError(
                f"proba size {arr.size} != n_classes {self.n_classes}"
            )
        self._outlet.push_sample(arr.tolist())
