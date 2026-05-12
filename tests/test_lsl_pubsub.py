"""Phase 4 LSL pub/sub tests: marker / command / status / proba.

Also covers the EEG-Marker clock-sync invariant (timestamps live on the
same LSL clock and align within ~50ms even when published from
separate outlets in the same process).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from pylsl import StreamInfo, StreamOutlet, local_clock, resolve_byprop, StreamInlet

from bci.lsl_io import (
    CommandReceiver,
    MarkerReceiver,
    ProbaPublisher,
    StatusPublisher,
)


def _make_marker_outlet(source_id: str, stream_type: str = "Markers") -> StreamOutlet:
    info = StreamInfo(
        name=f"TestMarkers_{source_id}",
        type=stream_type,
        channel_count=1,
        nominal_srate=0.0,
        channel_format="int32",
        source_id=source_id,
    )
    return StreamOutlet(info)


def _make_command_outlet(source_id: str, stream_type: str = "Commands") -> StreamOutlet:
    info = StreamInfo(
        name=f"TestCommands_{source_id}",
        type=stream_type,
        channel_count=1,
        nominal_srate=0.0,
        channel_format="string",
        source_id=source_id,
    )
    return StreamOutlet(info)


def test_marker_receiver_pulls_pairs() -> None:
    stream_type = "TestMarkers_mark_basic"
    outlet = _make_marker_outlet("mark_basic", stream_type=stream_type)
    try:
        rx = MarkerReceiver(stream_type=stream_type)
        rx.connect()
        time.sleep(0.3)
        t0 = local_clock()
        outlet.push_sample([0], t0)
        outlet.push_sample([1], t0 + 0.5)
        outlet.push_sample([99], t0 + 1.0)
        time.sleep(0.5)

        items = rx.pull()
        assert len(items) == 3
        labels = [lbl for _, lbl in items]
        assert labels == [0, 1, 99]
        ts = [t for t, _ in items]
        assert ts[0] == pytest.approx(t0, abs=0.01)
        # Subsequent pull is empty (queue drained).
        assert rx.pull() == []
    finally:
        del outlet


def test_command_receiver_pulls_strings() -> None:
    stream_type = "TestCommands_cmd_basic"
    outlet = _make_command_outlet("cmd_basic", stream_type=stream_type)
    try:
        rx = CommandReceiver(stream_type=stream_type)
        rx.connect()
        time.sleep(0.3)
        outlet.push_sample(["start_collection:subject=A"])
        outlet.push_sample(["shutdown"])
        time.sleep(0.5)

        cmds = rx.pull()
        assert cmds == ["start_collection:subject=A", "shutdown"]
        assert rx.pull() == []
    finally:
        del outlet


def test_eeg_marker_clock_sync_within_50ms() -> None:
    """Both EEG and Marker timestamps live on the LSL local_clock so a
    same-instant push must come back within 50ms after pylsl drift
    correction (this is far above the typical sub-ms loopback skew)."""
    eeg_info = StreamInfo(
        name="TestSyncEEG",
        type="TestSyncEEG",
        channel_count=1,
        nominal_srate=128.0,
        channel_format="float32",
        source_id="sync_eeg",
    )
    eeg_outlet = StreamOutlet(eeg_info)
    marker_type = "TestMarkers_sync_mark"
    marker_outlet = _make_marker_outlet("sync_mark", stream_type=marker_type)

    eeg_streams = resolve_byprop("type", "TestSyncEEG", timeout=5.0)
    eeg_inlet = StreamInlet(
        next(s for s in eeg_streams if s.source_id() == "sync_eeg")
    )
    eeg_inlet.open_stream(timeout=2.0)

    rx_marker = MarkerReceiver(stream_type=marker_type)
    rx_marker.connect()
    time.sleep(0.3)
    try:
        diffs: list[float] = []
        for k in range(5):
            t = local_clock()
            eeg_outlet.push_sample([float(k)], t)
            marker_outlet.push_sample([k % 2], t)
            time.sleep(0.05)

            samples, eeg_ts = eeg_inlet.pull_chunk(timeout=0.5, max_samples=16)
            assert samples, "EEG sample missing"
            mark_items = rx_marker.pull()
            assert mark_items, "marker missing"

            # Compare the most recent of each.
            eeg_t = float(eeg_ts[-1])
            mark_t = mark_items[-1][0]
            diffs.append(abs(eeg_t - mark_t))

        avg = float(np.mean(diffs))
        worst = float(np.max(diffs))
        # Tight bound: in-process loopback should be sub-ms but we leave
        # plenty of slack for CI variance.
        assert worst < 0.05, f"worst skew {worst*1000:.2f}ms exceeds 50ms"
        assert avg < 0.02, f"avg skew {avg*1000:.2f}ms exceeds 20ms"
    finally:
        del eeg_outlet
        del marker_outlet


def test_status_publisher_push_round_trip() -> None:
    pub = StatusPublisher(name="TestStatus", source_id="status_basic")
    try:
        streams = resolve_byprop("type", "Status", timeout=5.0)
        inlet = StreamInlet(
            next(s for s in streams if s.source_id() == "status_basic")
        )
        inlet.open_stream(timeout=2.0)
        time.sleep(0.3)
        pub.push("state:COLLECTING")
        pub.push("collection_progress:5/40")
        time.sleep(0.5)

        samples, _ts = inlet.pull_chunk(timeout=0.5, max_samples=16)
        msgs = [s[0] for s in samples]
        assert "state:COLLECTING" in msgs
        assert "collection_progress:5/40" in msgs
    finally:
        del pub


def test_proba_publisher_push_round_trip() -> None:
    pub = ProbaPublisher(n_classes=2, name="TestProba", source_id="proba_basic")
    try:
        streams = resolve_byprop("type", "BCI_Proba", timeout=5.0)
        inlet = StreamInlet(
            next(s for s in streams if s.source_id() == "proba_basic")
        )
        inlet.open_stream(timeout=2.0)
        time.sleep(0.3)
        pub.push(np.array([0.7, 0.3], dtype=np.float32))
        pub.push(np.array([0.2, 0.8], dtype=np.float32))
        time.sleep(0.5)

        samples, _ts = inlet.pull_chunk(timeout=0.5, max_samples=16)
        arr = np.array(samples)
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr[0], [0.7, 0.3], atol=1e-5)
        np.testing.assert_allclose(arr[1], [0.2, 0.8], atol=1e-5)
    finally:
        del pub


def test_proba_publisher_rejects_wrong_size() -> None:
    pub = ProbaPublisher(n_classes=2, name="TestProbaBad", source_id="proba_bad")
    try:
        with pytest.raises(ValueError):
            pub.push(np.array([0.3, 0.3, 0.4], dtype=np.float32))
    finally:
        del pub
