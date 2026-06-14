"""Mock EPOC X EEG LSL outlet with an intentional amplitude outlier.

Use this to verify Phase 5 calibration sanity checks:

    python scripts/dev_fake_eeg_bad.py

Then start service calibration and markers. The emitted EEG carries a
large microvolt value, so service.py should publish
``error:bad_collection_data:reason=amp_exceeded`` and return to IDLE.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bci.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("bci.dev_fake_eeg_bad")

CHANNEL_OFFSET = 3
N_EEG = 14
EXTRA_CHANNELS = 2
EPOCX_LABELS: tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)


def _build_stream_info(fs: float, name: str, source_id: str) -> tuple[StreamInfo, int]:
    total = CHANNEL_OFFSET + N_EEG + EXTRA_CHANNELS
    info = StreamInfo(
        name=name,
        type="EEG",
        channel_count=total,
        nominal_srate=fs,
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
    for label in labels:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    return info, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--name", default="BadMockEmotiv")
    parser.add_argument("--source-id", default="bad_mock_emotiv")
    parser.add_argument("--bad-channel", default="FC5")
    parser.add_argument("--bad-uv", type=float, default=1000.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 = run until Ctrl-C")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.bad_channel not in EPOCX_LABELS:
        parser.error(f"--bad-channel must be one of {EPOCX_LABELS}")
    bad_idx = EPOCX_LABELS.index(args.bad_channel)

    setup_logging(REPO_ROOT / "logs", level="INFO")
    info, total_channels = _build_stream_info(args.fs, args.name, args.source_id)
    outlet = StreamOutlet(info, chunk_size=16, max_buffered=360)
    logger.info(
        "Streaming bad EEG (total_ch=%d, bad_channel=%s, bad_uv=%.1f)",
        total_channels,
        args.bad_channel,
        args.bad_uv,
    )

    stop = False

    def _on_signal(signum, frame):  # noqa: ANN001
        nonlocal stop
        stop = True
        logger.info("Caught signal %d, stopping", signum)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    rng = np.random.default_rng(args.seed)
    chunk_n = 16
    period = chunk_n / args.fs
    sample_idx = 0
    t_next = time.time()
    t0 = time.time()
    while not stop:
        if args.duration > 0 and (time.time() - t0) >= args.duration:
            break
        block = np.zeros((chunk_n, total_channels), dtype=np.float32)
        block[:, 0] = (np.arange(sample_idx, sample_idx + chunk_n) % 256).astype(
            np.float32
        )
        block[:, 2] = 1.0
        eeg = rng.standard_normal((chunk_n, N_EEG)).astype(np.float32) * 1.0
        eeg[:, bad_idx] = float(args.bad_uv)
        block[:, CHANNEL_OFFSET : CHANNEL_OFFSET + N_EEG] = eeg
        outlet.push_chunk(block.tolist(), local_clock())
        sample_idx += chunk_n

        t_next += period
        sleep_for = t_next - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            t_next = time.time()

    logger.info("Stopped after pushing %d samples", sample_idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
