"""Publish a short Status stream sequence for Unity progress UI checks.

Run while Unity is in Play mode:

    python scripts/dev_status_progress.py
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

from pylsl import StreamInfo, StreamOutlet

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bci.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("bci.dev_status_progress")


def main() -> int:
    setup_logging(REPO_ROOT / "logs", level="INFO")

    info = StreamInfo(
        name="DevStatusProgress",
        type="Status",
        channel_count=1,
        nominal_srate=0.0,
        channel_format="string",
        source_id="dev_status_progress",
    )
    outlet = StreamOutlet(info)
    messages = [
        "state:COLLECTING",
        "collection_progress:0/40",
        "collection_progress:5/40",
        "collection_progress:10/40",
        "collection_done:session=S001_20260509T123000",
        "training_done:session=S001_20260509T123000,decoder=csp_lda,acc=0.823",
        "error:test_warning",
    ]

    stop = False

    def _on_signal(signum, frame):  # noqa: ANN001
        nonlocal stop
        stop = True
        logger.info("Caught signal %d, stopping", signum)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Give Unity's resolver a moment to see the new stream.
    time.sleep(1.0)
    for message in messages:
        if stop:
            break
        outlet.push_sample([message])
        logger.info("status: %s", message)
        time.sleep(1.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
