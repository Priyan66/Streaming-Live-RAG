import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from engine import Runtime  # noqa: E402
from simulator.stream_replay import replay  # noqa: E402
from telemetry.logger import TraceLogger  # noqa: E402


@pytest.fixture(scope="session")
def rt():
    return Runtime()


@pytest.fixture
def run(rt):
    """Replay an examples/*.json scenario; returns (outputs, events)."""

    def _run(name: str, **overrides):
        logger = TraceLogger()
        eng = rt.engine(logger, **overrides)
        sc = json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))
        outs = asyncio.run(replay(eng, sc))
        return outs, logger.events

    return _run


def events_of(events, turn, kind=None):
    return [e for e in events if e.get("turn") == turn and (kind is None or e["event"] == kind)]
