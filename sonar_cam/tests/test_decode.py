import struct
from pathlib import Path

import numpy as np
import pytest

from sonar_cam.data import BAG, Bag, TOPIC_SONAR, decode_ping

pytestmark = pytest.mark.skipif(not Path(BAG).exists(), reason="bus_outside bag not available")


def build_ping(t_sec=1494876479, nsec=809775000, head=-126.0, max_range=6.0, n=250):
    """Hand-built ProcessedRange CDR with the layout documented in decode_ping."""
    d = struct.pack("<4B", 0, 1, 0, 0) + struct.pack("<IIi", t_sec, nsec, 6) + b"sonar\0" + b"\0\0"
    d += struct.pack("<I", n) + bytes(range(n)) [:n] + b"\0" * ((-n) % 4)
    d += struct.pack("<I", 256) + struct.pack("<fIf", head, 1, max_range) + b"\0" * 20
    return d


def test_synthetic_roundtrip():
    p = decode_ping(build_ping())
    assert p.head_deg == pytest.approx(-126.0) and p.max_range_m == 6.0 and len(p.intensity) == 250
    assert p.t == pytest.approx(1494876479.809775)


def test_real_messages():
    bag = Bag()
    rows = bag.rowids(TOPIC_SONAR)
    blobs = [bag.blob(r) for r in rows[:50]]
    pings = [decode_ping(b) for b in blobs]
    assert all(len(p.intensity) == 250 and p.max_range_m == 6.0 for p in pings)
    head = np.array([p.head_deg for p in pings])
    assert head[0] == pytest.approx(-126.0)
    steps = np.diff(head)
    assert np.allclose(steps[steps > 0], 0.9, atol=1e-3)  # 0.9 deg per step (400 steps / revolution)
    t = np.array([p.t for p in pings])
    assert np.all(np.diff(t) > 0) and 0.005 < np.median(np.diff(t)) < 0.02  # ~95 Hz
    # header stamp equals bag receive time to < 1 us (so either clock may be used)
    ts = bag.con.execute("select timestamp from messages where id=?", (int(rows[0]),)).fetchone()[0] * 1e-9
    assert abs(ts - pings[0].t) < 1e-6
