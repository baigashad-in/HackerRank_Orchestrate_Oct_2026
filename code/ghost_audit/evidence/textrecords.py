"""Parser for plain-text operational records (shift handover notes and similar)."""

from __future__ import annotations

import re
from typing import List, Tuple

from ..loaders import EvidenceFile
from .base import Observation, SnapshotScope
from .records import observation_from_lookup

SNAPSHOT_RE = re.compile(r"Snapshot:\s*(\S+)")
ROW_RE = re.compile(r"^\s*[-*]?\s*(?P<ts>\d{4}-\d{2}-\d{2}T[\d:]+Z)\s*\|\s*(?P<system>[\w-]+)\s*\|\s*(?P<rid>[\w-]+)\s*\|\s*"
                    r"(?P<rtype>[\w-]+)\s*\|\s*(?P<result>[\w-]+)\s*\|\s*(?P<subject>.*?)\s*$")


def parse_text(ef: EvidenceFile) -> Tuple[List[Observation], List[SnapshotScope]]:
    text = open(ef.abs_path, encoding="utf-8", errors="replace").read()
    m = SNAPSHOT_RE.search(text)
    default_ts = m.group(1) if m else ef.captured_at
    out: List[Observation] = []
    for line in text.splitlines():
        row = ROW_RE.match(line)
        if not row:
            continue
        obs = observation_from_lookup(ef.file_id, row.group("ts") or default_ts, row.group("system"), row.group("rid"),
                                      row.group("rtype"), row.group("result"), row.group("subject"),
                                      source="text_handover")
        if obs:
            out.append(obs)
    return out, []
