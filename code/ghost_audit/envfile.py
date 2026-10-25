"""Minimal ``.env`` loader (no third-party dependency).

The challenge rules ask for credentials to be read from environment variables,
with an *uncommitted* ``.env`` file when convenient.  ``load_dotenv`` reads
``KEY=VALUE`` lines from ``<repo>/.env`` (and, if different, ``./.env``) into
``os.environ`` without overriding variables that are already set.  Values are
never printed or logged; callers only learn which keys were loaded.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List

LINE_RE = re.compile(r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$""")


def _parse_value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    # strip an unquoted trailing comment
    if " #" in raw:
        raw = raw.split(" #", 1)[0].rstrip()
    return raw


def load_dotenv(directories: Iterable[str]) -> List[str]:
    """Load the first ``.env`` found in ``directories``. Returns the names of the keys that were set."""
    loaded: List[str] = []
    seen = set()
    for directory in directories:
        path = os.path.join(os.path.abspath(directory), ".env")
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                m = LINE_RE.match(line)
                if not m:
                    continue
                key, value = m.group(1), _parse_value(m.group(2))
                if not value:
                    continue   # "ANTHROPIC_API_KEY=" left blank in the template: not a key
                if key in os.environ and os.environ[key] != "":
                    continue   # a real environment variable always wins
                os.environ[key] = value
                loaded.append(key)
        break
    return loaded
