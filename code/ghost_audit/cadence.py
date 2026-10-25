"""The deletion worker's timing, learned from the participant data at run time.

Unlabeled worker rows (an anonymised analytics profile, a 404 account) can only
be tied to a request through *when* they happened: the worker processes one
request as a run of steps on a regular schedule after the account step. That
schedule is not hardcoded. It is learned on every run from the requests whose
account is pinned to the person by an identifier (no answer labels are used):

* **families**: worker message templates (ids and numbers removed) that share
  a step gap form one family. A template's single-step gaps are the values
  that make up at least 10% of the gaps between its consecutive pinned rows.
  Templates without gap data join the family whose seconds pattern explains
  their rows best, or the largest family;
* **gaps**: the single-step gap range of each family (``gaps``), and the
  seconds a step adds to the clock (``step_seconds``, gap mod 60);
* **base**: the second of the minute at which a family's first step lands,
  relative to the account step (the mode of the earliest on-family row after
  each pinned account step, merging +-1 s);
* **first-step delay**: the window of minutes between the account step and a
  family's first step: the 5th to 95th percentile of the on-base earliest
  rows, widened by two step gaps on each side (``WINDOW_RULE``, chosen by
  cross-validation, see ``code/evaluation/holdout_tests.py``).

If there are too few pinned requests to learn from (fewer than
``MIN_ANCHORS``), the engine falls back to ``DEFAULT_CADENCE``, which holds
the values measured on this dataset, and says so in its warnings.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MIN_ANCHORS = 20          # pinned requests needed before learned values replace the defaults
MIN_FAMILY_SAMPLES = 8    # earliest-row samples needed to learn a family's base and delay window
DRIFT = 1
# Delay window rule: (low quantile, high quantile, step gaps added on each side). Chosen by the 2-fold
# hold-out in code/evaluation/holdout_tests.py among seven candidates (six learned rules and the measured
# defaults), ranking first by wrongly attributed worker rows (in an audit, reporting another person's record
# is worse than missing one), then by accounts and rows recovered. The 5th-95th percentile widened by two
# step gaps had the fewest wrong attributions and recovered as many accounts as the defaults; the full range
# let one outlier stretch the window and accepted implausible 95-100 minute delays.
WINDOW_RULE = (0.05, 0.95, 2.0)


def template(details: str) -> str:
    """Worker message with ids, system names and numbers removed."""
    t = re.sub(r"\b[a-z]+(?:_[a-z]+)*_\d+\b", "<id>", details or "")
    t = re.sub(r"\bfrom [a-z_]+\.", "from <system>.", t)
    t = re.sub(r"\bto [a-z_]+\.", "to <system>.", t)
    return re.sub(r"\d+", "<n>", t).strip()


@dataclass
class Cadence:
    family_base: Dict[str, int]                        # family -> second-of-minute of its first step
    step_seconds: Tuple[int, ...]                      # seconds each further step adds (mod 60)
    gaps: Dict[str, Tuple[int, int]]                   # family -> single-step gap range (seconds)
    first_offset: Dict[str, Tuple[float, float, float, float]]   # lo, hi minutes; step lo, hi minutes
    template_family: Dict[str, str] = field(default_factory=dict)
    default_family: str = "F"
    source: str = "default"
    anchors: int = 0
    notes: List[str] = field(default_factory=list)

    def family(self, details: str) -> str:
        if not self.template_family:
            # default rule measured on this dataset: policy confirmations run on their own schedule
            return "P" if (details or "").startswith("Policy action completed") else "F"
        return self.template_family.get(template(details), self.default_family)

    def describe(self) -> List[str]:
        out = [f"cadence source: {self.source} ({self.anchors} pinned requests)"]
        for fam in sorted(self.family_base):
            lo, hi, slo, shi = self.first_offset[fam]
            g = self.gaps[fam]
            tpls = sorted(t for t, f in self.template_family.items() if f == fam)
            out.append(f"  family {fam}: first step at second {self.family_base[fam]}, step gap {g[0]}-{g[1]} s, "
                       f"first step {lo:.1f}-{hi:.1f} min after the account step"
                       + (f"; templates: {' | '.join(tpls)}" if tpls else ""))
        out.append(f"  each step adds {'/'.join(str(s) for s in self.step_seconds)} s to the clock (drift +-{DRIFT} s)")
        out.extend("  note: " + n for n in self.notes)
        return out


DEFAULT_CADENCE = Cadence(
    family_base={"P": 41, "F": 52},
    step_seconds=(29, 30),
    gaps={"P": (390, 449), "F": (1110, 1169)},
    first_offset={"F": (80, 170, 18, 20), "P": (12, 65, 6, 8)},
    source="default (measured on the October 2026 dataset)",
)


# ---------------------------------------------------------------------------
def _sec_distance(a: int, b: int) -> int:
    d = abs(a - b) % 60
    return min(d, 60 - d)


def _lattice_seconds(base: int, steps: Tuple[int, ...], max_k: int = 6) -> set:
    secs, frontier = set(), {base}
    for _ in range(max_k + 1):
        for s in frontier:
            for d in range(-DRIFT, DRIFT + 1):
                secs.add((s + d) % 60)
        frontier = {(s + x) % 60 for s in frontier for x in steps}
    return secs


def _mode_merged(counter: Counter) -> Optional[int]:
    """Peak second: the densest 3-second window (absorbs +-1 s drift), then its most frequent second."""
    if not counter:
        return None
    centre = max(range(60), key=lambda b: (sum(counter[(b + d) % 60] for d in (-1, 0, 1)), counter[b], -b))
    return max(((centre + d) % 60 for d in (-1, 0, 1)), key=lambda b: (counter[b], -abs(((b - centre + 30) % 60) - 30)))


def _quantile(values: List[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def learn_cadence(anchors, worker_rows, owners_of, parse_ts, *, minimum: int = MIN_ANCHORS) -> Cadence:
    """Learn the schedule.

    ``anchors``: list of (user_id, account_row, pinned_rows) for requests whose account is pinned by an
    identifier; ``pinned_rows`` are deletion-worker rows of resources pinned to the same person.
    ``worker_rows``: every deletion-worker row (time ordered). ``owners_of(resource_id)``: user ids the
    evidence ties the resource to (empty when unknown).
    """
    if len(anchors) < minimum:
        c = Cadence(**{k: getattr(DEFAULT_CADENCE, k) for k in ("family_base", "step_seconds", "gaps", "first_offset")})
        c.source = f"default: only {len(anchors)} pinned requests (need {minimum})"
        c.anchors = len(anchors)
        return c

    # 1. single-step gaps per template, from consecutive pinned rows of the same template
    tpl_gaps: Dict[str, List[int]] = defaultdict(list)
    tpl_secs: Dict[str, Counter] = defaultdict(Counter)
    for _, head, rows in anchors:
        t0 = parse_ts(head.event_time)
        by_tpl: Dict[str, List[int]] = defaultdict(list)
        for w in rows:
            d = int((parse_ts(w.event_time) - t0).total_seconds())
            by_tpl[template(w.details)].append(d)
            tpl_secs[template(w.details)][d % 60] += 1
        for t, ds in by_tpl.items():
            ds.sort()
            tpl_gaps[t].extend(b - a for a, b in zip(ds, ds[1:]) if b > a)
    singles: Dict[str, List[int]] = {}
    for t, g in tpl_gaps.items():
        if len(g) < 5:
            continue
        cnt = Counter(g)
        dominant = sorted(v for v, n in cnt.items() if n >= max(2, 0.10 * len(g)))
        if dominant:
            singles[t] = dominant

    # 2. families: templates whose single-step gaps overlap (within 10%) share a family
    fams: List[Tuple[int, int, List[str]]] = []
    for t, dom in sorted(singles.items(), key=lambda x: min(x[1])):
        lo, hi = min(dom), max(dom)
        for i, (flo, fhi, members) in enumerate(fams):
            if lo <= fhi * 1.1 and hi >= flo * 0.9:
                fams[i] = (min(lo, flo), max(hi, fhi), members + [t])
                break
        else:
            fams.append((lo, hi, [t]))
    if not fams:
        c = Cadence(**{k: getattr(DEFAULT_CADENCE, k) for k in ("family_base", "step_seconds", "gaps", "first_offset")})
        c.source, c.anchors = "default: no repeated worker steps to learn gaps from", len(anchors)
        return c
    names = {i: f"G{i + 1}" for i in range(len(fams))}
    template_family = {t: names[i] for i, (_, _, members) in enumerate(fams) for t in members}
    gaps = {names[i]: (lo, hi) for i, (lo, hi, _) in enumerate(fams)}
    steps = tuple(sorted({v % 60 for dom in singles.values() for v in dom}))

    # 3. base second per family: earliest row of the family after each pinned account step
    def earliest(fam_of) -> Dict[str, List[Tuple[int, int]]]:
        out: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for uid, head, _ in anchors:
            t0 = parse_ts(head.event_time)
            seen = set()
            for w in worker_rows:
                if w.target_system == "account_service":
                    continue
                d = int((parse_ts(w.event_time) - t0).total_seconds())
                if d <= 120:
                    continue
                if d > 6 * 3600:
                    break
                owners = owners_of(w.resource_id)
                if owners and uid not in owners:
                    continue
                f = fam_of(w)
                if f is None or f in seen:
                    continue
                seen.add(f)
                out[f].append((d, d % 60))
        return out

    gap_templates = dict(template_family)
    # the family with the most worker rows takes templates there is too little data to place
    largest = max(gaps, key=lambda f: (sum(sum(tpl_secs[t].values()) for t in gap_templates if gap_templates[t] == f), f))
    notes: List[str] = []

    def estimate_bases(fam_map) -> Dict[str, int]:
        """The peak second of each family's earliest rows. Seconds alone cannot tell the first step from the
        second one (a step adds 29-30 s, so the pattern repeats half a minute later): of the two candidate
        peaks, the one whose rows come earlier in minutes is the first step."""
        first = earliest(lambda w: fam_map.get(template(w.details)))
        out = {}
        for f in gaps:
            samples = first.get(f, [])
            if len(samples) < MIN_FAMILY_SAMPLES:
                continue
            counts = Counter(s for _, s in samples)
            b = _mode_merged(counts)
            shifted = Counter({k: v for k, v in counts.items() if _sec_distance(k, b) > 3 * DRIFT})
            b2 = _mode_merged(shifted) if shifted else None
            if b2 is not None and any(_sec_distance(b2, (b + x) % 60) <= DRIFT or _sec_distance(b, (b2 + x) % 60) <= DRIFT
                                      for x in steps):
                near = lambda base: sorted(d for d, s in samples if _sec_distance(s, base) <= DRIFT)  # noqa: E731
                m1, m2 = near(b), near(b2)
                if len(m2) >= 0.2 * len(m1) and m2 and m1 and m2[len(m2) // 2] < m1[len(m1) // 2]:
                    b = b2
            out[f] = b
        return out

    def place_templates(fam_map, bases) -> Dict[str, str]:
        """Templates without gap data join the family whose seconds pattern explains their rows best."""
        placed = dict(gap_templates)
        lattices = {f: _lattice_seconds(b, steps) for f, b in bases.items()}
        for t, secs in tpl_secs.items():
            if t in placed:
                continue
            total = sum(secs.values())
            if total < 5:
                placed[t] = largest          # too few rows to tell; the largest family is the safest guess
                continue
            scores = {f: sum(n for s, n in secs.items() if s in lat) / total for f, lat in lattices.items()}
            best = max(scores, key=scores.get) if scores else None
            placed[t] = best if best is not None and scores[best] >= 0.6 else largest
        return placed

    # 3-4. base seconds and template placement, iterated: the first estimate uses only the templates that
    # have gap data; once every template is placed, the earliest step of a family is seen correctly
    base = estimate_bases(template_family)
    for _ in range(3):
        template_family = place_templates(template_family, base)
        new_base = estimate_bases(template_family)
        if new_base == base:
            break
        base = new_base
    if len(base) < len(gaps):
        missing = [f for f in gaps if f not in base]
        notes.append(f"too few first-step samples for {', '.join(missing)}; default base used")
    # 5. first-step delay window per family (on-base earliest rows), now with every template placed
    first = earliest(lambda w: template_family.get(template(w.details), largest))
    first_offset = {}
    for f, (glo, ghi) in gaps.items():
        b = base.get(f)
        mins = [d / 60.0 for d, s in first.get(f, []) if b is not None and _sec_distance(s, b) <= DRIFT]
        step_lo, step_hi = math.floor(glo / 60.0), math.ceil(ghi / 60.0)
        if len(mins) >= MIN_FAMILY_SAMPLES:
            qlo, qhi, widen = WINDOW_RULE
            lo = max(2.0, _quantile(mins, qlo) - widen * ghi / 60.0)
            hi = _quantile(mins, qhi) + widen * ghi / 60.0
            first_offset[f] = (round(lo, 1), round(hi, 1), float(step_lo), float(step_hi))
        else:
            first_offset[f] = (2.0, 6 * 60.0, float(step_lo), float(step_hi))
            notes.append(f"too few first-step samples for {f}; delay window left open")
        if b is None:
            base[f] = 0
    c = Cadence(family_base=base, step_seconds=steps or DEFAULT_CADENCE.step_seconds, gaps=gaps,
                first_offset=first_offset, template_family=template_family, default_family=largest,
                source="learned from the participant activity", anchors=len(anchors), notes=notes)
    return c
