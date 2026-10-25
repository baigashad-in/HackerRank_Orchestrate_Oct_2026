"""Which identifier fields still expose the person, per record.

``output.csv`` can only name the record (the contract has no field for it), so
this module answers the finer question for ``--explain`` and for the
``--exposures`` report: which of the subject's identifiers the record still
carries, and whether each one is the current value or a historical value from
``user_audit_log.csv``. Only field names are reported, never the values.
"""

from __future__ import annotations

from typing import Iterable, List

from .evidence.base import Identifiers, Observation, norm_email, norm_name, norm_phone
from .identity import UserProfile, sha256_email

FIELD_ORDER = ("name", "email", "phone", "email_hash", "user_id")
UNSPECIFIED = "direct identifier (type not stated in the evidence)"
CONTENTS_NOT_SHOWN = "record linked to the person, but its identifier fields are not shown in the evidence"


def describe_identifiers(user: UserProfile, ids: Identifiers, *, include_user_ref: bool = False) -> List[str]:
    """Field names (with current/historical) of the subject's identifiers present in ``ids``."""
    found = set()
    cur_email = norm_email(user.current_email or "")
    cur_phone = norm_phone(user.current_phone or "")
    cur_name = norm_name(user.name or "")
    cur_hashes = {(user.current_hash or "").lower()}
    if user.current_email:
        cur_hashes.add(sha256_email(user.current_email))
    for e in ids.emails:
        if e in user.emails:
            found.add(("email", "current" if e == cur_email else "historical"))
    for p in ids.phones:
        if p in user.phones:
            found.add(("phone", "current" if p == cur_phone else "historical"))
    for n in ids.names:
        if n in user.names:
            found.add(("name", "current" if n == cur_name else "historical"))
    for h in ids.hashes:
        if h in user.hashes:
            found.add(("email_hash", "current" if h in cur_hashes else "historical"))
    if include_user_ref and user.user_id in ids.user_refs:
        found.add(("user_id", "stable reference"))
    return [f"{f} ({q})" for f, q in sorted(found, key=lambda x: (FIELD_ORDER.index(x[0]), x[1]))]


def decisive_observations(observations: Iterable[Observation], *, want_identifiable: bool) -> List[Observation]:
    """Observations at the latest time (the ones the decision rests on)."""
    obs = list(observations)
    if not obs:
        return []
    latest = max(o.observed_at for o in obs)
    at_latest = [o for o in obs if o.observed_at == latest]
    if want_identifiable:
        exposed = [o for o in at_latest if o.identifiable is True] or [o for o in obs if o.identifiable is True]
        return exposed or at_latest
    present = [o for o in at_latest if o.present is True] or at_latest
    return present


def _unlisted(idx, ids: Identifiers, user: UserProfile) -> List[str]:
    """Direct identifiers on the record that belong to nobody in the identity records (not someone else's)."""
    if idx is None:
        return []
    out = set()
    for field_name, values in (("email", ids.emails), ("phone", ids.phones), ("email_hash", ids.hashes)):
        for v in values:
            probe = Identifiers()
            getattr(probe, {"email": "emails", "phone": "phones", "email_hash": "hashes"}[field_name]).add(v)
            owners = idx.users_for(probe)
            if not owners:
                out.add(field_name)
    return [f"{f} (value not in the identity records)" for f in FIELD_ORDER if f in out]


def exposed_fields(user: UserProfile, observations: Iterable[Observation], *, pii_finding: bool,
                   idx=None) -> List[str]:
    """Identifier fields the record still carries.

    For an exposed-PII finding (a record that must be redacted or anonymised) the stable user_id is allowed
    and is not listed. For a remaining record (which must be deleted), every identifier that ties it to the
    person is listed, including a stable reference. Identifiers that belong to another known person are never
    listed. When the evidence shows only the key the record was found by (a search or lookup), that key is
    listed and marked as such; when it shows neither, the answer says so instead of guessing.
    """
    obs = decisive_observations(observations, want_identifiable=pii_finding)
    carried = Identifiers()
    for o in obs:
        carried.merge(o.identifiers)
    fields = describe_identifiers(user, carried, include_user_ref=not pii_finding)
    fields += _unlisted(idx, carried, user)
    if fields:
        return fields
    keys = lookup_fields(user, obs)
    if keys:
        return [f"{k} [lookup key: the record is returned by a search on it]" for k in keys]
    if pii_finding and any(o.identifiable for o in obs):
        return [UNSPECIFIED]
    return [CONTENTS_NOT_SHOWN]


def lookup_fields(user: UserProfile, observations: Iterable[Observation]) -> List[str]:
    """Identifiers the evidence used to find the record (for example a search by an old e-mail)."""
    keys = Identifiers()
    for o in observations:
        keys.merge(o.lookup)
    return describe_identifiers(user, keys, include_user_ref=True)
