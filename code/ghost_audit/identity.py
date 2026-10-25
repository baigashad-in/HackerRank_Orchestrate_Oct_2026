"""Identity resolution: current and historical identifiers for every user."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .evidence.base import Identifiers, norm_email, norm_name, norm_phone
from .loaders import AuditEntry, Identity


def sha256_email(email: str) -> str:
    return hashlib.sha256(norm_email(email).encode("utf-8")).hexdigest()


@dataclass
class UserProfile:
    user_id: str
    name: str
    current_email: str
    current_phone: str
    current_hash: str
    emails: Set[str] = field(default_factory=set)         # current + historical, lower-case
    phones: Set[str] = field(default_factory=set)         # digits only
    hashes: Set[str] = field(default_factory=set)         # current + historical + sha256 of every email
    names: Set[str] = field(default_factory=set)
    history: List[AuditEntry] = field(default_factory=list)

    def identifiers(self) -> Identifiers:
        ids = Identifiers()
        ids.emails |= self.emails
        ids.phones |= self.phones
        ids.hashes |= self.hashes
        ids.names |= self.names
        ids.user_refs.add(self.user_id)
        return ids

    def historical_emails(self) -> Set[str]:
        return {e for e in self.emails if e != norm_email(self.current_email)}


class IdentityIndex:
    """Fast lookup from any identifier to the user(s) carrying it."""

    def __init__(self, identities: List[Identity], audit_log: List[AuditEntry]) -> None:
        self.users: Dict[str, UserProfile] = {}
        for ident in identities:
            prof = UserProfile(ident.user_id, ident.name, ident.email, ident.phone, ident.email_hash)
            prof.emails.add(norm_email(ident.email))
            prof.phones.add(norm_phone(ident.phone))
            prof.hashes.add(ident.email_hash.lower())
            prof.hashes.add(sha256_email(ident.email))
            prof.names.add(norm_name(ident.name))
            self.users[ident.user_id] = prof
        for entry in audit_log:
            prof = self.users.get(entry.user_id)
            if prof is None:
                continue
            prof.history.append(entry)
            for value in (entry.old_value, entry.new_value):
                if not value:
                    continue
                if entry.field_name == "email":
                    prof.emails.add(norm_email(value))
                    prof.hashes.add(sha256_email(value))
                elif entry.field_name == "email_hash":
                    prof.hashes.add(value.lower())
                elif entry.field_name == "phone":
                    prof.phones.add(norm_phone(value))
                elif entry.field_name == "name":
                    prof.names.add(norm_name(value))
        self.by_email: Dict[str, Set[str]] = {}
        self.by_phone: Dict[str, Set[str]] = {}
        self.by_hash: Dict[str, Set[str]] = {}
        self.by_name: Dict[str, Set[str]] = {}
        for uid, prof in self.users.items():
            for e in prof.emails:
                self.by_email.setdefault(e, set()).add(uid)
            for p in prof.phones:
                self.by_phone.setdefault(p, set()).add(uid)
            for h in prof.hashes:
                self.by_hash.setdefault(h, set()).add(uid)
            for n in prof.names:
                self.by_name.setdefault(n, set()).add(uid)

    def resolve_email(self, email: str) -> Optional[UserProfile]:
        uids = self.by_email.get(norm_email(email), set())
        if len(uids) == 1:
            return self.users[next(iter(uids))]
        if not uids:
            return None
        # prefer the user whose *current* email it is
        for uid in uids:
            if norm_email(self.users[uid].current_email) == norm_email(email):
                return self.users[uid]
        return self.users[sorted(uids)[0]]

    def users_for(self, ids: Identifiers, *, use_names: bool = False) -> Set[str]:
        """All users that share at least one strong identifier with ``ids``."""
        found: Set[str] = set()
        for e in ids.emails:
            found |= self.by_email.get(e, set())
        for p in ids.phones:
            found |= self.by_phone.get(p, set())
        for h in ids.hashes:
            found |= self.by_hash.get(h, set())
        for r in ids.user_refs:
            if r in self.users:
                found.add(r)
        if use_names:
            for n in ids.names:
                found |= self.by_name.get(n, set())
        return found

    def all_emails(self) -> Set[str]:
        return set(self.by_email)
