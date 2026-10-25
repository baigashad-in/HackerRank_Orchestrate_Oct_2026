"""CSV loaders for the participant-facing dataset."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List

from . import config


@dataclass
class Case:
    case_id: str
    email: str
    requested_at: str
    reported_status: str
    reported_at: str
    audit_at: str
    request_text: str


@dataclass
class Identity:
    user_id: str
    name: str
    email: str
    phone: str
    email_hash: str
    updated_at: str


@dataclass
class AuditEntry:
    audit_id: str
    user_id: str
    event_time: str
    field_name: str
    old_value: str
    new_value: str


@dataclass
class Activity:
    activity_id: str
    event_time: str
    event_type: str
    source_system: str
    target_system: str
    resource_id: str
    reported_result: str
    details: str


@dataclass
class Rule:
    rule_id: str
    data_category: str
    condition: str
    required_action: str


@dataclass
class EvidenceFile:
    file_id: str
    file_path: str
    origin: str
    captured_at: str

    @property
    def abs_path(self) -> str:
        return os.path.join(config.DATASET_DIR, self.file_path)

    @property
    def ext(self) -> str:
        return os.path.splitext(self.file_path)[1].lower()


@dataclass
class Dataset:
    cases: List[Case] = field(default_factory=list)
    identities: List[Identity] = field(default_factory=list)
    audit_log: List[AuditEntry] = field(default_factory=list)
    activity: List[Activity] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    evidence_files: List[EvidenceFile] = field(default_factory=list)
    dataset_dir: str = ""

    @property
    def evidence_by_id(self) -> Dict[str, EvidenceFile]:
        return {e.file_id: e for e in self.evidence_files}


def _read_csv(path: str) -> List[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_cases(path: str) -> List[Case]:
    return [Case(**{k: (row.get(k) or "").strip() for k in Case.__dataclass_fields__}) for row in _read_csv(path)]


def load_dataset(dataset_dir: str = config.DATASET_DIR, requests_file: str = "deletion_requests.csv") -> Dataset:
    ds = Dataset()
    ds.cases = load_cases(os.path.join(dataset_dir, requests_file))
    ds.dataset_dir = dataset_dir
    ds.identities = [Identity(**{k: (r.get(k) or "").strip() for k in Identity.__dataclass_fields__})
                     for r in _read_csv(os.path.join(dataset_dir, "user_identities.csv"))]
    ds.audit_log = [AuditEntry(**{k: (r.get(k) or "").strip() for k in AuditEntry.__dataclass_fields__})
                    for r in _read_csv(os.path.join(dataset_dir, "user_audit_log.csv"))]
    ds.activity = [Activity(**{k: (r.get(k) or "").strip() for k in Activity.__dataclass_fields__})
                   for r in _read_csv(os.path.join(dataset_dir, "system_activity.csv"))]
    ds.rules = [Rule(**{k: (r.get(k) or "").strip() for k in Rule.__dataclass_fields__})
                for r in _read_csv(os.path.join(dataset_dir, "deletion_rules.csv"))]
    ds.evidence_files = [EvidenceFile(**{k: (r.get(k) or "").strip() for k in EvidenceFile.__dataclass_fields__})
                         for r in _read_csv(os.path.join(dataset_dir, "evidence_files.csv"))]
    return ds
