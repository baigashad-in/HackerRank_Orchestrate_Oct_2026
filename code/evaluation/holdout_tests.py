#!/usr/bin/env python3
"""Hold-out tests for the parts of the engine that infer links no evidence states.

    python code/evaluation/holdout_tests.py            # learned schedule vs measured defaults, 2-fold

The reference answers cover only 32 cases, so these tests create their own ground truth from the
participant data: a request whose account (or worker row) is pinned to the person by an identifier has a
known answer. The test hides those identifier links and checks whether the run-timing logic recovers them.

Two-fold cross-validation: the pinned requests are split in two halves by case number. The worker schedule
is learned on one half only, and the links of the other half are hidden and recovered with it. The measured
default schedule is evaluated on exactly the same hidden halves, so the comparison is like for like and the
learned schedule is never graded on the requests it was learned from.

Tests:
* account attribution: hide every identifier anchor of the held-out accounts, re-run the assignment;
* worker-row attribution: hide a random half (fixed seed) of the held-out requests' pinned worker rows,
  re-run the chain attribution; count rows recovered, missed (left unattributed) and wrongly attributed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ghost_audit import config, loaders  # noqa: E402
from ghost_audit.activity import parse_ts  # noqa: E402
from ghost_audit.cadence import DEFAULT_CADENCE, learn_cadence  # noqa: E402
from ghost_audit.linking import set_cadence  # noqa: E402
from ghost_audit.pipeline import AuditEngine  # noqa: E402
from ghost_audit.request_classifier import classify_request  # noqa: E402


def fold_of(case_id: str) -> int:
    return int(case_id.split("_")[-1]) % 2


def all_requests(ds):
    cases = {c.case_id: c for c in loaders.load_cases(os.path.join(config.DATASET_DIR, "sample_deletion_requests.csv"))}
    for c in ds.cases:
        cases.setdefault(c.case_id, c)
    return list(cases.values())


def build(fresh: bool = True):
    ds = loaders.load_dataset(config.DATASET_DIR)
    ds.cases = all_requests(ds)
    eng = AuditEngine(ds, cadence="default")
    users = {c.case_id: eng.idx.resolve_email(c.email) for c in ds.cases}
    deletion = [c for c in ds.cases if users[c.case_id] is not None and classify_request(c.request_text).authorises_deletion]
    return ds, eng, users, deletion


def cadence_for(kind: str, eng, users, deletion, train_fold: int):
    if kind == "default":
        return DEFAULT_CADENCE
    train = [c for c in deletion if fold_of(c.case_id) == train_fold]
    anchors = eng.linker.anchors(train, users)
    return learn_cadence(anchors, eng.act.worker_rows, lambda rid: eng.linker.resource_users.get(rid, set()), parse_ts)


def account_test(kind: str):
    total = {"cases": 0, "recovered": 0, "wrong": 0, "unassigned": 0}
    for test_fold in (0, 1):
        ds, eng, users, deletion = build()
        linker = eng.linker
        set_cadence(cadence_for(kind, eng, users, deletion, 1 - test_fold))
        truth = {}
        for c in deletion:
            if fold_of(c.case_id) != test_fold:
                continue
            u = users[c.case_id]
            anch = [a for a in linker.account_candidates(c) if u.user_id in linker.resource_users.get(a.resource_id, set())]
            if len(anch) == 1:
                truth[c.case_id] = anch[0].resource_id
        for cid, rid in truth.items():
            u = users[cid].user_id
            linker.resource_users[rid].discard(u)
            linker.user_resources[u].discard(rid)
        res = linker.assign_accounts(deletion, users)
        for cid, rid in truth.items():
            a, _ = res[cid]
            total["cases"] += 1
            if a is None:
                total["unassigned"] += 1
            elif a.resource_id == rid:
                total["recovered"] += 1
            else:
                total["wrong"] += 1
    return total


def chain_test(kind: str, seed: int = 7):
    total = {"hidden_rows": 0, "recovered": 0, "missed": 0, "wrong": 0}
    for test_fold in (0, 1):
        ds, eng, users, deletion = build()
        linker = eng.linker
        set_cadence(cadence_for(kind, eng, users, deletion, 1 - test_fold))
        accounts = linker.assign_accounts(deletion, users)
        truth = {}
        for c in deletion:
            if fold_of(c.case_id) != test_fold:
                continue
            a, m = accounts[c.case_id]
            if a is None or m != "identifier":
                continue
            rows = [w for w in linker.user_chain_events(users[c.case_id], a) if w.source_system == "deletion_worker"]
            if rows:
                truth[c.case_id] = rows
        rnd = random.Random(seed)
        hidden = {}
        for cid in sorted(truth, key=config.case_sort_key):
            for w in truth[cid]:
                if rnd.random() < 0.5:
                    u = users[cid].user_id
                    linker.resource_users[w.resource_id].discard(u)
                    linker.user_resources[u].discard(w.resource_id)
                    hidden.setdefault(cid, []).append(w)
        chains = eng._assign_chains(deletion, users, accounts)
        owner = {}
        for cid, rows in chains.items():
            for w in rows:
                owner.setdefault(w.resource_id, cid)
        for cid, rows in hidden.items():
            got = {w.resource_id for w in chains.get(cid, [])}
            for w in rows:
                total["hidden_rows"] += 1
                if w.resource_id in got:
                    total["recovered"] += 1
                elif w.resource_id in owner:
                    total["wrong"] += 1
                else:
                    total["missed"] += 1
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None, help="also write the numbers to this JSON file")
    args = ap.parse_args(argv)
    out = {}
    for kind in ("learned", "default"):
        acc = account_test(kind)
        ch = chain_test(kind)
        out[kind] = {"accounts": acc, "worker_rows": ch}
        print(f"[{kind:7}] accounts: {acc['recovered']}/{acc['cases']} recovered "
              f"({acc['recovered'] / max(1, acc['cases']):.1%}), {acc['wrong']} wrong, {acc['unassigned']} unassigned | "
              f"worker rows: {ch['recovered']}/{ch['hidden_rows']} recovered, {ch['missed']} missed, {ch['wrong']} wrong")
    set_cadence(DEFAULT_CADENCE)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
