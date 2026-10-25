"""Decide whether a request text authorises deletion.

Deterministic, multilingual rules first (English, Spanish, Portuguese, French,
German, Italian, Dutch, Japanese, Korean, Chinese).  An optional model call can
override ambiguous texts when the model layer is enabled (see ``llm.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Phrases that explicitly say the person does NOT want deletion.
NEGATION_PATTERNS = [
    r"\bnot\s+(?:a\s+)?(?:request(?:ing)?|asking)\s+(?:for\s+|to\s+)?(?:deletion|delete|erase|erasure|closure)",
    r"\bnot\s+a\s+request\s+to\s+(?:erase|delete|remove|close)",
    r"\b(?:do\s+not|don't|dont|please\s+do\s+not)\s+(?:delete|close|erase|remove|change)\b",
    r"\b(?:leave|keep)\s+(?:my|the|this)\s+(?:account|profile|records?|purchase\s+history|data)\b",
    r"\bkeep\s+the\s+account\s+as\s+it\s+is\b",
    r"\bstill\s+(?:need|use|want)\s+(?:the|this|my)\s+account\b",
    r"\bcontinue\s+using\b",
    r"\bwithout\s+(?:removing|deleting|closing)\b",
    r"\bno\s+(?:deseo|quiero)\s+(?:eliminar|borrar)",
    r"\bn[ãa]o\s+(?:quero|desejo)\s+(?:excluir|apagar)",
    r"\bne\s+(?:souhaite|veux)\s+pas\s+(?:supprimer|effacer)",
    r"削除(?:は|を)?(?:希望|求め)(?:しません|ていません)",
    r"삭제(?:를)?\s*원하지\s*않",
]

# Requests for something other than deletion (access, export, correction, opt-out, login help).
NON_DELETION_PATTERNS = [
    r"\b(?:copy|download|export)\s+(?:of\s+)?(?:my|the)\s+(?:account\s+)?(?:information|data|invoices?|records?)\b",
    r"\bportable\s+format\b",
    r"\bwhat\s+(?:personal\s+)?(?:information|data)\s+(?:do\s+)?you\s+hold\b",
    r"\binformation\s+request\b",
    r"\bsubject\s+access\b",
    r"\b(?:correct|fix|update|replace|change)\s+(?:it|the|my|this)?\s*(?:surname|name|number|email|address|phone|mobile)",
    r"\bmisspell",
    r"\bout\s+of\s+date\b",
    r"\breset(?:ting)?\s+(?:the\s+|my\s+)?(?:login|password)\b",
    r"\bcan'?t\s+access\b",
    r"\bunsubscribe\b",
    r"\bstop\s+(?:the\s+)?(?:promotional|marketing)\s+(?:emails?|messages?)\b",
    r"\bremove\s+me\s+from\s+(?:the\s+)?marketing\s+lists?\b",
    r"\bmarketing\s+lists?\s+only\b",
    r"\bopt[-\s]?out\b",
]

# Phrases that authorise deletion / erasure / account closure.
DELETION_PATTERNS = [
    # English
    r"\b(?:delete|deleting|erase|erasing|remove|removing|close|closing|wipe)\s+(?:my|this|the)\s+(?:entire\s+)?(?:account|profile|personal\s+data|personal\s+information|data|records?)\b",
    r"\b(?:account|profile|data)\s+(?:be\s+)?(?:deleted|erased|removed|closed|wiped)\b",
    r"\b(?:deletion|erasure|removal)\s+(?:of\s+)?(?:my|this|the)?\s*(?:account|profile|data|personal)",
    r"\baccount[-\s]deletion\b",
    r"\bdeletion\s+request\b",
    r"\brequest\s+(?:for\s+)?(?:permanent\s+)?(?:account\s+)?(?:deletion|erasure)\b",
    r"\bpermanently\s+(?:delete|deleted|erase|erased|remove|removed|close|closed)\b",
    r"\b(?:delete|erase|remove|close)\s+(?:the\s+)?account\b",
    r"\bbegin\s+the\s+deletion\s+process\b",
    r"\bi\s+(?:have\s+)?deleted\s+(?:my|this)\s+account\b",
    r"\bremove\s+(?:any\s+)?remaining\s+(?:copy|copies|records?|data)\b",
    r"\bfinish\s+the\s+deletion\b",
    r"\bverify\s+that\s+my\s+account\s+was\s+deleted\b",
    r"\bremove\s+(?:my|the)\s+name\s+and\s+contact\s+details\b",
    r"\bpersonal[-\s]data\s+deletion\b",
    r"\bclosed\b.*\baccount\b|\baccount\b.*\bclosed\b",
    r"\b(?:account|profile|data|information|records?)\b.*\bto\s+be\s+(?:erased|deleted|removed|wiped)\b",
    r"\b(?:deleted|deletion|erased|erasure)\b",
    r"\bremove\b.*\b(?:records?|search\s+result|cop(?:y|ies)|traces?|profile|details|rest|remaining|data|information|export|file)\b",
    r"\bleft\s+behind\b",
    r"\bstill\s+(?:appears|exists?|recogni[sz]ed|works|points|shows|can\s+(?:still\s+)?be\s+found)\b",
    r"\b(?:is|are|was|were)\s+gone\b",
    r"\bcannot\s+identify\s+me\b|\bcan\s+no\s+longer\s+(?:identify|be\s+linked)\b",
    r"\bno\s+longer\s+(?:send|copy|recreate|restore)\b",
    r"\bmust\s+not\s+be\s+able\s+to\s+restore\b",
    r"\bbeen\s+(?:removed|deleted|erased|anonymi[sz]ed|redacted)\b",
    r"\bprivacy\s+(?:page|request)\b.*\b(?:failed|removed|deleted)\b",
    # Spanish
    r"\b(?:eliminar|eliminación|borrar|borrado|suprimir|supresión)\b.*\b(?:cuenta|datos|perfil)\b",
    r"\b(?:cuenta|datos|perfil)\b.*\b(?:eliminad[ao]|borrad[ao]|eliminación|eliminar)\b",
    # Portuguese
    r"\b(?:excluir|exclusão|apagar|eliminar|eliminação|remover|remoção)\b.*\b(?:conta|dados|perfil)\b",
    r"\b(?:conta|dados|perfil)\b.*\b(?:exclu[íi]d[ao]|apagad[ao]|exclusão)\b",
    # French
    r"\b(?:supprimer|suppression|effacer|effacement)\b.*\b(?:compte|donn[ée]es|profil)\b",
    r"\b(?:compte|donn[ée]es|profil)\b.*\b(?:supprim[ée]e?s?|effac[ée]e?s?|suppression)\b",
    # German
    r"\b(?:l[öo]schen|l[öo]schung|entfernen)\b.*\b(?:konto|daten|profil)\b",
    r"\b(?:konto|daten|profil)\b.*\b(?:l[öo]schen|gel[öo]scht|l[öo]schung)\b",
    # Italian
    r"\b(?:cancellare|cancellazione|eliminare|eliminazione|rimuovere)\b.*\b(?:account|dati|profilo)\b",
    # Dutch
    r"\b(?:verwijder|verwijderen|verwijdering|wissen)\b.*\b(?:account|gegevens|profiel)\b",
    # Japanese
    r"(?:アカウント|個人情報|データ|プロフィール).*(?:削除|消去|抹消)",
    r"(?:削除|消去|抹消).*(?:アカウント|個人情報|データ)",
    # Korean
    r"(?:계정|개인\s*정보|데이터|프로필).*(?:삭제|제거|지워)",
    r"(?:삭제|제거).*(?:계정|개인\s*정보|데이터)",
    # Chinese
    r"(?:删除|註銷|注销|清除).*(?:账户|帳戶|帐户|账号|資料|资料|数据)",
    r"(?:账户|帳戶|帐户|账号|資料|资料|数据).*(?:删除|註銷|注销|清除)",
]

_NEG = [re.compile(p, re.I | re.S) for p in NEGATION_PATTERNS]
_NON = [re.compile(p, re.I | re.S) for p in NON_DELETION_PATTERNS]
_DEL = [re.compile(p, re.I | re.S) for p in DELETION_PATTERNS]


@dataclass
class Classification:
    authorises_deletion: bool
    confidence: float
    reason: str
    method: str = "rules"


def classify_request(text: str) -> Classification:
    t = " ".join((text or "").split())
    neg = [p.pattern for p in _NEG if p.search(t)]
    non = [p.pattern for p in _NON if p.search(t)]
    dele = [p.pattern for p in _DEL if p.search(t)]
    if neg:
        return Classification(False, 0.98, f"explicit non-deletion statement: {neg[0]}")
    if dele and not non:
        return Classification(True, 0.95, f"deletion phrase: {dele[0]}")
    if dele and non:
        # both kinds of signal: deletion wins only when it is the primary verb of the request
        if re.search(r"^\s*(?:please\s+)?(?:permanently\s+)?(?:delete|erase|remove|close)\b", t, re.I) or \
                re.search(r"\b(?:deletion|erasure)\b", t, re.I):
            return Classification(True, 0.7, f"deletion primary despite other asks: {dele[0]}")
        return Classification(False, 0.6, f"non-deletion ask dominates: {non[0]}")
    if non:
        return Classification(False, 0.9, f"non-deletion request: {non[0]}")
    return Classification(False, 0.5, "no deletion phrase found")
