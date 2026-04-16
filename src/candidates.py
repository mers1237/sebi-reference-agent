"""Regex candidate generation for SEBI reference extraction.

These patterns seed the LLM with likely candidate spans, improving recall on long
PDFs and anchoring the model's attention. They are NOT used as final extractions —
the LLM is the authority on what counts as a genuine external reference.
"""
import re
from dataclasses import dataclass
from typing import List, Set, Tuple


# --- Patterns ---------------------------------------------------------------

# SEBI circular ID: SEBI/HO/CFD/DIL1/CIR/P/2020/37 or SEBI/CIR/IMD/DF/7/2011
SEBI_CIRCULAR_ID = re.compile(
    r'\bSEBI[/\-][A-Z0-9]+(?:[/\-][A-Z0-9]+){2,}',
    re.IGNORECASE,
)

# "Circular dated <date>" or "Circular No. X dated <date>"
CIRCULAR_DATED = re.compile(
    r'\b[Cc]ircular(?:\s+[Nn]o\.?\s*[A-Z0-9/\-]+)?\s+dated\s+'
    r'(?:\d{1,2}[\s\-/][A-Za-z]+[\s\-/]\d{2,4}|\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|[A-Z][a-z]+\s+\d{1,2},?\s*\d{4})'
)

# Master Circular on X / Master Circular dated X
MASTER_CIRCULAR = re.compile(
    r'\bMaster\s+Circular(?:\s+(?:on|for|dated)\s+[^.;\n]{1,150})?',
    re.IGNORECASE,
)

# SEBI (X) Regulations, YYYY
SEBI_REGULATIONS = re.compile(
    r'\bSEBI\s*\([^)]{3,150}\)\s*Regulations,\s*\d{4}'
)

# Acts: "Securities and Exchange Board of India Act, 1992", "Companies Act, 2013", etc.
ACT_REFERENCE = re.compile(
    r'\b(?:[A-Z][A-Za-z&,\s]{2,70}?)Act,\s*\d{4}'
)

# Gazette Notification
GAZETTE_NOTIFICATION = re.compile(
    r'\b(?:Gazette\s+[Nn]otification|notified\s+in\s+the\s+Gazette)[^.;\n]{0,150}'
)

# Consultation Paper
CONSULTATION_PAPER = re.compile(
    r'\bConsultation\s+Paper[^.;\n]{0,150}',
    re.IGNORECASE,
)

# "vide circular / vide notification / vide letter ..."
VIDE_REFERENCE = re.compile(
    r'\bvide\s+(?:circular|notification|letter)\s+(?:no\.?\s*)?[A-Z0-9/\-]{4,}',
    re.IGNORECASE,
)

# "section X of the Y Act, YYYY"
SECTION_OF_ACT = re.compile(
    r'\b[Ss]ection\s+\d+[A-Z]?(?:\s*\(\d+\))?\s+of\s+the\s+[A-Z][A-Za-z\s&()]+?Act,\s*\d{4}'
)

# "Regulation X of the SEBI (...) Regulations, YYYY"
REGULATION_OF = re.compile(
    r'\b[Rr]egulation\s+\d+[A-Z]?(?:\s*\(\d+\))?\s+of\s+(?:the\s+)?SEBI\s*\([^)]{3,150}\)\s*Regulations,\s*\d{4}'
)

# Circular No. X/Y/Z
CIRCULAR_NO = re.compile(
    r'\b[Cc]ircular\s+[Nn]o\.?\s*[:\-]?\s*[A-Z0-9/\-]{6,}'
)

# Notification No. X/Y/Z
NOTIFICATION_NO = re.compile(
    r'\b[Nn]otification\s+[Nn]o\.?\s*[:\-]?\s*[A-Z0-9/\-]{6,}'
)

PATTERNS = {
    'sebi_circular_id': SEBI_CIRCULAR_ID,
    'circular_dated': CIRCULAR_DATED,
    'master_circular': MASTER_CIRCULAR,
    'sebi_regulations': SEBI_REGULATIONS,
    'act_reference': ACT_REFERENCE,
    'gazette_notification': GAZETTE_NOTIFICATION,
    'consultation_paper': CONSULTATION_PAPER,
    'vide_reference': VIDE_REFERENCE,
    'section_of_act': SECTION_OF_ACT,
    'regulation_of': REGULATION_OF,
    'circular_no': CIRCULAR_NO,
    'notification_no': NOTIFICATION_NO,
}


# --- Self-reference filter --------------------------------------------------

SELF_REFERENCE_PATTERNS = [
    re.compile(r'\bthis\s+circular\b', re.IGNORECASE),
    re.compile(r'\bthe\s+present\s+circular\b', re.IGNORECASE),
    re.compile(r'\bthis\s+master\s+circular\b', re.IGNORECASE),
    re.compile(r'\bthis\s+notification\b', re.IGNORECASE),
    re.compile(r'\bthe\s+present\s+notification\b', re.IGNORECASE),
    re.compile(r'\bthis\s+consultation\s+paper\b', re.IGNORECASE),
]


def is_self_reference(text: str) -> bool:
    return any(p.search(text or '') for p in SELF_REFERENCE_PATTERNS)


# --- Candidate dataclass ----------------------------------------------------

@dataclass
class Candidate:
    pattern_name: str
    text: str
    start: int
    end: int


def find_candidates(page_text: str) -> List[Candidate]:
    """Run all patterns against page text and return deduplicated candidate matches."""
    candidates: List[Candidate] = []
    seen: Set[Tuple[str, str]] = set()
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(page_text):
            text = match.group(0).strip()
            key = (name, text.lower())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(Candidate(
                pattern_name=name,
                text=text,
                start=match.start(),
                end=match.end(),
            ))
    return candidates


def filter_self_references(candidates: List[Candidate]) -> List[Candidate]:
    return [c for c in candidates if not is_self_reference(c.text)]
