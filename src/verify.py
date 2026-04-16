"""Stage 6: verification, date validation, self-reference filtering, within-page dedup."""
import re
from typing import Dict, List, Set, Tuple

from .candidates import is_self_reference


_ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def is_valid_date(date_str) -> bool:
    """True for ISO YYYY-MM-DD strings in plausible range, or None (missing is OK)."""
    if date_str is None:
        return True
    if not isinstance(date_str, str):
        return False
    if not _ISO_DATE_RE.match(date_str):
        return False
    try:
        y, m, d = (int(x) for x in date_str.split('-'))
    except ValueError:
        return False
    return 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31


def _normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '')).strip().lower()


_MONTH_NAMES = (
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
)
_NUMERIC_DATE_RE = re.compile(r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}')


def _date_grounded(date_str: str, evidence: str) -> bool:
    """True if the evidence contains a real date expression, not just a bare year.

    Catches the common hallucination where the LLM sees 'Regulations, 2018'
    and fabricates '2018-01-01'.
    """
    if not date_str or not evidence:
        return False
    year = date_str[:4]
    if year not in evidence:
        return False
    ev_lower = evidence.lower()
    if any(m in ev_lower for m in _MONTH_NAMES):
        return True
    if _NUMERIC_DATE_RE.search(evidence):
        return True
    return False


def evidence_present(evidence: str, page_text: str) -> bool:
    """Whitespace-tolerant containment check for verbatim evidence."""
    if not evidence:
        return False
    return _normalize(evidence) in _normalize(page_text)


def dedupe_within_page(mentions: List[Dict]) -> List[Dict]:
    """Deduplicate mentions sharing the same page and canonical key."""
    seen: Set[Tuple] = set()
    out: List[Dict] = []
    for m in mentions:
        key = (
            m.get('source_page'),
            (m.get('doc_id') or '').lower(),
            _normalize(m.get('title') or ''),
            _normalize(m.get('mention_text') or ''),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def verify_mentions(mentions: List[Dict], pages_text: List[str], strict_evidence: bool = True) -> List[Dict]:
    """Apply all stage-6 validation rules to a mention list.

    - Drop mentions without evidence_text.
    - Clear bad dates to None rather than dropping the whole mention.
    - Drop self-references.
    - Drop any mention whose evidence_text is not present on its claimed page
      (when `strict_evidence=True` and the page is in range).
    - Dedupe within page.
    """
    out: List[Dict] = []
    for m in mentions:
        if not m.get('evidence_text'):
            continue
        if not is_valid_date(m.get('date')):
            m = {**m, 'date': None}
        if m.get('date') and not _date_grounded(m['date'], m.get('evidence_text', '')):
            m = {**m, 'date': None}
        if is_self_reference(m.get('mention_text') or '') or is_self_reference(m.get('evidence_text') or ''):
            continue
        sp = m.get('source_page')
        if strict_evidence and isinstance(sp, int) and 0 <= sp < len(pages_text):
            if not evidence_present(m['evidence_text'], pages_text[sp]):
                continue
        out.append(m)
    return dedupe_within_page(out)
