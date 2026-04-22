"""Gemini 2.5 Flash extraction agent.

The agent is called once per page. It receives the page text and regex candidate
hints, and returns a JSON object with a `mentions` array. Every mention must
include `evidence_text` copied verbatim from the page — this is the anchor that
stage 6 uses to verify the extraction.

The `google.generativeai` SDK is imported lazily so this module loads even when
the package is not installed (useful for the unit suite).
"""
import json
import re
import time
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings('ignore', category=FutureWarning)

from .candidates import Candidate


SYSTEM_PROMPT = """You extract references to external documents from SEBI (Securities and Exchange Board of India) regulatory PDFs.

You are given the full text of ONE page. Your job: find every reference this page makes to another document — circulars, master circulars, regulations, acts, gazette notifications, consultation papers, etc.

PAY SPECIAL ATTENTION TO:
- Authority clauses at the end of circulars ("This circular is issued in exercise of powers conferred under Section X of the ... Act, YYYY, read with Regulation Y of ... Regulations, YYYY"). These reference Acts AND Regulations — extract EACH one separately.
- Multiple Acts/Regulations in a single sentence — extract each as a separate mention.
- Include the year in Act/Regulation titles when it appears (e.g. "SEBI (Mutual Funds) Regulations, 1996"). The year IS part of the canonical name for Acts and Regulations.
- **"Securities and Exchange Board of India Act, 1992" IS an external reference** — it is a distinct statutory instrument even though SEBI (the agency) is named in the Act title. ALWAYS extract it when it appears, especially in authority clauses like "under Section 11(1) of the Securities and Exchange Board of India Act, 1992".
- Do NOT append a date/year to the `title` for CIRCULAR or MASTER_CIRCULAR types — put the date in the `date` field instead. Example: title="Master Circular for Mutual Funds", date="2026-03-20". (This is OPPOSITE of Acts/Regulations where the year stays in the title.)
- Notifications: if a notification is cited with a date but no title ("vide notification dated March 21, 2026"), set doc_type="gazette_notification", title=null, date="2026-03-21". Do NOT skip it.

HARD RULES (non-negotiable):
1. Every extracted reference MUST include `evidence_text` — a short verbatim snippet (<=250 chars) copied character-for-character from the page text. Never paraphrase or reformat.
2. If you cannot identify a title with confidence, set `title` to null. NEVER guess a title.
3. Exclude self-references like "this circular", "the present circular", "this notification", "this Master Circular".
4. `doc_type` must be one of: circular, master_circular, regulation, act, consultation_paper, gazette_notification, other.
5. `relation_type` must be one of: references, modifies, supersedes, issued_under.
6. `confidence` must be one of: high, medium, low.
7. `date` must be ISO format YYYY-MM-DD, or null if unknown / partial / ambiguous.
8. Return ONLY valid JSON. No prose, no markdown fences, no explanation.

Output schema (JSON object):
{
  "mentions": [
    {
      "mention_text": "the surface phrase naming the referenced doc",
      "evidence_text": "verbatim snippet from the page containing the mention",
      "doc_type": "circular",
      "doc_id": "SEBI/HO/... or null",
      "date": "2020-05-13 or null",
      "title": "Full canonical title or null",
      "relation_type": "references",
      "confidence": "high"
    }
  ]
}
"""


def build_page_prompt(page_text: str, candidates: List[Candidate], page_index: int) -> str:
    hint_lines = [f"- [{c.pattern_name}] {c.text}" for c in candidates[:40]]
    hints = '\n'.join(hint_lines) if hint_lines else "(none)"
    return f"""PAGE {page_index} TEXT:
<<<
{page_text}
>>>

REGEX CANDIDATE HINTS (not exhaustive, may contain noise — use your judgment):
{hints}

Return JSON only.
"""


_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)


def strip_fences(s: str) -> str:
    s = (s or '').strip()
    if s.startswith('```'):
        s = _FENCE_RE.sub('', s).strip()
    return s


def parse_response(raw: str) -> List[Dict]:
    """Parse a Gemini response into a list of mention dicts.

    Tolerates: markdown code fences, leading/trailing prose, missing `mentions`
    key. Returns [] on any parse failure rather than raising.
    """
    if not raw:
        return []
    cleaned = strip_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(obj, dict):
        return []
    mentions = obj.get('mentions')
    return mentions if isinstance(mentions, list) else []


def dedupe_mentions(mentions: List[Dict]) -> List[Dict]:
    """Deduplicate mentions across pages by (source_page, doc_id, title, mention_text)."""
    seen = set()
    out = []
    for m in mentions:
        key = (
            m.get('source_page'),
            (m.get('doc_id') or '').strip().lower(),
            (m.get('title') or '').strip().lower(),
            (m.get('mention_text') or '').strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


class GeminiAgent:
    """Thin wrapper around google-generativeai's Gemini 2.5 Flash model.

    If no API key is supplied (or the SDK isn't installed), `extract_page` is a
    no-op that returns []. This lets the full pipeline run end-to-end in dry-run
    mode for pipeline testing.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash", delay: float = 1.0):
        self.model_name = model
        self.delay = delay
        self._client = None
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(
                    model_name=model,
                    system_instruction=SYSTEM_PROMPT,
                )
            except ImportError:
                self._client = None

    def extract_page(self, page_text: str, candidates: List[Candidate], page_index: int,
                      max_retries: int = 3) -> List[Dict]:
        if self._client is None:
            return []
        prompt = build_page_prompt(page_text, candidates, page_index)
        raw = ''
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.0,
                    },
                )
                raw = resp.text if hasattr(resp, 'text') else ''
                break
            except Exception as e:
                err_str = str(e)
                if '429' in err_str and attempt < max_retries:
                    wait = self.delay * (2 ** (attempt + 1))
                    print(f"[agent] page {page_index} rate-limited, retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    continue
                print(f"[agent] page {page_index} error: {e}")
                break
        if self.delay:
            time.sleep(self.delay)
        return parse_response(raw)
