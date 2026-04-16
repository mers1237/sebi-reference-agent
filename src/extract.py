"""PyMuPDF per-page text extraction with zone detection.

Zones: body, footnote, table, annexure. Detection is heuristic (whitespace /
horizontal separators / keyword-based). `fitz` is imported lazily so the module
can be loaded in environments where PyMuPDF is not installed (e.g. unit tests).
"""
import re
from dataclasses import dataclass
from typing import List, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


_ANNEX_RE = re.compile(r'\b(?:ANNEX(?:URE)?|APPENDIX|SCHEDULE)\b', re.IGNORECASE)
_FOOTNOTE_LINE_RE = re.compile(r'^\s*(?:\d{1,3}|\*|†|‡)\s+\S')


@dataclass
class Page:
    index: int          # 0-based
    display_page: int   # 1-based
    text: str
    body_text: str
    footnote_text: str
    table_text: str
    is_annexure: bool


def _split_footnotes(text: str) -> Tuple[str, str]:
    """Walk up from the bottom; lines that look like '<num> <word>...' form the
    footnote block. Stop as soon as we hit a non-matching line."""
    lines = text.split('\n')
    if not lines:
        return text, ''
    cut = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        if not ln.strip():
            continue
        if _FOOTNOTE_LINE_RE.match(ln):
            cut = i
        else:
            break
    if cut >= len(lines):
        return text, ''
    body = '\n'.join(lines[:cut]).rstrip()
    foot = '\n'.join(lines[cut:]).strip()
    return body, foot


def _detect_tables(text: str) -> str:
    """Return lines that look table-like (tab/pipe separators or 3+ columns of whitespace)."""
    table_lines = []
    for ln in text.split('\n'):
        if '\t' in ln or '|' in ln:
            table_lines.append(ln)
            continue
        if len(re.findall(r'\s{2,}', ln)) >= 2 and len(ln.split()) >= 4:
            table_lines.append(ln)
    return '\n'.join(table_lines)


def extract_pages(pdf_path: str) -> List[Page]:
    """Extract every page of the PDF with zone-annotated text."""
    if fitz is None:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF extraction. "
            "Install with: pip install PyMuPDF"
        )
    doc = fitz.open(pdf_path)
    pages: List[Page] = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ''
            body, foot = _split_footnotes(text)
            tables = _detect_tables(text)
            is_annex = bool(_ANNEX_RE.search(text[:500]))
            pages.append(Page(
                index=i,
                display_page=i + 1,
                text=text,
                body_text=body,
                footnote_text=foot,
                table_text=tables,
                is_annexure=is_annex,
            ))
    finally:
        doc.close()
    return pages
