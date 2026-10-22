"""Keyword retrieval over the PaperPilot store.

What each FTS5 piece operates on:
- MATCH   : the segmented index text in pages_fts (title / body_seg columns)
- BM25    : the same segmented columns, with explicit column weights
  (title weight 8.0, body weight 1.0; SQLite bm25 is a distance: smaller
  means more relevant)
- snippet(): NOT used for evidence. The FTS copy is segmented and must not
  pass as原文. Evidence snippets are built here from pages.body (raw text)
  by locating query terms at their real match positions.

Unicode: indexing and querying normalize with NFKC (see paperpilot.segment),
so evidence terms are normalized the same way. Raw bodies may still contain
compatibility characters ('⼤', 'ﬁ', ...). Highlighting is attempted on the
raw text first; when that fails, positions from the NFKC form are used only
if the mapping is provably 1:1. Otherwise nothing is highlighted and the
result is marked body_matched_no_highlight instead of pretending a title-only
hit -- raw pages.body is never replaced by the normalized copy.

Query safety: user text is segmented first, every resulting token is
quoted and bound as a SQL parameter. Punctuation, quotes and FTS5
operators never reach the MATCH expression. Empty or punctuation-only
queries return [] without touching SQL.
"""

from __future__ import annotations

import re
import unicodedata

from dataclasses import dataclass

from paperpilot.segment import is_cjk, segment_tokens
from paperpilot.store import PaperStore

TITLE_WEIGHT = 8.0
BODY_WEIGHT = 1.0
DEFAULT_TOP_K = 5
SNIPPET_RADIUS = 40
MAX_SNIPPET_WINDOWS = 3

STATUS_HIGHLIGHTED = 'highlighted'
STATUS_BODY_NO_HIGHLIGHT = 'body_matched_no_highlight'
STATUS_TITLE_ONLY = 'title_only'


@dataclass(frozen=True)
class Evidence:
    paper_id: int
    title: str
    page_id: int
    page_no: int
    snippet: str
    score: float
    status: str = STATUS_TITLE_ONLY


@dataclass(frozen=True)
class SnippetResult:
    text: str
    status: str
    body_matched: bool


def evidence_terms(query: str) -> list[str]:
    """Substrings to locate in pages.body (longest first, deduped).

    The query is NFKC-normalized first, the same rule as indexing, so terms
    are always in canonical form. CJK runs contribute the full run (when
    length >= 2) and their bigrams; single CJK characters contribute
    themselves; alnum runs contribute the run as-is.
    """
    query = unicodedata.normalize('NFKC', query)
    terms: list[str] = []
    i = 0
    n = len(query)
    while i < n:
        char = query[i]
        if is_cjk(char):
            j = i
            while j < n and is_cjk(query[j]):
                j += 1
            run = query[i:j]
            if len(run) == 1:
                terms.append(run)
            else:
                terms.append(run)
                terms.extend(run[k:k + 2] for k in range(len(run) - 1))
            i = j
        elif char.isalnum():
            j = i
            while j < n and query[j].isalnum() and not is_cjk(query[j]):
                j += 1
            terms.append(query[i:j])
            i = j
        else:
            i += 1
    seen = set()
    unique = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            unique.append(term)
    return sorted(unique, key=len, reverse=True)


def build_match_query(query: str) -> str:
    """Safe FTS5 MATCH expression: every token double-quoted, params later.

    Returns '' when the query has no tokens (empty / punctuation only).
    """
    tokens = segment_tokens(query)
    if not tokens:
        return ''
    return ' '.join('"' + token.replace('"', '""') + '"' for token in tokens)


def _find_spans(
    body: str,
    terms: list[str],
    mapped: str | None = None,
) -> list[tuple[int, int]]:
    """Real match spans in raw ``body`` (indices interchangeable with ``mapped``).

    Each term is looked up in the raw text first; only when a term has no raw
    hit is the provably 1:1 NFKC form consulted, so every returned span is a
    genuine raw-text position.
    """
    spans: list[tuple[int, int]] = []
    taken: list[tuple[int, int]] = []
    haystacks = (body, mapped) if mapped is not None else (body,)
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for hay in haystacks:
            for match in pattern.finditer(hay):
                start, end = match.start(), match.end()
                if any(not (end <= s or start >= e) for s, e in taken):
                    continue  # longest term already marked this span
                taken.append((start, end))
                spans.append((start, end))
    spans.sort()
    return spans


def _render_snippet(
    body: str,
    spans: list[tuple[int, int]],
    radius: int,
    max_windows: int,
) -> str:
    windows: list[tuple[int, int]] = []
    for start, end in spans:
        left = max(0, start - radius)
        right = min(len(body), end + radius)
        if windows and left <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], right))
        else:
            windows.append((left, right))
        if len(windows) >= max_windows:
            break

    parts = []
    for left, right in windows:
        piece = ''
        cursor = left
        for start, end in spans:
            if start < left or end > right:
                continue
            piece += body[cursor:start] + '[' + body[start:end] + ']'
            cursor = end
        piece += body[cursor:right]
        prefix = '…' if left > 0 else ''
        suffix = '…' if right < len(body) else ''
        parts.append(prefix + piece + suffix)
    return ' … '.join(parts)


def _one_to_one_nfkc(text: str) -> str | None:
    """NFKC(text), but only when every character maps 1:1 onto its own index.

    Returns None whenever offsets cannot be proven to line up ('ﬁ' -> 'fi'
    expands, Hangul jamo compose, mixed cases shift boundaries), so callers
    never paint highlights at unverified positions.
    """
    normalized = unicodedata.normalize('NFKC', text)
    if len(normalized) != len(text):
        return None
    for index, char in enumerate(text):
        if unicodedata.normalize('NFKC', char) != normalized[index]:
            return None
    return normalized


def locate_snippet(
    body: str,
    terms: list[str],
    *,
    radius: int = SNIPPET_RADIUS,
    max_windows: int = MAX_SNIPPET_WINDOWS,
) -> SnippetResult:
    """Locate evidence in raw page text without ever faking positions.

    status:
      - highlighted:                 body matched, markers at real spans
      - body_matched_no_highlight:   body matched, but normalization changed
                                     offsets so highlighting is unsafe
      - title_only:                  no body match (e.g. the hit is in title)
    """
    if not body or not terms:
        return SnippetResult('', STATUS_TITLE_ONLY, False)

    mapped = _one_to_one_nfkc(body)
    spans = _find_spans(body, terms, mapped)
    if spans:
        return SnippetResult(
            _render_snippet(body, spans, radius, max_windows), STATUS_HIGHLIGHTED, True
        )

    if mapped is not None:
        # 1:1 normalization proved and still nothing: body has no such terms.
        return SnippetResult('', STATUS_TITLE_ONLY, False)

    # Length-changing normalization: the body does match via the normalized
    # form, but offsets cannot be proven to map back -- never fake highlights.
    if _find_spans(unicodedata.normalize('NFKC', body), terms):
        return SnippetResult('', STATUS_BODY_NO_HIGHLIGHT, True)
    return SnippetResult('', STATUS_TITLE_ONLY, False)


def build_snippet(
    body: str,
    terms: list[str],
    *,
    radius: int = SNIPPET_RADIUS,
    max_windows: int = MAX_SNIPPET_WINDOWS,
) -> str:
    """Original-text evidence windows around real term matches (text only)."""
    return locate_snippet(body, terms, radius=radius, max_windows=max_windows).text


def search(store: PaperStore, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[Evidence]:
    """Keyword search returning traceable evidence rows (title + page + 原文)."""
    match_expr = build_match_query(query)
    if not match_expr:
        return []
    terms = evidence_terms(query)
    rows = store.conn.execute(
        'SELECT pg.page_id, pg.page_no, pg.body, p.paper_id, p.title, '
        f'bm25(pages_fts, {TITLE_WEIGHT}, {BODY_WEIGHT}) AS score '
        'FROM pages_fts '
        'JOIN pages pg ON pg.page_id = pages_fts.rowid '
        'JOIN papers p ON p.paper_id = pg.paper_id '
        'WHERE pages_fts MATCH ? '
        'ORDER BY score '
        'LIMIT ?',
        (match_expr, int(top_k)),
    ).fetchall()
    results = []
    for row in rows:
        snippet = locate_snippet(row['body'], terms)
        results.append(
            Evidence(
                paper_id=row['paper_id'],
                title=row['title'],
                page_id=row['page_id'],
                page_no=row['page_no'],
                snippet=snippet.text,
                score=row['score'],
                status=snippet.status,
            )
        )
    return results
