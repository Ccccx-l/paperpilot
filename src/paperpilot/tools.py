"""PaperPilot agent tools: import papers, search evidence, read pages.

Exactly three tools, registered as PAPER_TOOLS and kept separate from
mini_harness.tool.box.TOOLS. All three take the harness ``cfg`` via the
standard ``(inp, cfg=CONFIG)`` shape, so the executor injects the active
config (work_space, sandbox_dir, path policy). The database location is
always ``default_db_path(cfg)`` -- models cannot supply one. Every path
goes through mini_harness validate_read; source PDFs are opened read-only
and never copied or modified.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mini_harness.config import CONFIG
from mini_harness.tool.box import ToolDefinition
from mini_harness.tool.path import validate_read

from paperpilot.pdf_import import PaperImportError
from paperpilot.search import (
    STATUS_BODY_NO_HIGHLIGHT,
    STATUS_TITLE_ONLY,
    search,
)
from paperpilot.store import PaperStore, default_db_path


class ImportPapersInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    paper_dir: str = Field(
        'papers',
        min_length=1,
        description='Directory of PDF files (non-recursive *.pdf) or a single PDF file, '
                    'inside the workspace. The index database location is fixed and '
                    'cannot be chosen.',
    )


class SearchPapersInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(
        min_length=1,
        description='2-6 concise keywords (e.g. "LoRA gradient alignment"). Do NOT paste '
                    'a full natural-language question: terms are combined with strict AND '
                    'semantics, so long sentences rarely match.',
    )
    top_k: int = Field(5, ge=1, le=20, description='Maximum number of evidence results.')


class ReadPaperPageInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    paper_id: int = Field(ge=1, description='Paper id from search_papers results.')
    page_no: int = Field(ge=1, description='1-based page number to read in full.')


def _open_store(cfg=CONFIG) -> PaperStore:
    return PaperStore(default_db_path(cfg))


def import_papers(inp: ImportPapersInput, cfg = CONFIG) -> str:
    target = validate_read(Path(inp.paper_dir), cfg=cfg)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob('*.pdf'))
    else:
        raise FileNotFoundError(f'[not found]: no such file or directory: {inp.paper_dir}')
    if not files:
        return f'no PDF files found in {inp.paper_dir}'

    store = _open_store(cfg)
    lines = []
    counts = {'imported': 0, 'updated': 0, 'skipped': 0, 'failed': 0}
    try:
        for file_path in files:
            try:
                result = store.import_pdf(file_path, cfg=cfg)
            except (PaperImportError, PermissionError, OSError) as e:
                counts['failed'] += 1
                lines.append(f'failed: {file_path.name}: {e}')
                continue
            counts[result.status] += 1
            lines.append(
                f'{result.status}: 《{result.title}》 ({result.page_count} pages) '
                f'[{result.source_path}]'
            )
    finally:
        store.close()

    summary = (
        f"{counts['imported']} imported, {counts['updated']} updated, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    return '\n'.join(lines) + f'\n---\n{summary}'


def search_papers(inp: SearchPapersInput, cfg = CONFIG) -> str:
    store = _open_store(cfg)
    try:
        results = search(store, inp.query, top_k=inp.top_k)
    finally:
        store.close()

    if not results:
        return (
            f'no results for query {inp.query!r}. '
            'Try fewer or different keywords; do not cite anything without evidence.'
        )
    lines = []
    for index, ev in enumerate(results, start=1):
        lines.append(
            f'{index}. 《{ev.title}》 p.{ev.page_no} '
            f'(paper_id={ev.paper_id}, score={ev.score:.6f})'
        )
        if ev.snippet:
            lines.append(f'   {ev.snippet}')
        elif ev.status == STATUS_BODY_NO_HIGHLIGHT:
            lines.append(
                '   (body matched; original query terms could not be located '
                'for highlighting -- read_paper_page returns the original text)'
            )
        elif ev.status == STATUS_TITLE_ONLY:
            lines.append('   (title matched; no direct match on this page body)')
        else:
            lines.append('   (match found; no highlight available)')
    return '\n'.join(lines)


def read_paper_page(inp: ReadPaperPageInput, cfg = CONFIG) -> str:
    store = _open_store(cfg)
    try:
        paper = store.conn.execute(
            'SELECT paper_id, title, source_path, page_count FROM papers WHERE paper_id = ?',
            (inp.paper_id,),
        ).fetchone()
        if paper is None:
            return f'[not found]: no paper with paper_id={inp.paper_id}'
        page = store.conn.execute(
            'SELECT page_no, body FROM pages WHERE paper_id = ? AND page_no = ?',
            (inp.paper_id, inp.page_no),
        ).fetchone()
        if page is None:
            return (
                f"[not found]: paper_id={inp.paper_id} 《{paper['title']}》 "
                f"has no page {inp.page_no} (page_count={paper['page_count']})"
            )
        return (
            f"《{paper['title']}》 {paper['source_path']} p.{page['page_no']}\n\n"
            f"{page['body']}"
        )
    finally:
        store.close()


PAPER_TOOLS = [
    ToolDefinition(
        'import_papers',
        'Import PDF files from a workspace directory (non-recursive *.pdf) or a single '
        'PDF into the local paper index. Returns per-file status: imported, updated, '
        'skipped (unchanged, already indexed) or failed. The database path is fixed; '
        'you cannot and need not provide one.',
        ImportPapersInput, import_papers, False,
    ),
    ToolDefinition(
        'search_papers',
        'Search indexed papers with concise keywords and get traceable evidence: paper '
        'id, exact title, page number, original-text snippet and relevance score. '
        'Prefer 2-6 keywords over full sentences (strict AND matching). If the tool '
        'says no results, treat the question as unverified and never invent citations.',
        SearchPapersInput, search_papers, False,
    ),
    ToolDefinition(
        'read_paper_page',
        'Read the full original text of one page using paper_id and page_no from '
        'search_papers. Returns the paper title, normalized source path, page number '
        'and raw page text. Clear error when the paper or page does not exist.',
        ReadPaperPageInput, read_paper_page, False,
    ),
]
