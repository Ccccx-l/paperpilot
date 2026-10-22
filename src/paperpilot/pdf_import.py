"""PDF parsing for PaperPilot: structured extraction only.

Reads one PDF and returns a PaperDocument. This layer never creates a
SQLite database, never writes into the paper directory, and never modifies
the source PDF. A failed parse raises PaperImportError and produces no
partial result, so callers can never record half an import.

Pages are 1-based; blank pages are kept as PageText with empty text.
Title comes from PDF metadata (/Title) and falls back to the file name
without its extension. sha256 covers the raw file bytes.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass
from pathlib import Path

MAX_PDF_BYTES = 64 * 1024 * 1024
MAX_PDF_PAGES = 1000


class PaperImportError(Exception):
    """A PDF cannot be imported; no partial record is ever produced."""


@dataclass(frozen=True)
class PageText:
    page_no: int
    text: str


@dataclass(frozen=True)
class PaperDocument:
    title: str
    source_path: str
    sha256: str
    pages: tuple[PageText, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _fallback_title(path: Path) -> str:
    return path.stem or path.name


def _normalize(text: str | None) -> str:
    if not text:
        return ''
    return text.replace('\r\n', '\n').replace('\r', '\n')


def extract_paper(
    path: Path,
    *,
    max_bytes: int = MAX_PDF_BYTES,
    max_pages: int = MAX_PDF_PAGES,
) -> PaperDocument:
    """Extract one PDF into a PaperDocument, or raise PaperImportError.

    FileNotFoundError propagates untouched for a missing path; everything
    else (corrupt file, encrypted file, oversize file, abnormal page count)
    is reported as PaperImportError with the reason.
    """
    from pypdf import PdfReader

    path = Path(path)
    size = path.stat().st_size
    if size > max_bytes:
        raise PaperImportError(
            f'oversize PDF: {size} bytes exceeds limit {max_bytes} bytes: {path}'
        )
    sha256 = _sha256(path)

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise PaperImportError(f'encrypted PDF is not supported: {path}')
        raw_pages = list(reader.pages)
    except PaperImportError:
        raise
    except Exception as e:
        raise PaperImportError(f'corrupt or unreadable PDF: {path}: {type(e).__name__}: {e}') from e

    if len(raw_pages) == 0:
        raise PaperImportError(f'abnormal page count 0 (PDF has no pages): {path}')
    if len(raw_pages) > max_pages:
        raise PaperImportError(
            f'abnormal page count {len(raw_pages)} exceeds limit {max_pages}: {path}'
        )

    pages = []
    for index, page in enumerate(raw_pages, start=1):
        try:
            text = _normalize(page.extract_text())
        except Exception as e:
            raise PaperImportError(
                f'failed to extract text on page {index}: {path}: {type(e).__name__}: {e}'
            ) from e
        pages.append(PageText(page_no=index, text=text))

    title = ''
    metadata = getattr(reader, 'metadata', None)
    if metadata is not None:
        title = _normalize(getattr(metadata, 'title', None) or '').strip()
    if not title:
        title = _fallback_title(path)

    return PaperDocument(
        title=title,
        source_path=str(path),
        sha256=sha256,
        pages=tuple(pages),
    )
