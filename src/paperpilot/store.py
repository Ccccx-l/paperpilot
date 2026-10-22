"""SQLite + FTS5 storage for PaperPilot.

Data model (identity = normalized source_path; sha256 is change detection
ONLY and carries no unique constraint):

- same normalized path + same sha256    -> idempotent skip
- same normalized path + changed sha256 -> update paper/pages/index in one
  transaction
- different paths + same sha256         -> separate records (v1 allows it)
- any failure                           -> rollback, no partial paper, no
  orphan pages, no stale index

Text boundaries:
- ``pages.body`` holds the RAW page text, never rewritten.
- ``pages_fts`` holds SEARCH-NORMALIZED copies (segmented title and body
  in its ``title`` / ``body_seg`` columns). MATCH and BM25 read those
  segmented columns; they must never be shown to a user as原文.
- ``pages_fts.rowid`` is always ``pages.page_id``, so retrieval can join
  back to the raw body and report the real title and page number.

Every import path goes through mini_harness path validation first
(validate_read: work_space bounds, deny lists, symlink resolution); the
recorded source_path is the validated file's path made relative to
cfg.work_space in POSIX form.
"""

from __future__ import annotations

import sqlite3

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mini_harness.config import CONFIG
from mini_harness.tool.path import validate_read

from paperpilot.pdf_import import PaperDocument, extract_paper
from paperpilot.segment import SEGMENT_VERSION, segment

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
  paper_id    INTEGER PRIMARY KEY,
  title       TEXT NOT NULL,
  source_path TEXT NOT NULL,
  sha256      TEXT NOT NULL,
  page_count  INTEGER NOT NULL,
  imported_at TEXT NOT NULL,
  UNIQUE(source_path)
);

CREATE TABLE IF NOT EXISTS pages (
  page_id  INTEGER PRIMARY KEY,
  paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  page_no  INTEGER NOT NULL,
  body     TEXT NOT NULL,
  UNIQUE(paper_id, page_no)
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_sha256 ON papers(sha256);

CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
  title,
  body_seg,
  tokenize='unicode61 remove_diacritics 2'
);
"""

SEGMENT_VERSION_KEY = 'segment_version'


class PaperStoreVersionError(Exception):
    """The database was written by a newer segmenter; refuse to downgrade."""


@dataclass(frozen=True)
class ImportResult:
    status: str  # 'imported' | 'updated' | 'skipped'
    paper_id: int
    title: str
    source_path: str
    sha256: str
    page_count: int


def normalize_source_path(path: Path, cfg=CONFIG) -> tuple[str, Path]:
    """Validate path via the harness policy and return (identity, resolved).

    identity is the validated file's path relative to cfg.work_space in
    POSIX form; the same file therefore has one stable identity no matter
    how the caller spelled the path.
    """
    resolved = validate_read(Path(path), cfg=cfg)
    try:
        rel = resolved.relative_to(cfg.work_space.resolve())
    except ValueError:
        rel = resolved.relative_to(cfg.work_space)
    return rel.as_posix(), resolved


def default_db_path(cfg=CONFIG) -> Path:
    """Default DB location: inside sandbox/, the harness' writable area."""
    return cfg.sandbox_dir / 'paperpilot' / 'index.db'


class PaperStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        try:
            self.init_schema()
            self._ensure_index_version()
        except BaseException:
            self.conn.close()
            raise

    def init_schema(self) -> None:
        """Create tables and the FTS index; safe to run repeatedly.

        Runs in autocommit (no transaction of its own).
        """
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---- index rebuild / segment-version migration --------------------
    #
    # Transaction ownership: the public entry points below (rebuild_index,
    # import_document) each open ONE transaction (BEGIN IMMEDIATE) and are
    # never called inside another. The *_locked helpers assume that
    # transaction is already open and never issue BEGIN.

    def _segment_version(self) -> int | None:
        row = self.conn.execute(
            'SELECT value FROM meta WHERE key = ?', (SEGMENT_VERSION_KEY,)
        ).fetchone()
        return int(row['value']) if row is not None else None

    def _rebuild_index_locked(self) -> int:
        """Recompute pages_fts from raw pages.body / papers.title.

        Only the FTS copy changes: pages.body, page ids, paper metadata and
        sha256 stay untouched. Caller owns the transaction.
        """
        conn = self.conn
        conn.execute('DELETE FROM pages_fts')
        rows = conn.execute(
            'SELECT pg.page_id, pg.body, p.title FROM pages pg '
            'JOIN papers p ON p.paper_id = pg.paper_id'
        ).fetchall()
        for row in rows:
            conn.execute(
                'INSERT INTO pages_fts (rowid, title, body_seg) VALUES (?, ?, ?)',
                (row['page_id'], segment(row['title']), segment(row['body'])),
            )
        return len(rows)

    def rebuild_index(self) -> int:
        """Explicitly rebuild the FTS index and stamp SEGMENT_VERSION.

        Rebuild + version update share one transaction; any failure rolls
        both back. Idempotent. Returns the number of indexed pages.
        """
        conn = self.conn
        try:
            conn.execute('BEGIN IMMEDIATE')
            count = self._rebuild_index_locked()
            conn.execute(
                'INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)',
                (SEGMENT_VERSION_KEY, str(SEGMENT_VERSION)),
            )
            conn.execute('COMMIT')
            return count
        except BaseException:
            conn.execute('ROLLBACK')
            raise

    def _ensure_index_version(self) -> None:
        current = self._segment_version()
        if current is not None and current > SEGMENT_VERSION:
            raise PaperStoreVersionError(
                f'database segment_version={current} is newer than supported '
                f'{SEGMENT_VERSION}; refusing to downgrade the index: {self.db_path}'
            )
        if current != SEGMENT_VERSION:
            # Missing meta / missing row / older version -> rebuild in place.
            self.rebuild_index()

    def import_pdf(self, path: Path, *, cfg=CONFIG) -> ImportResult:
        """Import one PDF under the path policy of ``cfg``.

        Parsing happens before any write; a parse error leaves the database
        untouched. Writes run in a single transaction.
        """
        identity, resolved = normalize_source_path(path, cfg=cfg)
        doc = extract_paper(resolved)
        return self.import_document(doc, source_path=identity)

    def import_document(self, doc: PaperDocument, *, source_path: str) -> ImportResult:
        """Record an already-parsed document under its normalized identity."""
        conn = self.conn
        try:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute(
                'SELECT paper_id, sha256 FROM papers WHERE source_path = ?',
                (source_path,),
            ).fetchone()
            now = datetime.now(timezone.utc).isoformat()

            if row is not None and row['sha256'] == doc.sha256:
                conn.execute('ROLLBACK')
                return ImportResult(
                    'skipped', row['paper_id'], doc.title, source_path,
                    doc.sha256, doc.page_count,
                )

            if row is not None:
                paper_id = row['paper_id']
                conn.execute(
                    'UPDATE papers SET title = ?, sha256 = ?, page_count = ?, '
                    'imported_at = ? WHERE paper_id = ?',
                    (doc.title, doc.sha256, doc.page_count, now, paper_id),
                )
                conn.execute(
                    'DELETE FROM pages_fts WHERE rowid IN '
                    '(SELECT page_id FROM pages WHERE paper_id = ?)',
                    (paper_id,),
                )
                conn.execute('DELETE FROM pages WHERE paper_id = ?', (paper_id,))
                status = 'updated'
            else:
                cursor = conn.execute(
                    'INSERT INTO papers (title, source_path, sha256, page_count, imported_at) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (doc.title, source_path, doc.sha256, doc.page_count, now),
                )
                paper_id = cursor.lastrowid
                status = 'imported'

            title_seg = segment(doc.title)
            for page in doc.pages:
                cursor = conn.execute(
                    'INSERT INTO pages (paper_id, page_no, body) VALUES (?, ?, ?)',
                    (paper_id, page.page_no, page.text),
                )
                conn.execute(
                    'INSERT INTO pages_fts (rowid, title, body_seg) VALUES (?, ?, ?)',
                    (cursor.lastrowid, title_seg, segment(page.text)),
                )
            conn.execute('COMMIT')
            return ImportResult(
                status, paper_id, doc.title, source_path, doc.sha256, doc.page_count,
            )
        except BaseException:
            conn.execute('ROLLBACK')
            raise

    def get_paper(self, source_path: str) -> sqlite3.Row | None:
        return self.conn.execute(
            'SELECT * FROM papers WHERE source_path = ?', (source_path,)
        ).fetchone()

    def page_by_id(self, page_id: int) -> sqlite3.Row | None:
        """Raw page row: join key from pages_fts.rowid back to原文."""
        return self.conn.execute(
            'SELECT page_id, paper_id, page_no, body FROM pages WHERE page_id = ?',
            (page_id,),
        ).fetchone()
