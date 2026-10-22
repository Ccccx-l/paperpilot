"""Tests for paperpilot.store: schema, identity rules, transactions, safety."""

import logging
import tempfile
import unittest

from dataclasses import replace
from pathlib import Path
from unittest import mock

from mini_harness.config import CONFIG

from paperpilot.pdf_import import PaperDocument, PageText
from paperpilot.search import search
from paperpilot.segment import SEGMENT_VERSION
from paperpilot.store import (
    SEGMENT_VERSION_KEY,
    ImportResult,
    PaperStore,
    PaperStoreVersionError,
    default_db_path,
    normalize_source_path,
)

from tests.test_paper_pdf_import import make_pdf


def make_doc(title, pages, sha='deadbeef', source_path='papers/x.pdf'):
    return PaperDocument(
        title=title,
        source_path=source_path,
        sha256=sha,
        pages=tuple(PageText(page_no=i + 1, text=t) for i, t in enumerate(pages)),
    )


class StoreTestBase(unittest.TestCase):
    def setUp(self):
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.ws = self.tmp / 'ws'
        self.papers_dir = self.ws / 'papers'
        self.papers_dir.mkdir(parents=True)
        self.cfg = replace(CONFIG, work_space=self.ws)
        self.store = PaperStore(self.tmp / 'index.db')
        self.addCleanup(self.store.close)

    def write_pdf(self, name, page_texts, title=None) -> Path:
        target = self.papers_dir / name
        target.write_bytes(make_pdf(page_texts, title=title))
        return target


class TestSchema(StoreTestBase):
    def test_init_and_reinit_are_idempotent(self):
        self.store.init_schema()
        self.store.init_schema()
        again = PaperStore(self.tmp / 'index.db')
        self.addCleanup(again.close)
        again.init_schema()
        tables = {
            row[0] for row in again.conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table')"
            )
        }
        self.assertLessEqual({'papers', 'pages', 'pages_fts'}, tables)

    def test_default_db_path_is_under_sandbox(self):
        path = default_db_path(self.cfg)
        self.assertEqual(path, self.ws / 'sandbox' / 'paperpilot' / 'index.db')
        self.assertTrue(str(path).startswith(str(self.cfg.sandbox_dir)))

    def test_persistence_across_reopen(self):
        self.store.import_pdf(self.write_pdf('p.pdf', ['hello body']), cfg=self.cfg)
        self.store.close()
        reopened = PaperStore(self.tmp / 'index.db')
        self.addCleanup(reopened.close)
        row = reopened.get_paper('papers/p.pdf')
        self.assertIsNotNone(row)
        self.assertEqual(row['page_count'], 1)


class TestIdentityAndImportRules(StoreTestBase):
    def test_identity_is_work_space_relative_posix(self):
        identity, resolved = normalize_source_path(self.papers_dir / 'a.pdf', cfg=self.cfg)
        self.assertEqual(identity, 'papers/a.pdf')
        self.assertTrue(resolved.is_absolute())

    def test_identity_is_stable_across_path_spellings(self):
        self.write_pdf('a.pdf', ['x'])
        one, _ = normalize_source_path(self.papers_dir / 'a.pdf', cfg=self.cfg)
        two, _ = normalize_source_path(self.ws / 'papers' / '..' / 'papers' / 'a.pdf', cfg=self.cfg)
        self.assertEqual(one, two)

    def test_import_writes_paper_pages_and_raw_body(self):
        # 中文内容走结构化导入：合成 PDF 的简单字体无法承载 CJK 字面串，
        # PDF 往返路径由 ASCII 用例覆盖。
        result = self.store.import_document(
            make_doc('标题甲', ['原文内容 ABC']), source_path='papers/a.pdf'
        )
        self.assertEqual(result.status, 'imported')
        self.assertEqual(result.source_path, 'papers/a.pdf')

        paper = self.store.get_paper('papers/a.pdf')
        self.assertEqual(paper['title'], '标题甲')
        page = self.store.conn.execute('SELECT * FROM pages').fetchone()
        self.assertEqual(page['body'], '原文内容 ABC')  # raw text, not segmented
        self.assertEqual(page['page_no'], 1)

    def test_pdf_roundtrip_records_ascii_page(self):
        self.write_pdf('a.pdf', ['hello pdf body'], title='Ascii Title')
        result = self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)
        self.assertEqual(result.status, 'imported')
        page = self.store.conn.execute('SELECT body FROM pages').fetchone()[0]
        self.assertEqual(page, 'hello pdf body')

    def test_same_file_reimport_is_idempotent(self):
        self.write_pdf('a.pdf', ['same content'])
        first = self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)
        second = self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)
        self.assertEqual(first.status, 'imported')
        self.assertEqual(second.status, 'skipped')
        self.assertEqual(first.paper_id, second.paper_id)
        count = self.store.conn.execute('SELECT count(*) FROM papers').fetchone()[0]
        self.assertEqual(count, 1)

    def test_same_path_new_content_updates_in_place(self):
        self.write_pdf('a.pdf', ['old text v1'])
        self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)
        self.write_pdf('a.pdf', ['new text v2', 'second page'])
        result = self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)

        self.assertEqual(result.status, 'updated')
        paper = self.store.get_paper('papers/a.pdf')
        self.assertEqual(paper['page_count'], 2)
        bodies = [r[0] for r in self.store.conn.execute('SELECT body FROM pages ORDER BY page_no')]
        self.assertEqual(bodies, ['new text v2', 'second page'])
        stale = self.store.conn.execute(
            "SELECT count(*) FROM pages_fts WHERE pages_fts MATCH 'old'"
        ).fetchone()[0]
        self.assertEqual(stale, 0, 'stale FTS rows must not survive an update')
        total = self.store.conn.execute('SELECT count(*) FROM pages').fetchone()[0]
        self.assertEqual(total, 2, 'no orphan pages')

    def test_same_content_different_paths_are_separate(self):
        self.write_pdf('a.pdf', ['identical bytes'])
        (self.papers_dir / 'b.pdf').write_bytes((self.papers_dir / 'a.pdf').read_bytes())
        r1 = self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)
        r2 = self.store.import_pdf(self.papers_dir / 'b.pdf', cfg=self.cfg)
        self.assertEqual(r1.status, 'imported')
        self.assertEqual(r2.status, 'imported')
        self.assertNotEqual(r1.paper_id, r2.paper_id)
        self.assertEqual(r1.sha256, r2.sha256)
        count = self.store.conn.execute('SELECT count(*) FROM papers').fetchone()[0]
        self.assertEqual(count, 2)

    def test_import_result_is_structured(self):
        self.write_pdf('a.pdf', ['x'])
        result = self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)
        self.assertIsInstance(result, ImportResult)
        self.assertIn(result.status, ('imported', 'updated', 'skipped'))
        self.assertEqual(result.page_count, 1)


class TestRowidMapping(StoreTestBase):
    def test_fts_rowid_equals_page_id(self):
        self.write_pdf('a.pdf', ['page one', 'page two', 'page three'])
        self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)
        fts_rowids = {r[0] for r in self.store.conn.execute('SELECT rowid FROM pages_fts')}
        page_ids = {r[0] for r in self.store.conn.execute('SELECT page_id FROM pages')}
        self.assertEqual(fts_rowids, page_ids)

    def test_join_from_fts_to_original_body(self):
        self.store.import_document(
            make_doc('论文标题', ['研究量子计算方法']), source_path='papers/a.pdf'
        )
        row = self.store.conn.execute(
            'SELECT pg.page_id, pg.page_no, pg.body, p.title FROM pages_fts '
            'JOIN pages pg ON pg.page_id = pages_fts.rowid '
            'JOIN papers p ON p.paper_id = pg.paper_id '
            "WHERE pages_fts MATCH '\"量子\"'"
        ).fetchone()
        self.assertEqual(row['body'], '研究量子计算方法')
        self.assertEqual(row['page_no'], 1)
        self.assertIsNotNone(self.store.page_by_id(row['page_id']))

    def test_fts_index_text_is_segmented_not_original(self):
        self.store.import_document(
            make_doc('论文标题', ['研究量子计算方法']), source_path='papers/a.pdf'
        )
        body = self.store.conn.execute('SELECT body FROM pages').fetchone()[0]
        seg = self.store.conn.execute('SELECT body_seg FROM pages_fts').fetchone()[0]
        self.assertEqual(body, '研究量子计算方法')
        self.assertIn('量子 子计', seg)
        self.assertNotEqual(body, seg)


class TestTransactions(StoreTestBase):
    def test_failure_mid_import_rolls_back_everything(self):
        self.write_pdf('a.pdf', ['p1', 'p2'])
        calls = {'n': 0}
        original_segment = __import__('paperpilot.store', fromlist=['segment']).segment

        def flaky(text):
            calls['n'] += 1
            if calls['n'] >= 3:
                raise RuntimeError('synthetic failure during FTS insert')
            return original_segment(text)

        with mock.patch('paperpilot.store.segment', side_effect=flaky):
            with self.assertRaises(RuntimeError):
                self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)

        self.assertEqual(self.store.conn.execute('SELECT count(*) FROM papers').fetchone()[0], 0)
        self.assertEqual(self.store.conn.execute('SELECT count(*) FROM pages').fetchone()[0], 0)
        self.assertEqual(
            self.store.conn.execute('SELECT count(*) FROM pages_fts').fetchone()[0], 0
        )

    def test_update_failure_keeps_previous_record_intact(self):
        self.write_pdf('a.pdf', ['good v1'])
        first = self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)
        self.write_pdf('a.pdf', ['good v2', 'extra'])
        with mock.patch('paperpilot.store.segment', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                self.store.import_pdf(self.papers_dir / 'a.pdf', cfg=self.cfg)
        row = self.store.get_paper('papers/a.pdf')
        self.assertEqual(row['sha256'], first.sha256, 'rollback must keep the v1 sha256')
        self.assertEqual(row['page_count'], 1)
        body = self.store.conn.execute('SELECT body FROM pages').fetchone()[0]
        self.assertEqual(body, 'good v1', 'rollback must keep the previous pages')

    def test_parse_failure_leaves_db_untouched(self):
        bad = self.papers_dir / 'bad.pdf'
        bad.write_bytes(b'%PDF-1.4 broken')
        with self.assertRaises(Exception):
            self.store.import_pdf(bad, cfg=self.cfg)
        self.assertEqual(self.store.conn.execute('SELECT count(*) FROM papers').fetchone()[0], 0)


class TestPathSafety(StoreTestBase):
    def test_path_outside_work_space_rejected(self):
        outside = self.tmp / 'outside.pdf'
        outside.write_bytes(make_pdf(['x']))
        with self.assertRaises(PermissionError):
            self.store.import_pdf(outside, cfg=self.cfg)

    def test_traversal_path_rejected(self):
        self.write_pdf('a.pdf', ['x'])
        escape = self.papers_dir / '..' / '..' / 'outside.pdf'
        (self.tmp / 'outside.pdf').write_bytes(make_pdf(['x']))
        with self.assertRaises(PermissionError):
            self.store.import_pdf(escape, cfg=self.cfg)

    def test_sensitive_name_rejected(self):
        secret = self.ws / '.env'
        secret.write_bytes(make_pdf(['x']))
        with self.assertRaises(PermissionError):
            self.store.import_pdf(secret, cfg=self.cfg)

    def test_symlink_escaping_work_space_rejected(self):
        outside = self.tmp / 'outside.pdf'
        outside.write_bytes(make_pdf(['x']))
        link = self.papers_dir / 'link.pdf'
        link.symlink_to(outside)
        with self.assertRaises(PermissionError):
            self.store.import_pdf(link, cfg=self.cfg)

    def test_symlink_inside_work_space_maps_to_real_identity(self):
        self.write_pdf('real.pdf', ['linked content'])
        link = self.papers_dir / 'link.pdf'
        link.symlink_to(self.papers_dir / 'real.pdf')
        result = self.store.import_pdf(link, cfg=self.cfg)
        self.assertEqual(result.source_path, 'papers/real.pdf')
        again = self.store.import_pdf(self.papers_dir / 'real.pdf', cfg=self.cfg)
        self.assertEqual(again.status, 'skipped', 'same file via symlink and real path is one record')

    def test_import_document_requires_normalized_identity(self):
        doc = make_doc('T', ['body'], sha='abc')
        result = self.store.import_document(doc, source_path='papers/x.pdf')
        self.assertEqual(result.status, 'imported')
        self.assertEqual(self.store.get_paper('papers/x.pdf')['sha256'], 'abc')


class TestIndexRebuild(StoreTestBase):
    """Explicit rebuild + segment-version migration on temporary databases."""

    COMPAT_BODY = '本⽂调研⼤语⾔模型的发展'

    def import_compat(self, source_path='papers/c.pdf'):
        return self.store.import_document(
            make_doc('兼容汉字论文', [self.COMPAT_BODY], sha='compat',
                     source_path=source_path),
            source_path=source_path,
        )

    def plant_stale_index(self):
        """Simulate a pre-NFKC index: compat runs were split/dropped."""
        self.store.conn.execute('DELETE FROM pages_fts')
        for page_id, in self.store.conn.execute('SELECT page_id FROM pages'):
            self.store.conn.execute(
                'INSERT INTO pages_fts (rowid, title, body_seg) VALUES (?, ?, ?)',
                (page_id, 'x', '语 模 型 模型'),
            )

    def set_version(self, value):
        self.store.conn.execute(
            'INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)',
            (SEGMENT_VERSION_KEY, str(value)),
        )

    def test_rebuild_fixes_stale_index(self):
        self.import_compat()
        self.plant_stale_index()
        self.assertEqual(search(self.store, '大语言模型'), [], 'stale index must miss')
        count = self.store.rebuild_index()
        self.assertEqual(count, 1)
        results = search(self.store, '大语言模型')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, '兼容汉字论文')

    def test_rebuild_preserves_rowid_page_id_mapping(self):
        self.import_compat()
        before = {r[0] for r in self.store.conn.execute('SELECT page_id FROM pages')}
        self.plant_stale_index()
        self.store.rebuild_index()
        fts = {r[0] for r in self.store.conn.execute('SELECT rowid FROM pages_fts')}
        self.assertEqual(fts, before)

    def test_rebuild_is_idempotent(self):
        self.import_compat()
        self.plant_stale_index()
        first = self.store.rebuild_index()
        snapshot = self.store.conn.execute(
            'SELECT rowid, title, body_seg FROM pages_fts ORDER BY rowid'
        ).fetchall()
        second = self.store.rebuild_index()
        self.assertEqual(first, second)
        self.assertEqual(
            self.store.conn.execute(
                'SELECT rowid, title, body_seg FROM pages_fts ORDER BY rowid'
            ).fetchall(),
            snapshot,
        )

    def test_old_version_triggers_migration_and_updates_version(self):
        self.import_compat()
        self.plant_stale_index()
        self.set_version(1)
        self.store.close()

        reopened = PaperStore(self.tmp / 'index.db')
        self.addCleanup(reopened.close)
        row = reopened.conn.execute(
            'SELECT value FROM meta WHERE key = ?', (SEGMENT_VERSION_KEY,)
        ).fetchone()
        self.assertEqual(row['value'], str(SEGMENT_VERSION))
        self.assertEqual(len(search(reopened, '大语言模型')), 1)

    def test_missing_meta_row_migrates_like_old_version(self):
        self.import_compat()
        self.plant_stale_index()
        self.store.conn.execute('DELETE FROM meta')
        self.store.close()

        reopened = PaperStore(self.tmp / 'index.db')
        self.addCleanup(reopened.close)
        self.assertEqual(len(search(reopened, '大语言模型')), 1)

    def test_same_sha_reimport_skips_and_does_not_rebuild(self):
        self.import_compat()
        self.plant_stale_index()
        result = self.import_compat()
        self.assertEqual(result.status, 'skipped', 'unchanged file is idempotent')
        self.assertEqual(search(self.store, '大语言模型'), [], 'skip must not pretend to rebuild')
        self.store.rebuild_index()
        self.assertEqual(len(search(self.store, '大语言模型')), 1)

    def test_rebuild_failure_rolls_back_index_and_version(self):
        self.import_compat()
        self.plant_stale_index()
        self.set_version(1)
        before = self.store.conn.execute(
            'SELECT rowid, title, body_seg FROM pages_fts ORDER BY rowid'
        ).fetchall()

        import paperpilot.store as store_mod
        original = store_mod.segment
        calls = {'n': 0}

        def flaky(text):
            calls['n'] += 1
            if calls['n'] >= 2:
                raise RuntimeError('synthetic failure mid-rebuild')
            return original(text)

        with mock.patch('paperpilot.store.segment', side_effect=flaky):
            with self.assertRaises(RuntimeError):
                self.store.rebuild_index()

        self.assertEqual(
            self.store.conn.execute(
                'SELECT rowid, title, body_seg FROM pages_fts ORDER BY rowid'
            ).fetchall(),
            before,
            'index changes must roll back',
        )
        row = self.store.conn.execute(
            'SELECT value FROM meta WHERE key = ?', (SEGMENT_VERSION_KEY,)
        ).fetchone()
        self.assertEqual(row['value'], '1', 'version stamp must roll back too')

    def test_higher_version_refuses_downgrade(self):
        self.import_compat()
        self.set_version(99)
        self.store.close()

        with self.assertRaises(PaperStoreVersionError):
            PaperStore(self.tmp / 'index.db')
        import sqlite3
        conn = sqlite3.connect(str(self.tmp / 'index.db'))
        try:
            count = conn.execute('SELECT count(*) FROM pages_fts').fetchone()[0]
            version = conn.execute(
                'SELECT value FROM meta WHERE key = ?', (SEGMENT_VERSION_KEY,)
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(version, '99', 'refusal must not downgrade the stamp')
        self.assertGreater(count, 0, 'refusal must not touch the index')


if __name__ == '__main__':
    unittest.main()
