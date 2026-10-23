"""Offline tests for the three PaperPilot agent tools (no model API calls)."""

import logging
import tempfile
import unittest

from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

from mini_harness.config import CONFIG
from mini_harness.tool.box import TOOLS, ToolDefinition, ToolExecution

from paperpilot.pdf_import import PaperDocument, PageText
from paperpilot.store import PaperStore, default_db_path
from paperpilot.tools import (
    PAPER_TOOLS,
    ImportPapersInput,
    ReadPaperPageInput,
    SearchPapersInput,
    import_papers,
    read_paper_page,
    search_papers,
)

from tests._support import RootSessionGuard, make_call
from tests.test_paper_pdf_import import make_pdf


def make_doc(title, pages, sha='sha', source_path='papers/x.pdf'):
    return PaperDocument(
        title=title,
        source_path=source_path,
        sha256=sha,
        pages=tuple(PageText(page_no=i + 1, text=t) for i, t in enumerate(pages)),
    )


class ToolsTestBase(RootSessionGuard, unittest.TestCase):
    def setUp(self):
        super().setUp()
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.ws = self.tmp / 'ws'
        self.papers_dir = self.ws / 'papers'
        self.papers_dir.mkdir(parents=True)
        self.cfg = replace(CONFIG, work_space=self.ws)

    def write_pdf(self, name, page_texts, title=None) -> Path:
        target = self.papers_dir / name
        target.write_bytes(make_pdf(page_texts, title=title))
        return target

    def add_doc(self, name, title, pages):
        source_path = f'papers/{name}'
        store = PaperStore(default_db_path(self.cfg))
        try:
            store.import_document(
                make_doc(title, pages, sha=f'sha-{name}', source_path=source_path),
                source_path=source_path,
            )
        finally:
            store.close()


class TestRegistration(ToolsTestBase):
    def test_exactly_three_tools(self):
        self.assertEqual(
            [t.name for t in PAPER_TOOLS],
            ['import_papers', 'search_papers', 'read_paper_page'],
        )
        self.assertTrue(all(isinstance(t, ToolDefinition) for t in PAPER_TOOLS))
        self.assertTrue(all(t.risky is False for t in PAPER_TOOLS))

    def test_isolated_from_default_tools(self):
        paper_names = {t.name for t in PAPER_TOOLS}
        default_names = {t.name for t in TOOLS}
        self.assertEqual(paper_names & default_names, set())
        self.assertEqual(len(TOOLS), 9, 'default TOOLS must stay untouched')
        self.assertIn('read_file', default_names)

    def test_all_tool_functions_accept_cfg(self):
        for tool in PAPER_TOOLS:
            with self.subTest(tool=tool.name):
                params = inspect_signature(tool.function)
                self.assertIn('cfg', params)

    def test_input_models_forbid_extra_fields(self):
        for model, kwargs in (
            (ImportPapersInput, {'paper_dir': 'papers'}),
            (SearchPapersInput, {'query': 'a'}),
            (ReadPaperPageInput, {'paper_id': 1, 'page_no': 1}),
        ):
            with self.subTest(model=model.__name__):
                with self.assertRaises(ValidationError):
                    model(**kwargs, db_path='/tmp/evil.db')
                with self.assertRaises(ValidationError):
                    model(**kwargs, unknown_field='x')

    def test_numeric_bounds(self):
        with self.assertRaises(ValidationError):
            SearchPapersInput(query='a', top_k=0)
        with self.assertRaises(ValidationError):
            SearchPapersInput(query='a', top_k=21)
        with self.assertRaises(ValidationError):
            SearchPapersInput(query='')
        with self.assertRaises(ValidationError):
            ReadPaperPageInput(paper_id=0, page_no=1)
        with self.assertRaises(ValidationError):
            ReadPaperPageInput(paper_id=1, page_no=0)

    def test_search_query_description_matches_implementation(self):
        """描述必须反映双路检索现状，不得回到「strict AND 长句必不中」的误导措辞。"""
        desc = SearchPapersInput.model_fields['query'].description
        self.assertIn('strict AND', desc, 'strict pass still exists')
        self.assertIn('looser', desc, 'loose recall pass must be mentioned')
        self.assertIn('expansion', desc, 'controlled expansion must be mentioned')
        self.assertIn('sub-quer', desc, 'preconfigured sub-query fusion must be mentioned')
        self.assertIn('read_paper_page', desc, 'verification via original text required')
        # 禁止重新引入已被否证的绝对化措辞
        self.assertNotIn('rarely match', desc)
        self.assertNotIn('Do NOT paste', desc)
        self.assertNotIn('semantic understanding, ', desc.replace('not semantic understanding', ''))
        self.assertIn('not semantic understanding', desc,
                      'must stay explicit that this is lexical, not semantic')

    def test_search_description_does_not_claim_solved_semantics(self):
        desc = SearchPapersInput.model_fields['query'].description
        for overclaim in ('semantic search', 'understands meaning', 'topic relevance solved',
                          'vector', 'embedding', 're-rank'):
            self.assertNotIn(overclaim, desc.lower())


def inspect_signature(func):
    import inspect
    return inspect.signature(func).parameters


class TestImportTool(ToolsTestBase):
    def test_import_success(self):
        self.write_pdf('a.pdf', ['hello pdf body'], title='Ascii Title')
        out = import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        self.assertIn('imported: 《Ascii Title》', out)
        self.assertIn('1 imported, 0 updated, 0 skipped, 0 failed', out)
        self.assertTrue(default_db_path(self.cfg).is_file())
        self.assertTrue(str(default_db_path(self.cfg)).startswith(str(self.cfg.sandbox_dir)))

    def test_reimport_is_skipped(self):
        self.write_pdf('a.pdf', ['stable content'])
        import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        out = import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        self.assertIn('skipped:', out)
        self.assertIn('0 imported, 0 updated, 1 skipped, 0 failed', out)

    def test_changed_content_is_updated(self):
        self.write_pdf('a.pdf', ['version one text'])
        import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        self.write_pdf('a.pdf', ['version two text'])
        out = import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        self.assertIn('updated:', out)
        store = PaperStore(default_db_path(self.cfg))
        try:
            body = store.conn.execute('SELECT body FROM pages').fetchone()[0]
        finally:
            store.close()
        self.assertEqual(body, 'version two text')

    def test_bad_pdf_reports_failed_without_partial_record(self):
        self.write_pdf('good.pdf', ['good content'])
        (self.papers_dir / 'bad.pdf').write_bytes(b'%PDF-1.4 broken')
        out = import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        self.assertIn('failed: bad.pdf:', out)
        self.assertIn('1 imported', out)
        self.assertIn('1 failed', out)
        store = PaperStore(default_db_path(self.cfg))
        try:
            rows = store.conn.execute('SELECT source_path FROM papers').fetchall()
        finally:
            store.close()
        self.assertEqual([r[0] for r in rows], ['papers/good.pdf'])

    def test_single_file_input(self):
        self.write_pdf('only.pdf', ['single file'])
        out = import_papers(ImportPapersInput(paper_dir='papers/only.pdf'), cfg=self.cfg)
        self.assertIn('imported:', out)

    def test_empty_directory(self):
        out = import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        self.assertIn('no PDF files found', out)

    def test_missing_directory(self):
        out = import_papers(ImportPapersInput(paper_dir='papers/nope'), cfg=self.cfg)
        self.assertIn('[not found]', out)
        self.assertIn('papers/nope', out, 'error must keep the concrete path')

    def test_missing_directory_does_not_raise(self):
        # 与 search_papers / read_paper_page 一致：缺失路径是可控返回，
        # 不得让裸 FileNotFoundError 逃逸成通用 EXECUTE_FAILED
        out = import_papers(ImportPapersInput(paper_dir='nonexistent'), cfg=self.cfg)
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith('[not found]'), out)

    def test_missing_single_file_reports_not_found(self):
        out = import_papers(ImportPapersInput(paper_dir='papers/ghost.pdf'), cfg=self.cfg)
        self.assertIn('[not found]', out)

    def test_normal_import_still_works_after_fix(self):
        self.write_pdf('ok.pdf', ['正常导入内容'])
        out = import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        self.assertIn('imported:', out)
        self.assertIn('1 imported', out)


class TestSearchTool(ToolsTestBase):
    def test_search_returns_title_page_and_evidence(self):
        self.add_doc('q', '量子计算导论', ['本文研究量子计算的基本原理。'])
        out = search_papers(SearchPapersInput(query='量子计算'), cfg=self.cfg)
        self.assertIn('《量子计算导论》 p.1', out)
        self.assertIn('paper_id=', out)
        self.assertIn('score=', out)
        self.assertIn('[量子计算]', out, 'evidence must quote original text')

    def test_search_no_results_message(self):
        self.add_doc('q', '量子计算导论', ['本文研究量子计算的基本原理。'])
        out = search_papers(SearchPapersInput(query='绝对没有的词'), cfg=self.cfg)
        self.assertIn('no results', out)
        self.assertNotIn('《', out, 'no invented citations')

    def test_search_top_k(self):
        for i in range(4):
            self.add_doc(f'p{i}', f'量子专题{i}', ['量子内容'])
        out = search_papers(SearchPapersInput(query='量子', top_k=2), cfg=self.cfg)
        self.assertEqual(out.count('paper_id='), 2)

    def test_body_hit_without_highlight_is_not_title_only(self):
        self.add_doc('lig', '连字论文', ['ﬁne-tuning methods'])
        out = search_papers(SearchPapersInput(query='fine'), cfg=self.cfg)
        self.assertIn('body matched', out, 'body hit must be reported as such')
        self.assertNotIn('title matched', out, 'must not claim a title-only hit')
        self.assertIn('read_paper_page', out)


class TestReadPageTool(ToolsTestBase):
    def test_read_page_returns_original_text(self):
        self.add_doc('q', '量子计算导论', ['第一页原文', '第二页原文'])
        out = read_paper_page(ReadPaperPageInput(paper_id=1, page_no=2), cfg=self.cfg)
        self.assertIn('《量子计算导论》 papers/q p.2', out)
        self.assertIn('第二页原文', out)

    def test_invalid_page_number(self):
        self.add_doc('q', '量子计算导论', ['只有一页'])
        out = read_paper_page(ReadPaperPageInput(paper_id=1, page_no=9), cfg=self.cfg)
        self.assertIn('[not found]', out)
        self.assertIn('page_count=1', out)

    def test_invalid_paper_id(self):
        out = read_paper_page(ReadPaperPageInput(paper_id=42, page_no=1), cfg=self.cfg)
        self.assertIn('[not found]: no paper with paper_id=42', out)

    def test_read_page_returns_raw_compat_text(self):
        raw = '本⽂调研⼤语⾔模型的发展'
        self.add_doc('c', '兼容汉字论文', [raw])
        out = read_paper_page(ReadPaperPageInput(paper_id=1, page_no=1), cfg=self.cfg)
        self.assertIn(raw, out, 'must return the raw body, not the normalized copy')
        self.assertIn('⼤', out)
        self.assertNotIn('本文调研大语言模型的发展', out)


class TestPathSafety(ToolsTestBase):
    def test_outside_directory_rejected(self):
        outside = self.tmp / 'outside'
        outside.mkdir()
        (outside / 'x.pdf').write_bytes(make_pdf(['x']))
        with self.assertRaises(PermissionError):
            import_papers(ImportPapersInput(paper_dir=str(outside)), cfg=self.cfg)

    def test_sensitive_directory_rejected(self):
        secret = self.ws / '.env'
        secret.mkdir()
        (secret / 'x.pdf').write_bytes(make_pdf(['x']))
        with self.assertRaises(PermissionError):
            import_papers(ImportPapersInput(paper_dir='.env'), cfg=self.cfg)

    def test_symlink_escape_rejected(self):
        outside = self.tmp / 'outside'
        outside.mkdir()
        (outside / 'x.pdf').write_bytes(make_pdf(['x']))
        link = self.ws / 'linked'
        link.symlink_to(outside)
        with self.assertRaises(PermissionError):
            import_papers(ImportPapersInput(paper_dir='linked'), cfg=self.cfg)

    def test_symlinked_file_escape_reported_as_failed(self):
        outside = self.tmp / 'outside'
        outside.mkdir()
        evil = outside / 'evil.pdf'
        evil.write_bytes(make_pdf(['x']))
        link = self.papers_dir / 'evil.pdf'
        link.symlink_to(evil)
        self.write_pdf('good.pdf', ['ok'])
        out = import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        self.assertIn('failed: evil.pdf:', out)
        self.assertIn('1 imported', out)


class TestCfgInjectionAndIsolation(ToolsTestBase):
    def test_execute_tool_passes_custom_cfg_end_to_end(self):
        self.write_pdf('a.pdf', ['end to end'])
        executer = ToolExecution(
            {t.name: t for t in PAPER_TOOLS}, lambda call, cfg=None: True, cfg=self.cfg
        )
        res = executer.execute_tool(
            make_call('import_papers', '{"paper_dir": "papers"}'), cfg=self.cfg
        )
        self.assertTrue(res.ok, res.content)
        self.assertIn('imported:', res.content)
        self.assertTrue(default_db_path(self.cfg).is_file())

    def test_execute_tool_without_cfg_falls_back_to_global(self):
        def spy(inp, cfg=CONFIG):
            return repr(cfg.work_space)

        tool = ToolDefinition('cfg_spy', 'probe', ImportPapersInput, spy, False)
        executer = ToolExecution({'cfg_spy': tool}, lambda call, cfg=None: True)
        res = executer.execute_tool(make_call('cfg_spy', '{"paper_dir": "papers"}'))
        self.assertIn(str(CONFIG.work_space), res.content)

    def test_db_path_not_controllable_by_model(self):
        self.write_pdf('a.pdf', ['x'])
        import_papers(ImportPapersInput(paper_dir='papers'), cfg=self.cfg)
        self.assertTrue(default_db_path(self.cfg).is_file())
        self.assertFalse((self.tmp / 'evil.db').exists())

    def test_paper_agent_registry_and_session_are_isolated(self):
        from mini_harness.agent import DeepSeekAgent
        from paperpilot.cli import build_paper_config

        paper_cfg = build_paper_config(self.cfg)
        self.assertTrue(paper_cfg.session_path.endswith('sandbox/paperpilot/session.json'))
        self.assertNotEqual(paper_cfg.session_path, str(self.ws / 'session.json'))
        self.assertNotEqual(paper_cfg.system_prompt, CONFIG.system_prompt)
        self.assertEqual(paper_cfg.model_main, self.cfg.model_main)
        self.assertEqual(paper_cfg.base_url, self.cfg.base_url)

        agent = DeepSeekAgent(PAPER_TOOLS, cfg=paper_cfg)
        self.assertEqual(set(agent.regis), {t.name for t in PAPER_TOOLS})
        self.assertNotIn('read_file', agent.regis)
        self.assertTrue(str(agent.session_memory).endswith('sandbox/paperpilot/session.json'))
        default_agent = DeepSeekAgent(TOOLS, cfg=self.cfg)
        self.assertIn('read_file', default_agent.regis)
        self.assertNotEqual(str(default_agent.session_memory), str(agent.session_memory))


if __name__ == '__main__':
    unittest.main()
