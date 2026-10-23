"""Tests for paperpilot.search: ranking, evidence snippets, query safety."""

import logging
import tempfile
import unittest

from dataclasses import replace
from pathlib import Path

from mini_harness.config import CONFIG

from paperpilot.pdf_import import PaperDocument, PageText
from paperpilot.search import (
    BODY_WEIGHT,
    MIN_LOOSE_TERM_HITS,
    TITLE_WEIGHT,
    TERM_ALIASES,
    build_loose_match,
    build_match_query,
    build_snippet,
    concept_coverage,
    concept_groups,
    discriminative_terms,
    evidence_terms,
    inflection_family,
    min_term_hits,
    search,
    term_coverage,
)
from paperpilot.store import PaperStore


def make_doc(title, pages, sha='sha', source_path='papers/x.pdf'):
    return PaperDocument(
        title=title,
        source_path=source_path,
        sha256=sha,
        pages=tuple(PageText(page_no=i + 1, text=t) for i, t in enumerate(pages)),
    )


class SearchTestBase(unittest.TestCase):
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

        self._add('quantum.pdf', '量子计算导论', ['本文研究量子计算的基本原理。'])
        self._add('image.pdf', '图像处理', ['量子在图像处理中的应用较少。'])
        self._add('db.pdf', '数据库系统', ['完全无关的文本内容。'])
        self._add('empty_body.pdf', '量子专题', [''])
        self._add('mixed.pdf', '混合文本', ['GPT-4模型与fts5检索。'])

    def _add(self, name, title, pages):
        # 中文/混合内容走结构化导入（合成 PDF 简单字体无法承载 CJK 字面串）；
        # PDF 往返与路径策略由 store 测试覆盖。
        source_path = f'papers/{name}'
        self.store.import_document(
            make_doc(title, pages, sha=f'sha-{name}', source_path=source_path),
            source_path=source_path,
        )


class TestQueries(SearchTestBase):
    def test_chinese_two_char_query(self):
        results = search(self.store, '量子')
        self.assertEqual({r.title for r in results}, {'量子计算导论', '图像处理', '量子专题'})

    def test_chinese_single_char_query(self):
        results = search(self.store, '量')
        self.assertTrue(results)
        self.assertEqual({r.title for r in results}, {'量子计算导论', '图像处理', '量子专题'})

    def test_chinese_multi_char_query(self):
        results = search(self.store, '量子计算')
        self.assertEqual([r.title for r in results], ['量子计算导论'])

    def test_english_query(self):
        results = search(self.store, 'fts5')
        self.assertEqual([r.title for r in results], ['混合文本'])

    def test_mixed_text_query(self):
        results = search(self.store, 'GPT-4')
        self.assertEqual([r.title for r in results], ['混合文本'])

    def test_no_results_query(self):
        self.assertEqual(search(self.store, '不存在的词汇xyz'), [])

    def test_empty_query_returns_empty(self):
        self.assertEqual(search(self.store, ''), [])

    def test_punctuation_only_query_returns_empty(self):
        self.assertEqual(search(self.store, '，。！？ *** ---'), [])

    def test_top_k_limits_results(self):
        results = search(self.store, '量子', top_k=2)
        self.assertEqual(len(results), 2)


class TestMaliciousQueries(SearchTestBase):
    def test_fts_operators_are_neutralized(self):
        for bad in (
            '"',
            '""""',
            'NEAR(a b, 3)',
            'a OR b',
            'title:量子',
            '量子*',
            '^量子',
            'foo" OR "bar',
            'x OR (y NEAR z)',
            '*',
            '()',
        ):
            with self.subTest(query=bad):
                results = search(self.store, bad)  # must never raise
                self.assertIsInstance(results, list)

    def test_injection_cannot_widen_results(self):
        harmless = search(self.store, '量子')
        injected = search(self.store, '量子" OR "图像')
        self.assertLessEqual({r.title for r in injected}, {r.title for r in harmless} | {'图像处理'})
        self.assertNotIn('数据库系统', {r.title for r in injected})

    def test_match_query_is_fully_quoted(self):
        expr = build_match_query('foo bar')
        self.assertEqual(expr, '"foo" "bar"')
        self.assertEqual(build_match_query('，。'), '')
        for token_expr in build_match_query('a* NEAR').split(' '):
            self.assertTrue(token_expr.startswith('"') and token_expr.endswith('"'))


class TestRanking(SearchTestBase):
    def test_title_weight_beats_body_match(self):
        results = search(self.store, '量子')
        titles = [r.title for r in results]
        self.assertIn('图像处理', titles)
        # 标题权重 8.0 > 正文权重 1.0：标题命中必须全部排在仅正文命中之前
        self.assertLess(titles.index('量子专题'), titles.index('图像处理'))
        self.assertLess(titles.index('量子计算导论'), titles.index('图像处理'))
        # 两个标题命中之间的次序由 BM25 长度归一化决定（短标题更密），
        # 这里只锁定 bm25 距离语义：分数越小越相关
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores))

    def test_weights_are_explicit(self):
        self.assertEqual(TITLE_WEIGHT, 8.0)
        self.assertEqual(BODY_WEIGHT, 1.0)
        self.assertGreater(TITLE_WEIGHT, BODY_WEIGHT)

    def test_result_fields_are_complete(self):
        results = search(self.store, '量子计算')
        row = results[0]
        self.assertIsInstance(row.paper_id, int)
        self.assertEqual(row.title, '量子计算导论')
        self.assertEqual(row.page_no, 1)
        self.assertIsInstance(row.score, float)


class TestEvidenceSnippets(SearchTestBase):
    def test_snippet_is_original_text_not_segmented(self):
        results = search(self.store, '量子计算')
        snippet = results[0].snippet
        self.assertIn('量子计算', snippet, 'evidence must contain the raw run')
        self.assertNotIn('量 子', snippet, 'segmented text must not appear as evidence')
        self.assertIn('[量子计算]', snippet)

    def test_snippet_marks_real_positions_only(self):
        snippet = build_snippet('前文研究量子计算方法后文', evidence_terms('量子计算'))
        self.assertIn('研究[量子计算]方法', snippet)
        self.assertEqual(build_snippet('没有命中内容', evidence_terms('量子')), '')

    def test_title_only_hit_has_empty_snippet(self):
        results = search(self.store, '量子专题')
        self.assertEqual(results[0].title, '量子专题')
        self.assertEqual(results[0].snippet, '', 'no body match -> no fabricated snippet')

    def test_snippet_preserves_original_characters(self):
        body = 'GPT-4模型与fts5检索。'
        snippet = build_snippet(body, evidence_terms('fts5'))
        self.assertIn('[fts5]', snippet)
        self.assertIn('GPT-4模型', snippet.replace('…', ''))

    def test_evidence_terms_are_original_substrings(self):
        terms = evidence_terms('量子计算')
        self.assertIn('量子', terms)
        self.assertIn('量子计算', terms)
        self.assertTrue(all(' ' not in t for t in terms))

    def test_page_id_maps_back_to_original_page(self):
        results = search(self.store, '量子计算')
        page = self.store.page_by_id(results[0].page_id)
        self.assertEqual(page['body'], '本文研究量子计算的基本原理。')
        self.assertEqual(page['page_no'], results[0].page_no)


class TestCompatHanEvidence(unittest.TestCase):
    """Compatibility characters in raw bodies: real hits, real highlights."""

    COMPAT_BODY = '本⽂调研⼤语⾔模型的发展'

    def setUp(self):
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = PaperStore(self.tmp / 'index.db')
        self.addCleanup(self.store.close)
        self.store.import_document(
            make_doc('兼容汉字论文', [self.COMPAT_BODY], sha='compat',
                     source_path='papers/c.pdf'),
            source_path='papers/c.pdf',
        )
        self.store.import_document(
            make_doc('连字论文', ['ﬁne-tuning methods'], sha='lig',
                     source_path='papers/f.pdf'),
            source_path='papers/f.pdf',
        )

    def test_normal_query_hits_compat_index_text(self):
        results = search(self.store, '大语言模型')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, '兼容汉字论文')
        self.assertEqual(results[0].page_no, 1)

    def test_snippet_marks_real_spans_in_raw_compat_body(self):
        result = search(self.store, '大语言模型')[0]
        self.assertIn('本⽂调研[⼤语⾔模型]', result.snippet, 'highlight must cover the raw span')
        self.assertIn('⼤', result.snippet, 'raw compatibility characters must be preserved')
        self.assertEqual(result.status, 'highlighted')
        page = self.store.page_by_id(result.page_id)
        self.assertEqual(page['body'], self.COMPAT_BODY, 'pages.body stays raw')

    def test_length_changing_normalization_never_fakes_highlights(self):
        results = search(self.store, 'fine')
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.title, '连字论文')
        self.assertEqual(result.snippet, '', 'no proven positions -> no highlight')
        self.assertEqual(result.status, 'body_matched_no_highlight')

    def test_evidence_terms_normalized(self):
        self.assertEqual(evidence_terms('⼤语⾔模型'), evidence_terms('大语言模型'))
        self.assertIn('大语言模型', evidence_terms('⼤语⾔模型'))
        self.assertNotIn('⼤', evidence_terms('⼤语⾔模型'))


class TestLooseTerms(unittest.TestCase):
    """Loose-pass vocabulary: distinctive only, never bare CJK unigrams."""

    def test_cjk_run_yields_bigrams_not_unigrams(self):
        terms = discriminative_terms('统计语言模型')
        self.assertEqual(
            terms,
            ['统计', '计语', '语言', '言模', '模型', 'SLM', 'SLMs'],
        )
        self.assertNotIn('统', terms)
        self.assertNotIn('语', terms)

    def test_latin_words_are_whole_runs_with_inflections(self):
        self.assertEqual(
            discriminative_terms('low-rank weight updates'),
            ['low', 'lows', 'rank', 'ranks', 'weight', 'weights', 'updates', 'update'],
        )

    def test_single_char_cjk_query_keeps_its_unigram(self):
        self.assertEqual(discriminative_terms('量'), ['量'])

    def test_duplicates_removed_order_stable(self):
        self.assertEqual(discriminative_terms('语言模型 语言'), ['语言', '言模', '模型'])
        self.assertEqual(
            discriminative_terms('大语言模型 大语'),
            ['大语', '语言', '言模', '模型', 'LLM', 'LLMs'],
        )

    def test_threshold_is_two_for_multi_term_queries(self):
        self.assertEqual(MIN_LOOSE_TERM_HITS, 2)
        self.assertEqual(min_term_hits(0), 0)
        self.assertEqual(min_term_hits(1), 1)
        for count in (2, 3, 5, 8):
            self.assertEqual(min_term_hits(count), 2)

    def test_coverage_is_token_exact_not_substring(self):
        self.assertEqual(term_coverage({'weights'}, ['weight']), 0)
        self.assertEqual(term_coverage({'weight'}, ['weights']), 0)
        self.assertEqual(term_coverage({'weight', 'weights'}, ['weight', 'weights']), 2)
        # casefolded membership, same as FTS5 unicode61 folding
        self.assertEqual(term_coverage({'transformer'}, ['Transformer']), 1)

    def test_loose_match_quotes_every_term(self):
        self.assertEqual(build_loose_match(['alpha', 'beta']), '"alpha" OR "beta"')
        self.assertEqual(build_loose_match([]), '')
        for piece in build_loose_match(['a* NEAR', '"x"']).split(' OR '):
            self.assertTrue(piece.startswith('"') and piece.endswith('"'))


class TestInflectionFamily(unittest.TestCase):
    """Controlled English word-form variants; no broad suffix truncation."""

    def test_plural_s_is_bilateral(self):
        self.assertEqual(inflection_family('updates'), ('updates', 'update'))
        self.assertEqual(inflection_family('update'), ('update', 'updates'))
        self.assertEqual(inflection_family('weights'), ('weights', 'weight'))

    def test_abbreviations_keep_their_case_when_pluralised(self):
        self.assertEqual(inflection_family('NLM'), ('NLM', 'NLMs'))
        self.assertEqual(inflection_family('SLM'), ('SLM', 'SLMs'))

    def test_ss_final_words_get_no_bogus_stem(self):
        self.assertEqual(inflection_family('class'), ('class',))
        self.assertNotIn('clas', inflection_family('class'))

    def test_irregular_freeze_family_is_whole_group(self):
        expected = {'freeze', 'freezes', 'froze', 'frozen', 'freezing'}
        for form in expected:
            self.assertEqual(set(inflection_family(form)), expected, form)

    def test_no_ed_or_ing_stripping(self):
        # '-ed'/'-ing' truncation would invent junk stems ('used' -> 'us');
        # only +/-s and the irregular table are allowed.
        self.assertEqual(inflection_family('used'), ('used', 'useds'))
        self.assertEqual(inflection_family('making'), ('making', 'makings'))

    def test_empty_and_blank_are_empty(self):
        self.assertEqual(inflection_family(''), ())
        self.assertEqual(inflection_family('   '), ())


class TestConceptGroups(unittest.TestCase):
    """One concept == one coverage unit, however many surface forms."""

    def test_cjk_bigrams_are_independent_concepts(self):
        groups = concept_groups('统计语言模型')
        self.assertEqual(groups[:5], [('统计',), ('计语',), ('语言',), ('言模',), ('模型',)])

    def test_attested_alias_adds_one_extra_concept(self):
        groups = concept_groups('神经语言模型')
        self.assertEqual(len(groups), 6, '5 bigrams + 1 alias concept')
        self.assertEqual(groups[-1], ('NLM', 'NLMs'))
        self.assertIn('神经语言模型', TERM_ALIASES)

    def test_aliases_are_only_attested_terms(self):
        # 无文档定义句的词不得进入术语表
        self.assertEqual(concept_groups('量子引力'), [('量子',), ('子引',), ('引力',)])
        for key in TERM_ALIASES:
            self.assertGreaterEqual(len(key), 2)

    def test_variants_of_one_concept_count_once(self):
        groups = [('updates', 'update'), ('weight', 'weights')]
        # 两个概念的所有变体都命中 -> 仍只计 2
        page = {'updates', 'update', 'weight', 'weights'}
        self.assertEqual(concept_coverage(page, groups), 2)
        # 只命中同一概念的两个变体 -> 只计 1
        self.assertEqual(concept_coverage({'updates', 'update'}, groups), 1)

    def test_expansion_does_not_change_threshold_semantics(self):
        """Concept count -- not the expanded term count -- drives admission.

        Replaces a tautology (plain and expanded were the same call).
        Each assertion below is sensitive to the wrong behaviour:
        counting surface forms instead of concepts would inflate
        coverage, and deriving the threshold from the flat term list
        would raise a one-concept query's bar from 1 to 2.
        """
        groups = concept_groups('frozen updates')
        flat = discriminative_terms('frozen updates')
        # 2 independent concepts, 7 surface forms after expansion
        self.assertEqual(len(groups), 2)
        self.assertGreater(len(flat), len(groups), 'variants must outnumber concepts')

        # every variant of BOTH concepts on one page -> still only 2
        self.assertEqual(concept_coverage({t.casefold() for t in flat}, groups), 2)

        # five variants of the SAME concept -> still only 1
        self.assertEqual(
            concept_coverage(
                {'freeze', 'freezes', 'froze', 'frozen', 'freezing'}, groups
            ),
            1,
        )

        # threshold comes from the concept count (2 -> MIN_LOOSE_TERM_HITS),
        # not from len(flat); a flat-count threshold would also break the
        # one-concept case below.
        self.assertEqual(min_term_hits(len(groups)), MIN_LOOSE_TERM_HITS)
        single = concept_groups('frozen')
        self.assertEqual(len(single), 1)
        self.assertGreater(len(discriminative_terms('frozen')), len(single))
        self.assertEqual(
            min_term_hits(len(single)), 1,
            'one-concept query keeps a threshold of 1 even after expansion '
            'to %d surface forms' % len(discriminative_terms('frozen')),
        )

    def test_adding_variants_does_not_add_concepts(self):
        # same two concepts, with and without inflection partners
        bare = [('updates',), ('weight',)]
        with_forms = [('updates', 'update'), ('weight', 'weights')]
        self.assertEqual(len(bare), len(with_forms), 'variant count != concept count')
        page = {'update', 'updates', 'weight', 'weights'}
        self.assertEqual(concept_coverage(page, bare), 2)
        self.assertEqual(concept_coverage(page, with_forms), 2)
        self.assertEqual(min_term_hits(len(bare)), min_term_hits(len(with_forms)))
        # one concept, both its variants present -> one hit, not two
        self.assertEqual(concept_coverage({'updates', 'update'}, with_forms), 1)

    def test_concept_coverage_requires_any_variant(self):
        groups = concept_groups('frozen')
        self.assertEqual(concept_coverage({'freezes'}, groups), 1)
        self.assertEqual(concept_coverage({'frozen'}, groups), 1)
        self.assertEqual(concept_coverage({'weight'}, groups), 0)


class TestLooseRetrieval(SearchTestBase):
    """Loose candidates rescue AND misses and compete on score, not arrival order."""

    def setUp(self):
        super().setUp()
        # 严格 AND 会因跨词边界 bigram '计语' 缺失而零命中；
        # 页面实际说的是「统计学习方法的语言模型」。
        self._add('slm.pdf', '统计方法', ['本文采用基于统计学习方法的语言模型建模。'])
        self._add('slm_exact.pdf', '语言模型', ['统计语言模型是常见基线模型。'])
        self._add('noise.pdf', '噪声页', ['这里只提到语言两个字而已。'])
        self._add('one_word.pdf', '单词页', ['quantum appears once here.'])

    def test_loose_rescues_and_zero_hit_query(self):
        results = search(self.store, '统计语言模型')
        titles = [r.title for r in results]
        self.assertIn('统计方法', titles, 'wording variant must be recalled')
        self.assertEqual(
            search(self.store, '统计语言模型', top_k=1)[0].title, '语言模型',
            'exact phrase page still ranks first',
        )

    def test_loose_fires_even_when_strict_already_hits(self):
        # 严格路径已能命中「语言模型」页，但不能因此放弃「统计方法」页
        strict_only = build_match_query('统计语言模型')
        self.assertTrue(strict_only)
        titles = [r.title for r in search(self.store, '统计语言模型')]
        self.assertIn('语言模型', titles)
        self.assertIn('统计方法', titles)

    def test_partial_coverage_below_threshold_is_refused(self):
        # 「噪声页」只覆盖 bigram「语言」= 1 < 2
        results = search(self.store, '统计语言模型')
        self.assertNotIn('噪声页', [r.title for r in results])

    def test_loose_page_can_enter_a_full_top_k(self):
        # 严格结果已填满 top_k=1，宽松页仍不能被钉死在末尾之外
        results = search(self.store, '统计语言模型', top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, '语言模型')
        # top_k=2 时第二席由分数决定，「统计方法」必须在候选池中参与竞争
        two = search(self.store, '统计语言模型', top_k=2)
        self.assertEqual([r.title for r in two], ['语言模型', '统计方法'])

    def test_negative_multi_term_query_stays_empty(self):
        results = search(self.store, 'quantum computing applications')
        self.assertEqual(results, [], 'coverage 1/3 must not pass min_hits=2')

    def test_single_term_query_still_retrieves(self):
        results = search(self.store, 'quantum')
        self.assertEqual([r.title for r in results], ['单词页'])

    def test_merged_results_keep_evidence_status(self):
        rescued = [r for r in search(self.store, '统计语言模型') if r.title == '统计方法'][0]
        self.assertEqual(rescued.status, 'highlighted')
        self.assertIn('[统计', rescued.snippet.replace('[统计学习', '[统计'))
        self.assertTrue(rescued.snippet, 'rescued page must still carry 原文 evidence')

    def test_scores_are_sorted_ascending(self):
        scores = [r.score for r in search(self.store, '统计语言模型')]
        self.assertEqual(scores, sorted(scores))


class TestQueryTimeExpansion(SearchTestBase):
    """Inflection + attested aliases widen recall without faking evidence."""

    def setUp(self):
        super().setUp()
        self._add('sgd.pdf', '梯度更新', ['we apply the gradient update to weight matrices.'])
        self._add('freeze.pdf', '参数冻结', ['this freezes the pre-trained weights.'])
        self._add('nlm.pdf', '神经模型史', ['NLMs learn task-independent representations.'])
        self._add('unrelated.pdf', '无关页', ['quantum computing applications only.'])

    def test_plural_query_reaches_singular_page(self):
        titles = [r.title for r in search(self.store, 'gradient update')]
        self.assertIn('梯度更新', titles)
        titles = [r.title for r in search(self.store, 'gradient updates')]
        self.assertIn('梯度更新', titles, 'plural query must reach singular text')

    def test_irregular_form_query_reaches_freezes_page(self):
        for q in ('pre-trained weights frozen', 'weights freezes'):
            with self.subTest(query=q):
                self.assertIn('参数冻结', [r.title for r in search(self.store, q)])

    def test_abbreviation_page_reachable_via_attested_alias(self):
        # 查询用中文全称；页面像 AI报告 p.4 那样以缩写为主、少量中文上下文
        self._add('nlm_ctx.pdf', '缩写页', ['NLMs learn representations of 语言模型 history.'])
        titles = [r.title for r in search(self.store, '神经语言模型')]
        self.assertIn('缩写页', titles, 'attested alias NLM must admit the page')
        # 纯缩写页只覆盖 1 个概念，仍应被 min_hits=2 拒绝（精度底线）
        self.assertNotIn('神经模型史', titles)

    def test_pure_abbreviation_page_still_needs_two_concepts(self):
        results = search(self.store, '神经语言模型')
        self.assertNotIn('神经模型史', [r.title for r in results],
                         'alias-only coverage = 1 concept must not pass alone')

    def test_expansion_does_not_fake_body_evidence(self):
        # 扩展词只负责召回；evidence_terms 仍用原始查询串
        self.assertNotIn('NLMs', evidence_terms('神经语言模型'))
        self.assertNotIn('update', evidence_terms('weight'))
        hit = [r for r in search(self.store, 'gradient updates') if r.title == '梯度更新'][0]
        # 页面含 'update'，但查询词是 'updates' -> 不得高亮成 'updates'
        self.assertNotIn('[updates]', hit.snippet)

    def test_negative_query_still_refused_despite_expansion(self):
        # 语料中不存在的主题：扩展不得把无关内容变成命中
        self.assertEqual(search(self.store, 'dark matter black hole'), [])
        self.assertEqual(search(self.store, 'quantum gravity singularity'), [])

    def test_expansion_cannot_widen_injection(self):
        harmless = {r.title for r in search(self.store, 'gradient update')}
        injected = {r.title for r in search(self.store, 'gradient update" OR "quantum')}
        self.assertLessEqual(injected, harmless | {'无关页'})
        self.assertNotIn('无关页', injected)


class TestAdmissionCountsConcepts(SearchTestBase):
    """Candidate admission: one hit per concept, however many word forms."""

    QUERY = 'frozen updates'   # 2 concepts: freeze-family + updates/update

    def setUp(self):
        super().setUp()
        # 反例：同一概念(freeze 族)的两个变体，缺第二概念
        self._add('one_concept.pdf', '单概念页', [
            'the freezes were frozen again yesterday.',
        ])
        # 正例：两个概念各命中一个变体
        self._add('two_concepts.pdf', '双概念页', [
            'this freezes the gradient update during training.',
        ])

    def test_two_forms_of_one_concept_do_not_pass_admission(self):
        # 若按词项计数，'freezes'+'frozen' = 2 会误准入；
        # 按概念计数 freeze 族 = 1 < 2，必须拒绝
        titles = [r.title for r in search(self.store, self.QUERY)]
        self.assertNotIn('单概念页', titles,
                         'two word forms of ONE concept must not pass min_hits=2')

    def test_one_form_per_concept_passes_admission(self):
        titles = [r.title for r in search(self.store, self.QUERY)]
        self.assertIn('双概念页', titles,
                      'one form in each of two concepts must pass min_hits=2')

    def test_single_concept_query_still_admits_one_form(self):
        # 单概念查询阈值为 1：只命中 freezes 也必须准入
        self._add('solo.pdf', '孤立页', ['the sample freezes during storage.'])
        titles = [r.title for r in search(self.store, 'frozen')]
        self.assertIn('孤立页', titles,
                      'one concept with one variant must pass its own threshold')


class TestStatusAfterExpansion(SearchTestBase):
    """Status must describe the body, never the pass that recalled the page."""

    def setUp(self):
        super().setUp()
        # 正文只含扩展变体（freeszes / update），原始查询词 frozen / updates 均无子串
        self._add('via_forms.pdf', '词形页', [
            'this freezes the gradient update during training.',
        ])
        # 真正的标题命中：正文与查询无关
        self._add('title_hit.pdf', 'frozen updates', [
            'entirely unrelated body text about gardening.',
        ])
        # 原词与变体都在：严格路径与扩展路径同时召回
        self._add('both_paths.pdf', '双路径页', [
            'the frozen updates and freezes of updates are recorded.',
        ])

    def test_inflection_body_hit_is_not_title_only(self):
        hits = {r.title: r for r in search(self.store, 'frozen updates')}
        self.assertIn('词形页', hits, 'expanded forms must recall the page')
        self.assertEqual(hits['词形页'].status, 'body_matched_no_highlight')
        self.assertEqual(hits['词形页'].snippet, '', 'no original term -> no highlight')

    def test_real_title_hit_stays_title_only(self):
        hits = {r.title: r for r in search(self.store, 'frozen updates')}
        self.assertIn('frozen updates', hits, 'title still recalls')
        self.assertEqual(hits['frozen updates'].status, 'title_only')
        self.assertEqual(hits['frozen updates'].snippet, '')

    def test_status_decided_by_body_coverage_not_by_pass(self):
        hits = {r.title: r for r in search(self.store, 'frozen updates')}
        # 同页同时被严格与扩展召回时，原文可定位 -> 仍是 highlighted
        self.assertEqual(hits['双路径页'].status, 'highlighted')
        self.assertIn('[frozen]', hits['双路径页'].snippet)
        self.assertIn('[updates]', hits['双路径页'].snippet)
        # 仅扩展召回的页走 body_matched_no_highlight，二者互不污染
        self.assertEqual(hits['词形页'].status, 'body_matched_no_highlight')
        self.assertEqual(hits['frozen updates'].status, 'title_only')

    def test_coverage_threshold_not_relaxed_for_status(self):
        # 正文只覆盖 1 个概念（freezes），不到 min_hits=2：
        # 页面靠标题进了结果，正文覆盖不足 -> 必须保持 title_only
        self._add('weak.pdf', 'frozen updates weak', [
            'this freezes nothing else of interest.',
        ])
        hits = {r.title: r for r in search(self.store, 'frozen updates')}
        self.assertIn('frozen updates weak', hits)
        self.assertEqual(hits['frozen updates weak'].status, 'title_only')

    def test_highlights_are_always_real_raw_substrings(self):
        import re
        for r in search(self.store, 'frozen updates'):
            body = self.store.page_by_id(r.page_id)['body']
            for m in re.finditer(r'\[(.*?)\]', r.snippet):
                self.assertIn(m.group(1), body, 'fabricated highlight')

    def test_nfkc_no_highlight_path_does_not_regress(self):
        # 既有行为：长度变化归一导致偏移不可证时仍是 body_matched_no_highlight
        self._add('lig.pdf', '连字', ['ﬁne-tuning of the model'])
        hits = {r.title: r for r in search(self.store, 'fine')}
        self.assertEqual(hits['连字'].status, 'body_matched_no_highlight')
        self.assertEqual(hits['连字'].snippet, '')


if __name__ == '__main__':
    unittest.main()
