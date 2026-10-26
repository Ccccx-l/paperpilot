// PaperPilot 研究工作台 — 正式前端
// 全部数据来自后端 JSON API；无合成数据，无预设回答。
// 所有动态文本通过 textContent / setAttribute 写入，不拼接 innerHTML。
(function () {
  'use strict';

  var state = {
    papers: [],
    paperId: null,
    pageNo: null,
    pageCount: null,
    busy: false
  };

  var $ = function (id) { return document.getElementById(id); };
  var chatScroll = $('chat-scroll');
  var chatEmpty = $('chat-empty');
  var paperList = $('paper-list');
  var metaBody = $('meta-body');
  var readerBody = $('reader-body');
  var readerPlaceholder = $('reader-placeholder');
  var pageText = $('page-text');
  var readerPos = $('reader-pos');
  var pageInput = $('page-input');
  var originAnchor = $('origin-anchor');
  var originBadge = $('origin-badge');
  var originSrc = $('origin-src');
  var pipeMsg = $('pipe-msg');
  var askInput = $('ask-input');
  var toast = $('toast');

  // ── helpers ──────────────────────────────────────────────

  function el(tag, className, text) {
    var n = document.createElement(tag);
    if (className) n.className = className;
    if (text != null) n.textContent = text;
    return n;
  }

  // ── Markdown rendering (safe DOM construction, never innerHTML) ──────
  // Renders a bounded Markdown subset: headings, bold/italic, inline code,
  // code blocks, ordered/unordered lists, blockquotes, horizontal rules.
  // All text goes through textContent, so no HTML/script can execute.

  function renderInline(parent, text) {
    // Tokenise inline markers; every literal run goes through textContent.
    var s = String(text || '');
    var i = 0;
    while (i < s.length) {
      // inline code `...`
      if (s[i] === '`') {
        var tickEnd = s.indexOf('`', i + 1);
        if (tickEnd > i + 1) {
          parent.appendChild(el('code', 'md-code', s.slice(i + 1, tickEnd)));
          i = tickEnd + 1;
          continue;
        }
      }
      // bold **...**
      if (s[i] === '*' && s[i + 1] === '*') {
        var boldEnd = s.indexOf('**', i + 2);
        if (boldEnd > i + 2) {
          var b = el('strong', 'md-bold');
          renderInline(b, s.slice(i + 2, boldEnd));
          parent.appendChild(b);
          i = boldEnd + 2;
          continue;
        }
      }
      // italic *...*
      if (s[i] === '*') {
        var italEnd = s.indexOf('*', i + 1);
        if (italEnd > i + 1) {
          var em = el('em', 'md-italic');
          renderInline(em, s.slice(i + 1, italEnd));
          parent.appendChild(em);
          i = italEnd + 1;
          continue;
        }
      }
      // literal run up to next marker
      var next = s.length;
      for (var j = i + 1; j < s.length; j++) {
        if (s[j] === '`' || s[j] === '*') { next = j; break; }
      }
      parent.appendChild(document.createTextNode(s.slice(i, next)));
      i = next;
    }
  }

  function renderMarkdown(text) {
    var frag = document.createDocumentFragment();
    var lines = String(text || '').split(/\r?\n/);
    var i = 0;
    while (i < lines.length) {
      var line = lines[i];
      // fenced code block
      if (/^```/.test(line)) {
        var codeLines = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i])) { codeLines.push(lines[i]); i++; }
        i++; // skip closing fence
        frag.appendChild(el('pre', 'md-pre', codeLines.join('\n')));
        continue;
      }
      // horizontal rule
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        frag.appendChild(el('hr', 'md-hr'));
        i++;
        continue;
      }
      // heading # .. ######
      var h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        var hn = h[1].length;
        var hv = el('h' + hn, 'md-h' + hn);
        renderInline(hv, h[2]);
        frag.appendChild(hv);
        i++;
        continue;
      }
      // blockquote
      if (/^\s*>/.test(line)) {
        var bqLines = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) {
          bqLines.push(lines[i].replace(/^\s*>\s?/, ''));
          i++;
        }
        var bq = el('blockquote', 'md-blockquote');
        bqLines.forEach(function (bl) {
          var p = el('p', 'md-p');
          renderInline(p, bl);
          bq.appendChild(p);
        });
        frag.appendChild(bq);
        continue;
      }
      // unordered list
      if (/^\s*[-*+]\s+/.test(line)) {
        var ul = el('ul', 'md-ul');
        while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
          var li = el('li', 'md-li');
          renderInline(li, lines[i].replace(/^\s*[-*+]\s+/, ''));
          ul.appendChild(li);
          i++;
        }
        frag.appendChild(ul);
        continue;
      }
      // ordered list
      if (/^\s*\d+[.)]\s+/.test(line)) {
        var ol = el('ol', 'md-ol');
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          var oli = el('li', 'md-li');
          renderInline(oli, lines[i].replace(/^\s*\d+[.)]\s+/, ''));
          ol.appendChild(oli);
          i++;
        }
        frag.appendChild(ol);
        continue;
      }
      // paragraph
      if (line.trim() === '') { i++; continue; }
      var para = el('p', 'md-p');
      renderInline(para, line);
      frag.appendChild(para);
      i++;
    }
    return frag;
  }

  function mdBubble(text) {
    var bubble = el('div', 'msg-bubble answer-text md');
    bubble.appendChild(renderMarkdown(text));
    return bubble;
  }

  function paperById(id) {
    for (var i = 0; i < state.papers.length; i++) {
      if (state.papers[i].paper_id === id) return state.papers[i];
    }
    return null;
  }

  function showToast(message, kind) {
    toast.textContent = message;
    toast.className = 'toast' + (kind ? ' ' + kind : '');
    toast.hidden = false;
    clearTimeout(showToast._t);
    showToast._t = setTimeout(function () { toast.hidden = true; }, 4000);
  }

  function setSvc(mode, text) {
    $('svc-status').setAttribute('data-svc', mode);
    $('svc-text').textContent = text;
  }

  function setPipeMsg(t) { pipeMsg.textContent = t; }

  function api(path, options) {
    return fetch(path, options).then(function (resp) {
      return resp.text().then(function (raw) {
        var data = null;
        try { data = raw ? JSON.parse(raw) : null; } catch (e) { data = null; }
        return { ok: resp.ok, status: resp.status, data: data, raw: raw };
      });
    });
  }

  function apiError(result, fallback) {
    if (result.data && result.data.error) return result.data.error;
    return fallback || ('请求失败 (HTTP ' + result.status + ')');
  }

  // ── 论文列表 / 元数据 ────────────────────────────────────

  function renderPaperList() {
    paperList.textContent = '';
    if (!state.papers.length) {
      var empty = el('li', 'reader-placeholder', '索引为空，请先导入论文');
      paperList.appendChild(empty);
      return;
    }
    state.papers.forEach(function (p) {
      var li = el('li', 'paper-item');
      li.setAttribute('role', 'option');
      li.setAttribute('tabindex', '0');
      li.setAttribute('data-paper', String(p.paper_id));
      li.setAttribute('aria-selected', p.paper_id === state.paperId ? 'true' : 'false');
      li.appendChild(el('div', 'paper-item-title', p.title));
      li.appendChild(el('div', 'paper-item-sub',
        'paper_id=' + p.paper_id + ' · ' + p.page_count + ' 页 · ' + p.source_path));
      paperList.appendChild(li);
    });
  }

  function renderMeta() {
    metaBody.textContent = '';
    var p = paperById(state.paperId);
    if (!p) {
      metaBody.appendChild(el('div', 'reader-placeholder', '未选中论文'));
      return;
    }
    metaBody.appendChild(el('div', 'meta-title', p.title));
    var rows = [
      ['paper_id', String(p.paper_id)],
      ['来源', p.source_path],
      ['页数', String(p.page_count)],
      ['sha256', p.sha256 || '—']
    ];
    rows.forEach(function (r) {
      var row = el('div', 'meta-row');
      row.appendChild(el('span', 'k', r[0]));
      row.appendChild(el('span', 'v', r[1]));
      metaBody.appendChild(row);
    });
  }

  // ── 原文阅读（真实 /api/page） ────────────────────────────

  function renderReaderLoading() {
    readerPlaceholder.hidden = false;
    readerPlaceholder.textContent = '读取原文…';
    pageText.hidden = true;
  }

  function loadPage(paperId, pageNo, flash) {
    if (!paperId || !pageNo) return;
    renderReaderLoading();
    api('/api/page?paper_id=' + encodeURIComponent(paperId) +
        '&page_no=' + encodeURIComponent(pageNo)).then(function (result) {
      if (!result.ok || !result.data) {
        readerPlaceholder.hidden = false;
        readerPlaceholder.textContent = '读取原文失败：' + apiError(result);
        pageText.hidden = true;
        setPipeMsg('读取原文失败');
        return;
      }
      var d = result.data;
      state.paperId = d.paper_id;
      state.pageNo = d.page_no;
      state.pageCount = d.page_count;
      readerPlaceholder.hidden = true;
      pageText.hidden = false;
      pageText.textContent = d.body;
      readerPos.textContent = 'p.' + d.page_no + ' / ' + d.page_count;
      pageInput.value = String(d.page_no);
      if (flash) {
        pageText.classList.remove('flash');
        void pageText.offsetWidth;
        pageText.classList.add('flash');
        readerBody.scrollTop = 0;
      }
    });
  }

  function selectPaper(id) {
    var p = paperById(id);
    if (!p) return;
    state.paperId = id;
    renderPaperList();
    renderMeta();
    loadPage(id, 1, false);
  }

  function gotoPage(n) {
    if (!state.paperId) { showToast('请先选择论文', 'err'); return; }
    if (!Number.isFinite(n) || n < 1 || (state.pageCount && n > state.pageCount)) {
      showToast('页码无效：1 ~ ' + (state.pageCount || '?'), 'err');
      return;
    }
    loadPage(state.paperId, n, false);
  }

  function jumpToEvidence(source) {
    if (!source || source.length < 2) {
      originAnchor.hidden = false;
      originBadge.textContent = '无确定来源';
      originSrc.textContent = '未匹配到已读取页面';
      setPipeMsg('该引文未匹配到来源页面，无法跳转');
      return;
    }
    var paperId = source[0], pageNo = source[1];
    var p = paperById(paperId);
    state.paperId = paperId;
    renderPaperList();
    renderMeta();
    loadPage(paperId, pageNo, true);
    originAnchor.hidden = false;
    originBadge.textContent = '页面区域';
    originSrc.textContent = 'paper_id=' + paperId + ' · p.' + pageNo +
      (p ? ' · ' + p.title : '');
    setPipeMsg('已定位到页面区域 paper_id=' + paperId + ' p.' + pageNo);
  }

  // ── 流水线（真实执行结果） ────────────────────────────────

  function resetPipeline() {
    Array.prototype.forEach.call(document.querySelectorAll('.pipe-step'), function (el2) {
      el2.setAttribute('data-state', 'idle');
      var d = el2.querySelector('[data-role="detail"]');
      if (d) d.textContent = '—';
    });
    setPipeMsg('等待任务…');
  }

  function renderPipeline(pipeline) {
    if (!pipeline) return;
    pipeline.forEach(function (step) {
      var node = document.querySelector('.pipe-step[data-step="' + step.key + '"]');
      if (!node) return;
      node.setAttribute('data-state', step.state || 'idle');
      var d = node.querySelector('[data-role="detail"]');
      if (d) d.textContent = step.detail || '';
    });
  }

  // ── 对话渲染（全部 textContent） ──────────────────────────

  function clearChat() {
    chatScroll.textContent = '';
  }

  function appendUser(text) {
    var wrap = el('div', 'msg msg-user');
    wrap.appendChild(el('div', 'msg-tag', '用户'));
    wrap.appendChild(el('div', 'msg-bubble', text));
    chatScroll.appendChild(wrap);
  }

  // ---- tool trace: two layers -------------------------------------------
  // Default: short summary lines derived from REAL tool_events/pipeline.
  // Expanded: real names, arguments, call/result pairing, real status.

  function toolSummaryLines(toolEvents, pipeline) {
    var lines = [];
    var byName = {};
    (toolEvents || []).forEach(function (ev) {
      (byName[ev.name] = byName[ev.name] || []).push(ev);
    });
    (pipeline || []).forEach(function (step) {
      if (step.key === 'verify') {
        if (step.state === 'done') lines.push('引文核验：' + (step.detail || '已执行'));
        else if (step.state === 'fail') lines.push('引文核验：失败');
        return;
      }
      var toolName = { import: 'import_papers', search: 'search_papers',
                       read: 'read_paper_page' }[step.key];
      if (!byName[toolName]) return;  // not called this turn: no summary line
      if (step.state === 'fail') {
        lines.push(step.label + '：失败' + (step.detail ? '，' + step.detail : ''));
      } else {
        lines.push(step.label + '：已执行' + (step.detail ? '，' + step.detail : ''));
      }
    });
    return lines;
  }

  function toolDetailNode(ev) {
    var box = el('div', 'trace-item');
    var status = !ev.has_result ? '未收到结果' : (ev.ok === false ? '失败' : '成功');
    box.appendChild(el('div', 'trace-head',
      ev.name + ' · ' + status + (ev.tool_call_id ? ' · ' + ev.tool_call_id : '')));
    if (ev.arguments) box.appendChild(el('div', 'trace-args', ev.arguments));
    if (ev.result_summary) box.appendChild(el('div', 'trace-result', ev.result_summary));
    return box;
  }

  function appendAgent(item) {
    var wrap = el('div', 'msg msg-agent');
    wrap.appendChild(el('div', 'msg-tag', 'PaperPilot · 回答'));
    var body = el('div', 'msg-body');
    body.appendChild(mdBubble(item.content || ''));

    // Product requirement: the conversation area shows ONLY the answer text.
    // Evidence cards are never rendered here -- regardless of verification
    // status (verified / mismatch / unverified) or content type. The backend
    // still computes and returns evidence; see the /api/ask payload for it.

    // Tool trace: summary first, raw details collapsed by default.
    var toolEvents = item.tool_events || [];
    if (toolEvents.length) {
      var summary = el('div', 'trace-summary');
      var lines = toolSummaryLines(toolEvents, item.pipeline);
      lines.forEach(function (line) {
        summary.appendChild(el('div', 'trace-summary-line', line));
      });
      var details = el('details', 'trace-details');
      details.appendChild(el('summary', 'trace-details-toggle', '查看执行详情'));
      var inner = el('div', 'trace');
      toolEvents.forEach(function (ev) { inner.appendChild(toolDetailNode(ev)); });
      details.appendChild(inner);
      summary.appendChild(details);
      body.appendChild(summary);
    }

    wrap.appendChild(body);
    chatScroll.appendChild(wrap);
  }

  function appendError(message) {
    var wrap = el('div', 'msg msg-agent');
    wrap.appendChild(el('div', 'msg-tag', 'PaperPilot · 错误'));
    wrap.appendChild(el('div', 'err-line', message));
    chatScroll.appendChild(wrap);
  }

  function scrollChat() { chatScroll.scrollTop = chatScroll.scrollHeight; }

  function renderHistory(items) {
    clearChat();
    if (!items || !items.length) {
      chatEmpty.style.display = '';
      return;
    }
    chatEmpty.style.display = 'none';
    items.forEach(function (item) {
      if (item.role === 'user') appendUser(item.content || '');
      else if (item.role === 'assistant') appendAgent(item);
    });
    scrollChat();
  }

  // ── 提问 ─────────────────────────────────────────────────

  function ask(question) {
    var q = (question || '').trim();
    if (!q || state.busy) return;
    state.busy = true;
    $('btn-ask').disabled = true;
    chatEmpty.style.display = 'none';
    appendUser(q);
    scrollChat();
    setSvc('busy', '本地服务 · 执行中');
    resetPipeline();
    setPipeMsg('Agent 执行中…');

    api('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    }).then(function (result) {
      if (!result.ok || !result.data) {
        appendError(apiError(result, '提问失败'));
        setSvc('err', '本地服务 · 错误');
        setPipeMsg('执行失败');
        return;
      }
      var d = result.data;
      appendAgent({
        content: d.answer,
        evidence: d.evidence,
        tool_events: d.tool_events
      });
      renderPipeline(d.pipeline);
      setPipeMsg('完成 · outcome=' + d.outcome);
      setSvc('ok', '本地服务 · 就绪');
    }).catch(function (e) {
      appendError('网络错误：' + e.message);
      setSvc('err', '本地服务 · 错误');
      setPipeMsg('请求失败');
    }).then(function () {
      state.busy = false;
      $('btn-ask').disabled = false;
      scrollChat();
    });
  }

  // ── 导入 / 重置 ──────────────────────────────────────────

  function doImport() {
    var dir = $('import-dir').value.trim();
    var msg = $('import-msg');
    $('btn-import').disabled = true;
    msg.className = 'import-msg';
    msg.textContent = '导入中…';
    api('/api/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper_dir: dir })
    }).then(function (result) {
      if (!result.ok || !result.data) {
        msg.className = 'import-msg err';
        msg.textContent = apiError(result, '导入失败');
        return;
      }
      var d = result.data;
      msg.className = 'import-msg ' + (d.status === 'failed' ? 'err' : 'ok');
      msg.textContent = d.output || ('导入状态：' + d.status);
      state.papers = d.papers || [];
      $('stat-papers').textContent = state.papers.length + ' 篇';
      renderPaperList();
    }).catch(function (e) {
      msg.className = 'import-msg err';
      msg.textContent = '网络错误：' + e.message;
    }).then(function () {
      $('btn-import').disabled = false;
    });
  }

  function doReset() {
    if (!window.confirm('清空当前对话历史？论文索引不受影响。')) return;
    api('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    }).then(function (result) {
      if (!result.ok) {
        showToast(apiError(result, '重置失败'), 'err');
        return;
      }
      clearChat();
      chatEmpty.style.display = '';
      resetPipeline();
      showToast('对话已清空', 'ok');
    }).catch(function (e) {
      showToast('网络错误：' + e.message, 'err');
    });
  }

  // ── 初始化 ────────────────────────────────────────────────

  function loadState() {
    return api('/api/state').then(function (result) {
      if (!result.ok || !result.data) {
        showToast(apiError(result, '状态加载失败'), 'err');
        return;
      }
      var d = result.data;
      $('stat-workspace').textContent = d.workspace || '—';
      state.papers = d.papers || [];
      $('stat-papers').textContent = (d.paper_count || state.papers.length) + ' 篇';
      renderPaperList();
      renderMeta();
      renderHistory(d.conversation || []);
    }).catch(function (e) {
      showToast('网络错误：' + e.message, 'err');
    });
  }

  function bind() {
    $('ask-form').addEventListener('submit', function (e) {
      e.preventDefault();
      var q = askInput.value;
      ask(q);
      askInput.value = '';
    });
    $('btn-import').addEventListener('click', doImport);
    $('btn-reset').addEventListener('click', doReset);
    $('btn-scroll-latest').addEventListener('click', scrollChat);

    paperList.addEventListener('click', function (e) {
      var t = e.target.closest ? e.target.closest('[data-paper]') : null;
      if (t) selectPaper(Number(t.getAttribute('data-paper')));
    });
    paperList.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      var t = e.target.closest ? e.target.closest('[data-paper]') : null;
      if (!t) return;
      e.preventDefault();
      selectPaper(Number(t.getAttribute('data-paper')));
    });

    $('btn-prev-page').addEventListener('click', function () { gotoPage(state.pageNo - 1); });
    $('btn-next-page').addEventListener('click', function () { gotoPage(state.pageNo + 1); });
    $('btn-goto-page').addEventListener('click', function () { gotoPage(Number(pageInput.value)); });
    pageInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); gotoPage(Number(pageInput.value)); }
    });
  }

  function init() {
    resetPipeline();
    bind();
    loadState();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.PaperPilotWorkbench = {
    state: state,
    ask: ask,
    jumpToEvidence: jumpToEvidence,
    selectPaper: selectPaper,
    gotoPage: gotoPage,
    loadState: loadState
  };
})();
