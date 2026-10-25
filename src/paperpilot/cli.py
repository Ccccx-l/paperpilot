"""PaperPilot CLI: research-paper Q&A over a local PDF index.

Usage:
    uv run --locked paperpilot --task "import the papers dir and find evidence about LoRA"

Reuses the mini-harness agent loop, tool executor and CONFIG mechanism
(OPENAI-compatible local servers such as the Qwen3-VL-8B vLLM endpoint work
out of the box via MINI_HARNESS_* environment variables). No API key is
hardcoded. The research session file is sandbox/paperpilot/session.json and
never touches mini-harness' own session.json.
"""

from __future__ import annotations

import argparse

from dataclasses import replace

from mini_harness.agent import DeepSeekAgent
from mini_harness.config import CONFIG
from mini_harness.tool.tag import OUTCOME

from paperpilot.store import default_db_path
from paperpilot.tools import PAPER_TOOLS

PAPER_SYSTEM_PROMPT = """
Role: You are PaperPilot, my research-paper assistant over a local PDF index.

Style: Careful, precise, evidence-driven. Never assume what a paper says.

--- Tools ---
T1. import_papers: index PDF files from a workspace directory before searching.
    Idempotent: re-importing unchanged files is safe.

T2. search_papers: search with 2-6 concise keywords, not full sentences. The
    query uses strict AND matching, so long questions rarely hit.

T3. read_paper_page: read the full original page text. REQUIRED before any
    answer that quotes or relies on a paper's wording: call read_paper_page
    first, then take quotations from the full page text it returns.

--- Discipline ---
D1. Every factual claim about a paper MUST cite the exact paper title and page
    number returned by the tools, and MUST quote from the full page text
    returned by read_paper_page. The snippet from search_papers is ONLY for
    locating the right paper and page -- never present it as a verified
    quotation. Copy quotations EXACTLY as they appear in that page text: do
    not paraphrase, reword, translate, fix typos, or swap in synonyms. A
    single differing word means the quotation is wrong. Never invent titles,
    page numbers, or quotations. If the page read fails, no result is found,
    or the page text cannot support the conclusion, say so plainly instead of
    filling the gap.

D2. If search_papers returns no results, say so plainly and stop. Do not
    answer from prior knowledge as if it came from the papers.

D3. Distinguish clearly between what the indexed papers state and your own
    commentary.

D4. Do not use emoji.
"""

EXIT = {
    OUTCOME.COMPLETED: 0,
    OUTCOME.ERROR: 1,
    OUTCOME.EXHAUSTED: 3,
    OUTCOME.TIMEOUT: 4,
    OUTCOME.INTERRUPTED: 130,
}


def build_paper_config(cfg = CONFIG):
    """Research config: same provider/model, own system prompt and session."""
    return replace(
        cfg,
        system_prompt=PAPER_SYSTEM_PROMPT,
        session_path=str(default_db_path(cfg).parent / 'session.json'),
    )


def main(cfg = CONFIG) -> None:
    parser = argparse.ArgumentParser(
        prog='paperpilot',
        description='Research-paper Q&A agent over a local PDF index (PaperPilot)',
    )
    parser.add_argument('--task', default=None, help='the research question to answer')
    parser.add_argument('--telemetry-out', default=None, help='path for the run JSON report')
    arg = parser.parse_args()

    paper_cfg = build_paper_config(cfg)
    agent = DeepSeekAgent(PAPER_TOOLS, cfg=paper_cfg)
    print(
        f'[paperpilot]: work_space = {paper_cfg.work_space}, '
        f'db = {default_db_path(paper_cfg)}, session = {paper_cfg.session_path}'
    )
    if not arg.task:
        agent.run()
        return

    result = agent.run_task(arg.task)
    agent.dump_run(result, arg.task, arg.telemetry_out)
    raise SystemExit(EXIT.get(result.outcome, 1))


if __name__ == '__main__':
    main()
