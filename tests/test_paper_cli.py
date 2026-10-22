"""Offline tests for the paperpilot CLI entry (no model API calls)."""

import tempfile
import unittest

from dataclasses import replace
from pathlib import Path
from unittest import mock

from mini_harness.agent import Result
from mini_harness.config import CONFIG

from paperpilot import cli
from paperpilot.tools import PAPER_TOOLS

from tests._support import RootSessionGuard


class FakeAgent:
    created = []

    def __init__(self, tools, cfg=CONFIG):
        FakeAgent.created.append(self)
        self.tools = tools
        self.cfg = cfg
        self.session_memory = Path(cfg.session_path)
        self.ran_task = None
        self.ran_interactive = False
        self.dumped = None

    def run(self, cfg=None):
        self.ran_interactive = True

    def run_task(self, task, cfg=None):
        self.ran_task = task
        return Result(
            outcome='completed', calls=0, turns=1, ok=0,
            failed_by_tag={}, calls_by_tool={},
            last_prompt=0, prompt_total=0, completion_total=0,
            wall=0.0, err='',
        )

    def dump_run(self, result, task, path=None, cfg=None):
        self.dumped = (task, path)


class TestPaperConfig(RootSessionGuard, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.cfg = replace(CONFIG, work_space=self.tmp)
        FakeAgent.created.clear()

    def test_build_paper_config_isolates_session_and_prompt(self):
        paper_cfg = cli.build_paper_config(self.cfg)
        self.assertTrue(paper_cfg.session_path.endswith('sandbox/paperpilot/session.json'))
        self.assertNotEqual(paper_cfg.session_path, str(self.tmp / 'session.json'))
        self.assertNotIn('Mini Harness', paper_cfg.system_prompt)
        self.assertIn('PaperPilot', paper_cfg.system_prompt)

    def test_build_paper_config_keeps_provider_settings(self):
        paper_cfg = cli.build_paper_config(self.cfg)
        for field in ('provider', 'model_main', 'base_url', 'api_key_env', 'max_turns_main'):
            self.assertEqual(getattr(paper_cfg, field), getattr(self.cfg, field))

    def test_no_api_key_in_cli_source(self):
        source = Path(cli.__file__).read_text(encoding='utf-8')
        self.assertNotIn('sk-', source)
        self.assertNotIn('api_key=', source.replace('api_key_env', ''))


class TestMain(RootSessionGuard, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.cfg = replace(CONFIG, work_space=self.tmp)
        FakeAgent.created.clear()

    def test_help_exits_zero(self):
        with mock.patch('sys.argv', ['paperpilot', '--help']):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(cfg=self.cfg)
        self.assertEqual(ctx.exception.code, 0)

    def test_task_runs_research_agent_with_paper_tools(self):
        with mock.patch.object(cli, 'DeepSeekAgent', FakeAgent):
            with mock.patch('sys.argv', ['paperpilot', '--task', 'find LoRA evidence']):
                with self.assertRaises(SystemExit) as ctx:
                    cli.main(cfg=self.cfg)
        self.assertEqual(ctx.exception.code, 0)

        agent = FakeAgent.created[-1]
        self.assertEqual({t.name for t in agent.tools}, {t.name for t in PAPER_TOOLS})
        self.assertTrue(agent.cfg.session_path.endswith('sandbox/paperpilot/session.json'))
        self.assertNotEqual(agent.cfg.session_path, str(self.tmp / 'session.json'))
        self.assertEqual(agent.ran_task, 'find LoRA evidence')
        self.assertEqual(agent.dumped[0], 'find LoRA evidence')

    def test_task_without_flag_runs_interactive(self):
        with mock.patch.object(cli, 'DeepSeekAgent', FakeAgent):
            with mock.patch('sys.argv', ['paperpilot']):
                cli.main(cfg=self.cfg)
        agent = FakeAgent.created[-1]
        self.assertEqual({t.name for t in agent.tools}, {t.name for t in PAPER_TOOLS})
        self.assertTrue(agent.ran_interactive)
        self.assertIsNone(agent.ran_task)


if __name__ == '__main__':
    unittest.main()
