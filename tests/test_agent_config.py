"""Regression tests: a custom Config passed to DeepSeekAgent must be used
consistently for the system prompt, the session path, the model request and the
execution policy, instead of silently falling back to the global CONFIG.
"""

import io
import json
import tempfile
import unittest

from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from mini_harness.agent import DeepSeekAgent, Result
from mini_harness.config import CONFIG
from mini_harness.tool.tag import MARK

from tests._support import OpenAISpy, RootSessionGuard


def make_result() -> Result:
    return Result(
        outcome='completed', calls=0, turns=1, ok=0,
        failed_by_tag={}, calls_by_tool={},
        last_prompt=0, prompt_total=0, completion_total=0,
        wall=0.0, err='',
    )


class TestAgentConfigPropagation(RootSessionGuard, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        OpenAISpy.instances.clear()

    def make_cfg(self, **overrides):
        base = dict(
            work_space=self.tmp,
            session_path=str(self.tmp / 'paper_session.json'),
            system_prompt='PAPERPILOT TEST PROMPT',
            base_url='http://127.0.0.1:9/unified-test-endpoint',
            profile='paperpilot-test',
            model_main='model-under-test',
            think_main='off',
            max_turns_main=7,
        )
        base.update(overrides)
        return replace(CONFIG, **base)

    def test_system_prompt_comes_from_constructor_cfg(self):
        cfg = self.make_cfg()
        agent = DeepSeekAgent([], cfg=cfg)
        self.assertIs(agent.cfg, cfg)
        self.assertEqual(agent.system, [{'role': 'system', 'content': 'PAPERPILOT TEST PROMPT'}])
        self.assertEqual(agent.message[0]['content'], 'PAPERPILOT TEST PROMPT')

    def test_session_path_comes_from_constructor_cfg(self):
        cfg = self.make_cfg()
        agent = DeepSeekAgent([], cfg=cfg)
        expected = self.tmp / 'paper_session.json'
        self.assertEqual(Path(agent.session_memory), expected)

        agent.message.append({'role': 'user', 'content': 'session payload'})
        agent._save_memory(quiet=True)
        self.assertTrue(expected.is_file())
        self.assertIn('session payload', expected.read_text(encoding='utf-8'))

    def test_default_session_path_is_under_work_space(self):
        cfg = self.make_cfg(session_path=None)
        agent = DeepSeekAgent([], cfg=cfg)
        self.assertEqual(Path(agent.session_memory), self.tmp / 'session.json')

        agent._save_memory(quiet=True)
        self.assertTrue((self.tmp / 'session.json').is_file())

    def test_save_memory_defaults_to_constructor_cfg(self):
        cfg = self.make_cfg()
        agent = DeepSeekAgent([], cfg=cfg)
        # No cfg argument: must resolve to the constructor cfg's session path.
        agent._save_memory(quiet=True)
        self.assertTrue((self.tmp / 'paper_session.json').is_file())

    def test_run_task_uses_constructor_cfg_for_request_and_execution(self):
        cfg = self.make_cfg()
        agent = DeepSeekAgent([], cfg=cfg)

        with mock.patch('mini_harness.agent.OpenAI', OpenAISpy):
            with mock.patch.object(DeepSeekAgent, '_run_turn', return_value=make_result()) as turn:
                result = agent.run_task('a research question')

        self.assertEqual(result.outcome, 'completed')
        self.assertEqual(len(OpenAISpy.instances), 1)
        self.assertEqual(OpenAISpy.instances[0].base_url, 'http://127.0.0.1:9/unified-test-endpoint')
        self.assertTrue(turn.call_args.kwargs.get('cfg') is cfg,
                        'run_task must pass the constructor cfg to _run_turn')
        self.assertTrue((self.tmp / 'paper_session.json').is_file())

    def test_run_task_explicit_cfg_still_overrides(self):
        cfg = self.make_cfg()
        other = replace(cfg, base_url='http://127.0.0.1:9/override-endpoint')
        agent = DeepSeekAgent([], cfg=cfg)

        with mock.patch('mini_harness.agent.OpenAI', OpenAISpy):
            with mock.patch.object(DeepSeekAgent, '_run_turn', return_value=make_result()) as turn:
                agent.run_task('a research question', cfg=other)

        self.assertEqual(OpenAISpy.instances[0].base_url, 'http://127.0.0.1:9/override-endpoint')
        self.assertTrue(turn.call_args.kwargs.get('cfg') is other,
                        'an explicit cfg argument must still win over the constructor cfg')

    def test_method_cfg_override_scope(self):
        """Documented config priority: a method-level cfg overrides request and
        execution options for that call only. It must NOT move the session file
        or replace the system prompt fixed at construction time."""
        cfg = self.make_cfg()
        other = replace(
            cfg,
            base_url='http://127.0.0.1:9/override-endpoint',
            system_prompt='OTHER PROMPT MUST NOT APPLY',
            session_path=str(self.tmp / 'other_session.json'),
        )
        agent = DeepSeekAgent([], cfg=cfg)

        with mock.patch('mini_harness.agent.OpenAI', OpenAISpy):
            with mock.patch.object(DeepSeekAgent, '_run_turn', return_value=make_result()) as turn:
                agent.run_task('a research question', cfg=other)

        # Overridden for this call: request endpoint and the cfg passed down.
        self.assertEqual(OpenAISpy.instances[0].base_url, 'http://127.0.0.1:9/override-endpoint')
        self.assertTrue(turn.call_args.kwargs.get('cfg') is other)
        # NOT overridden: the system prompt is fixed at construction time.
        self.assertEqual(agent.message[0]['content'], 'PAPERPILOT TEST PROMPT')
        # NOT overridden: the session file stays at the constructor path.
        self.assertTrue((self.tmp / 'paper_session.json').is_file())
        self.assertFalse((self.tmp / 'other_session.json').exists())

    def test_dump_run_defaults_to_constructor_cfg(self):
        cfg = self.make_cfg()
        agent = DeepSeekAgent([], cfg=cfg)
        out = self.tmp / 'telemetry.json'

        with redirect_stdout(io.StringIO()) as stdout:
            agent.dump_run(make_result(), 'the task', str(out))

        payload = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(payload['profile'], 'paperpilot-test')
        self.assertEqual(payload['model'], 'model-under-test')
        self.assertEqual(payload['thinking'], 'off')
        self.assertEqual(payload['max_turns_main'], 7)
        self.assertEqual(payload['task'], 'the task')
        self.assertIn(MARK, stdout.getvalue())

    def test_dump_run_explicit_cfg_still_overrides(self):
        cfg = self.make_cfg()
        other = replace(cfg, profile='other-profile', model_main='model-other')
        agent = DeepSeekAgent([], cfg=cfg)
        out = self.tmp / 'telemetry_other.json'

        with redirect_stdout(io.StringIO()):
            agent.dump_run(make_result(), 'the task', str(out), cfg=other)

        payload = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(payload['profile'], 'other-profile')
        self.assertEqual(payload['model'], 'model-other')

    def test_run_quit_saves_to_constructor_session_path(self):
        cfg = self.make_cfg()
        agent = DeepSeekAgent([], cfg=cfg)

        with mock.patch('mini_harness.agent.OpenAI', OpenAISpy):
            with mock.patch('builtins.input', return_value='quit'):
                with redirect_stdout(io.StringIO()):
                    agent.run()

        self.assertTrue((self.tmp / 'paper_session.json').is_file())
        self.assertFalse((self.tmp / 'session.json').exists())


if __name__ == '__main__':
    unittest.main()
