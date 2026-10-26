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


class TestRunTaskHistory(RootSessionGuard, unittest.TestCase):
    """Batch A: run_task() optional history + turn boundary + memory toggle.

    Covers T1 (single-turn default unchanged) and T2 (history injection keeps
    system + prior + current user order) from the multi-turn plan, plus the
    two design corrections: turn_start must account for system messages, and
    web runs must be able to suppress session.json writes.
    """

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
            system_prompt='SYSTEM PROMPT',
            base_url='http://127.0.0.1:9/history-test',
            profile='paperpilot-test',
            model_main='model-under-test',
            think_main='off',
            max_turns_main=7,
        )
        base.update(overrides)
        return replace(CONFIG, **base)

    def _run(self, task='q', prior=None, **agent_kwargs):
        cfg = agent_kwargs.pop('cfg', None) or self.make_cfg()
        agent = DeepSeekAgent([], cfg=cfg, **agent_kwargs)
        with mock.patch('mini_harness.agent.OpenAI', OpenAISpy):
            with mock.patch.object(DeepSeekAgent, '_run_turn', return_value=make_result()):
                agent.run_task(task, prior=prior)
        return agent

    def test_t1_default_is_single_turn_system_plus_user(self):
        agent = self._run('only question')
        self.assertEqual([m['role'] for m in agent.message], ['system', 'user'])
        self.assertEqual(agent.message[0]['content'], 'SYSTEM PROMPT')
        self.assertEqual(agent.message[1]['content'], 'only question')

    def test_t2_prior_keeps_system_prior_user_order(self):
        prior = [
            {'role': 'user', 'content': 'old question'},
            {'role': 'assistant', 'content': 'old answer'},
        ]
        agent = self._run('new question', prior=prior)
        self.assertEqual(
            [(m['role'], m['content']) for m in agent.message],
            [('system', 'SYSTEM PROMPT'),
             ('user', 'old question'),
             ('assistant', 'old answer'),
             ('user', 'new question')],
        )

    def test_turn_start_points_at_current_user_after_system(self):
        prior = [{'role': 'user', 'content': 'old question'},
                 {'role': 'assistant', 'content': 'old answer'}]
        agent = self._run('new question', prior=prior)
        # NOT len(prior): system messages must be counted too.
        self.assertEqual(agent.turn_start, 3)
        self.assertEqual(agent.message[agent.turn_start]['role'], 'user')
        self.assertEqual(agent.message[agent.turn_start]['content'], 'new question')
        self.assertEqual([m['content'] for m in agent.message[:agent.turn_start]],
                         ['SYSTEM PROMPT', 'old question', 'old answer'])

    def test_turn_start_without_prior_is_just_after_system(self):
        agent = self._run('q')
        self.assertEqual(agent.turn_start, 1)
        self.assertEqual(agent.message[agent.turn_start]['content'], 'q')

    def test_prior_is_deep_copied_not_aliased(self):
        prior = [{'role': 'user', 'content': 'old', 'tool_calls': []}]
        agent = self._run('new', prior=prior)
        self.assertIsNot(agent.message[1], prior[0])
        prior[0]['content'] = 'MUTATED BY CALLER'
        self.assertEqual(agent.message[1]['content'], 'old')
        agent.message[1]['content'] = 'MUTATED BY AGENT'
        self.assertEqual(prior[0]['content'], 'MUTATED BY CALLER')

    def test_system_list_is_not_shared_with_prior_slice(self):
        prior = [{'role': 'user', 'content': 'u1'}, {'role': 'assistant', 'content': 'a1'}]
        agent = self._run('u2', prior=prior)
        self.assertEqual(len(agent.system), 1)
        self.assertEqual([m['role'] for m in agent.message[:3]], ['system', 'user', 'assistant'])

    def test_memory_enabled_false_writes_no_session_file(self):
        self._run('q', memory_enabled=False)
        self.assertFalse((self.tmp / 'paper_session.json').exists())

    def test_memory_enabled_default_still_saves(self):
        self._run('q')
        self.assertTrue((self.tmp / 'paper_session.json').is_file())

    def test_memory_enabled_false_suppresses_save_memory_call(self):
        agent = DeepSeekAgent([], cfg=self.make_cfg(), memory_enabled=False)
        agent.message.append({'role': 'user', 'content': 'payload'})
        agent._save_memory(quiet=True)
        self.assertFalse((self.tmp / 'paper_session.json').exists())


if __name__ == '__main__':
    unittest.main()
