"""Regression tests for the tool-call interface of ToolExecution.execute_tool.

Dispatch policy under test: the executor cfg is passed to a tool function only
when that function declares a parameter named `cfg`. Tools without such a
parameter (e.g. the TUI offline demo's `lambda inp: ...`) keep the legacy
funs(args) call shape. Security policy and path validation must be unchanged.
"""

import json
import tempfile
import unittest

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from dataclasses import replace

from mini_harness.config import CONFIG
from mini_harness.tool.box import (
    ReadFileInput, TOOLS, ToolDefinition, ToolExecution, read_file,
)
from mini_harness.tool.tag import TAG

from tests._support import RootSessionGuard, make_call


class LabelInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: str = Field(description='probe label')


class TestToolCfgInjection(RootSessionGuard, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.cfg = replace(CONFIG, work_space=self.tmp)
        self.seen = {}

    def run_tool(self, definition: ToolDefinition, arguments: dict, cfg=None, confirm=None):
        executer = ToolExecution({definition.name: definition},
                                 confirm or (lambda call, cfg=None: True), cfg=self.cfg)
        call = make_call(definition.name, json.dumps(arguments))
        return executer.execute_tool(call, cfg=self.cfg if cfg is None else cfg)

    def test_tool_with_cfg_receives_executor_cfg(self):
        def spy_with_cfg(inp: LabelInput, cfg=CONFIG) -> str:
            self.seen['cfg'] = cfg
            return f'ok:{inp.label}'

        definition = ToolDefinition('spy_with_cfg', 'probe', LabelInput, spy_with_cfg, False)
        res = self.run_tool(definition, {'label': 'a'})

        self.assertTrue(res.ok)
        self.assertEqual(res.content, 'ok:a')
        self.assertTrue(self.seen['cfg'] is self.cfg,
                        'tool must receive the exact cfg object passed to execute_tool')
        self.assertEqual(self.seen['cfg'].work_space, self.tmp)

    def test_tool_without_cfg_keeps_legacy_call_shape(self):
        # One-argument tool, exactly like the TUI offline demo replacement.
        def plain(inp: LabelInput) -> str:
            return f'plain:{inp.label}'

        definition = ToolDefinition('plain_tool', 'probe', LabelInput, plain, False)
        res = self.run_tool(definition, {'label': 'b'})

        self.assertTrue(res.ok)
        self.assertEqual(res.content, 'plain:b')

    def test_single_argument_lambda_does_not_break(self):
        definition = ToolDefinition('lambda_tool', 'probe', LabelInput,
                                   lambda inp: 'lambda-ok', False)
        res = self.run_tool(definition, {'label': 'c'})
        self.assertTrue(res.ok)
        self.assertEqual(res.content, 'lambda-ok')

    def test_positional_only_cfg_is_passed_positionally(self):
        def pos_only(inp: LabelInput, cfg, /) -> str:
            self.seen['cfg'] = cfg
            return 'ok'

        definition = ToolDefinition('pos_only_tool', 'probe', LabelInput, pos_only, False)
        res = self.run_tool(definition, {'label': 'd'})

        self.assertTrue(res.ok)
        self.assertTrue(self.seen['cfg'] is self.cfg,
                        'positional-only cfg must still receive the executor cfg')

    def test_bare_kwargs_does_not_receive_cfg(self):
        def kwargs_tool(inp: LabelInput, **kwargs) -> str:
            self.seen['kwargs'] = dict(kwargs)
            return 'ok'

        definition = ToolDefinition('kwargs_tool', 'probe', LabelInput, kwargs_tool, False)
        res = self.run_tool(definition, {'label': 'e'})

        self.assertTrue(res.ok)
        self.assertEqual(self.seen['kwargs'], {})

    def test_var_keyword_named_cfg_is_not_injected(self):
        # `**cfg` is a collector, not a config parameter: injecting cfg=...
        # here would silently swallow the config instead of receiving it.
        def collector(inp: LabelInput, **cfg) -> str:
            self.seen['kwargs'] = dict(cfg)
            return 'ok'

        definition = ToolDefinition('collector_tool', 'probe', LabelInput, collector, False)
        res = self.run_tool(definition, {'label': 'h'})

        self.assertTrue(res.ok)
        self.assertEqual(self.seen['kwargs'], {},
                         '**cfg collector must be called as fun(args), without cfg')

    def test_var_positional_named_cfg_is_not_injected(self):
        def collector(inp: LabelInput, *cfg) -> str:
            self.seen['args'] = cfg
            return 'ok'

        definition = ToolDefinition('var_pos_tool', 'probe', LabelInput, collector, False)
        res = self.run_tool(definition, {'label': 'i'})

        self.assertTrue(res.ok)
        self.assertEqual(self.seen['args'], (),
                         '*cfg collector must be called as fun(args), without cfg')

    def test_keyword_only_cfg_is_injected(self):
        def keyword_only(inp: LabelInput, *, cfg=CONFIG) -> str:
            self.seen['cfg'] = cfg
            return 'ok'

        definition = ToolDefinition('keyword_only_tool', 'probe', LabelInput, keyword_only, False)
        res = self.run_tool(definition, {'label': 'j'})

        self.assertTrue(res.ok)
        self.assertTrue(self.seen['cfg'] is self.cfg,
                        'keyword-only cfg is an ordinary config parameter and must receive it')

    def test_execute_tool_default_cfg_is_global_config(self):
        def spy(inp: LabelInput, cfg=CONFIG) -> str:
            self.seen['cfg'] = cfg
            return 'ok'

        definition = ToolDefinition('default_cfg_tool', 'probe', LabelInput, spy, False)
        executer = ToolExecution({definition.name: definition}, lambda call, cfg=None: True)
        res = executer.execute_tool(make_call(definition.name, json.dumps({'label': 'f'})))

        self.assertTrue(res.ok)
        self.assertTrue(self.seen['cfg'] is CONFIG,
                        'execute_tool called without cfg must fall back to the global CONFIG')

    def test_read_file_receives_custom_work_space(self):
        (self.tmp / 'evidence.txt').write_text('paper evidence line', encoding='utf-8')
        definition = ToolDefinition('read_file', 'read', ReadFileInput, read_file, False)
        res = self.run_tool(definition, {'file_path': 'evidence.txt'})

        self.assertTrue(res.ok)
        self.assertIn('paper evidence line', res.content)

    def test_read_file_still_denies_outside_work_space(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        victim = Path(outside.name) / 'outside.txt'
        victim.write_text('outside content', encoding='utf-8')

        definition = ToolDefinition('read_file', 'read', ReadFileInput, read_file, False)
        res = self.run_tool(definition, {'file_path': str(victim)})

        self.assertFalse(res.ok)
        self.assertTrue(res.tag.startswith(TAG.EXECUTE_FAILED))
        self.assertIn('PermissionError', res.content)
        self.assertNotIn('outside content', res.content)

    def test_read_file_still_denies_env_files(self):
        (self.tmp / '.env').write_text('FAKE_PLACEHOLDER=not-a-real-key', encoding='utf-8')
        definition = ToolDefinition('read_file', 'read', ReadFileInput, read_file, False)
        res = self.run_tool(definition, {'file_path': '.env'})

        self.assertFalse(res.ok)
        self.assertIn('PermissionError', res.content)
        self.assertNotIn('not-a-real-key', res.content)

    def test_risky_tool_still_requires_confirmation(self):
        def risky_probe(inp: LabelInput, cfg=CONFIG) -> str:
            return 'should not run'

        definition = ToolDefinition('risky_probe', 'probe', LabelInput, risky_probe, True)
        res = self.run_tool(definition, {'label': 'g'}, confirm=lambda call, cfg=None: False)

        self.assertFalse(res.ok)
        self.assertEqual(res.tag, TAG.DENIED)

    def test_stock_registry_functions_accept_cfg(self):
        # Every stock tool declares `cfg`; the dispatcher must pass it through.
        for tool in TOOLS:
            with self.subTest(tool=tool.name):
                import inspect
                self.assertIn('cfg', inspect.signature(tool.function).parameters)


if __name__ == '__main__':
    unittest.main()
