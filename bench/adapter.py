import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import override

from dotenv import load_dotenv
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from bench import atif

REPO_ROOT = Path(__file__).resolve().parent.parent
GUEST_PKG = "/opt/mini-harness"
GUEST_SECRET = "/opt/mini_harness_secret"
KEY_FILE = f"{GUEST_SECRET}/api.key"
LOG_DIR = "/logs/agent"
UV_ENV = '. "$HOME/.local/bin/env"'
load_dotenv(REPO_ROOT / ".env")
AGENT_VERSION = atif.git_version(REPO_ROOT)


class MiniHarnessAgent(BaseInstalledAgent):
    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        sub_model: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens_main: int | None = None,
        max_tokens_sub: int | None = None,
        compact_limit: int | None = None,
        **kwargs,
    ):
        settings = {**os.environ, **(kwargs.get("extra_env") or {})}
        model = model_name or settings.get("MINI_HARNESS_MODEL")
        prefix = model.partition("/")[0] if model else None
        inferred = (
            prefix
            if prefix in {"openai", "deepseek"}
            else (
                "deepseek" if not model or model.startswith("deepseek-") else "openai"
            )
        )
        provider = (
            provider or settings.get("MINI_HARNESS_PROVIDER") or inferred
        ).lower()
        if provider not in {"openai", "deepseek"}:
            raise ValueError(f"Unsupported provider: {provider}")
        if prefix in {"openai", "deepseek"}:
            if prefix != provider:
                raise ValueError("Model prefix conflicts with provider")
            model = model.partition("/")[2]
        if model is not None and not model.strip():
            raise ValueError("model_name must not be empty")
        self._model_id = model or (
            "gpt-4.1" if provider == "openai" else "deepseek-v4-flash"
        )
        key_env = (
            api_key_env
            or settings.get("MINI_HARNESS_API_KEY_ENV")
            or f"{provider.upper()}_API_KEY"
        )
        self._api_key = settings.get("MINI_HARNESS_API_KEY") or settings.get(key_env)
        if not self._api_key:
            raise RuntimeError(f"[adapter]: set {key_env} or MINI_HARNESS_API_KEY")
        self._run_env = {
            "MINI_HARNESS_PROFILE": "bench",
            "MINI_HARNESS_PROVIDER": provider,
            "MINI_HARNESS_API_KEY_ENV": key_env,
            "MINI_HARNESS_MODEL": self._model_id,
            "PYTHONUNBUFFERED": "1",
        }
        options = {
            "BASE_URL": base_url,
            "SUB_MODEL": sub_model,
            "REASONING_EFFORT": reasoning_effort,
            "MAX_TOKENS_MAIN": max_tokens_main,
            "MAX_TOKENS_SUB": max_tokens_sub,
            "COMPACT_LIMIT": compact_limit,
        }
        for name, value in options.items():
            value = value if value is not None else settings.get(f"MINI_HARNESS_{name}")
            if value is not None:
                if name in {"MAX_TOKENS_MAIN", "MAX_TOKENS_SUB", "COMPACT_LIMIT"}:
                    if int(value) <= 0:
                        raise ValueError(f"{name.lower()} must be positive")
                self._run_env[f"MINI_HARNESS_{name}"] = str(value)
        if provider == "openai" and "MINI_HARNESS_BASE_URL" not in self._run_env:
            if value := settings.get("OPENAI_BASE_URL"):
                self._run_env["MINI_HARNESS_BASE_URL"] = value
        super().__init__(
            logs_dir=logs_dir, model_name=f"{provider}/{self._model_id}", **kwargs
        )

    @staticmethod
    @override
    def name() -> str:
        return "mini-harness"

    @override
    def get_version_command(self) -> str | None:
        return None

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl ca-certificates",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "pkg"
            stage.mkdir()
            shutil.copy(REPO_ROOT / "pyproject.toml", stage)
            shutil.copytree(REPO_ROOT / "src", stage / "src")
            if (REPO_ROOT / "README.md").exists():
                shutil.copy(REPO_ROOT / "README.md", stage)
            await environment.upload_dir(stage, GUEST_PKG)
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp) / "secret"
            secret.mkdir()
            (secret / "api.key").write_text(self._api_key, encoding="utf-8")
            await environment.upload_dir(secret, GUEST_SECRET)
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                f"{UV_ENV} && "
                f"uv tool install --python 3.12 {GUEST_PKG} && "
                "which mini-harness"
            ),
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        try:
            d = Path(self.logs_dir)
            traj = atif.build(d, version=AGENT_VERSION, model_name=self._model_id)
            if traj is None:
                print(
                    f"[adapter]: no trajectory for {d.parent.name} (missing or empty session)"
                )
                return
            atif.write_trajectory(d, traj)
            fm = traj.final_metrics
            if fm is not None:
                context.n_input_tokens = fm.total_prompt_tokens
                context.n_output_tokens = fm.total_completion_tokens
        except Exception as e:
            print(
                f"[adapter]: trajectory conversion failed for {self.logs_dir}: {type(e).__name__}: {e}"
            )

    @override
    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        escaped = shlex.quote(instruction)
        await self.exec_as_agent(
            environment,
            command=(
                f"{UV_ENV}; "
                f'export MINI_HARNESS_API_KEY="$(cat {KEY_FILE} 2>/dev/null)"; '
                f"rm -f {KEY_FILE}; "
                'test -n "$MINI_HARNESS_API_KEY" || exit 3; '
                f"mkdir -p {LOG_DIR}; "
                f"mini-harness --task {escaped} "
                f"--telemetry-out {LOG_DIR}/mini_harness_tele.json "
                f"2>&1 | stdbuf -oL tee {LOG_DIR}/mini_harness.txt"
            ),
            env=self._run_env,
        )
