import asyncio
import re
import shlex
import tempfile
from pathlib import Path
from typing import override

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from bench.adapter import LOG_DIR, MiniHarnessAgent
from bench.sb_cli import Prediction, instance_id_for_trial, write_json


class SBCLIPredictionAgent(MiniHarnessAgent):
    def __init__(self, logs_dir: Path, repo_dir: str = "/testbed", **kwargs):
        super().__init__(logs_dir=logs_dir, **kwargs)
        self.repo_dir = repo_dir

    async def export_prediction(
        self, environment: BaseEnvironment, base_commit: str
    ) -> None:
        remote_patch = f"{LOG_DIR}/model.patch"
        command = (
            "set -euo pipefail; "
            "index_dir=$(mktemp -d); "
            "trap 'rm -rf \"$index_dir\"' EXIT; "
            'export GIT_INDEX_FILE="$index_dir/index"; '
            f"git read-tree {shlex.quote(base_commit)}; "
            "git -c core.fileMode=true add -A -- .; "
            f"mkdir -p {shlex.quote(LOG_DIR)}; "
            f"git -c core.fileMode=true diff --cached --binary --full-index --no-ext-diff --no-textconv {shlex.quote(base_commit)} -- > {shlex.quote(remote_patch)}"
        )
        await self.exec_as_agent(
            environment, command=command, cwd=self.repo_dir, timeout_sec=30
        )
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "model.patch"
            await environment.download_file(remote_patch, patch)
            prediction = Prediction(
                instance_id=instance_id_for_trial(Path(self.logs_dir).parent),
                model_name_or_path=f"mini-harness__{self.model_name.replace('/', '__')}",
                model_patch=patch.read_text(encoding="utf-8"),
            )
            path = Path(directory) / "prediction.json"
            write_json(path, prediction.model_dump())
            await environment.upload_file(path, f"{LOG_DIR}/prediction.json")
            write_json(
                Path(self.logs_dir) / "prediction.json",
                prediction.model_dump(),
                overwrite=True,
            )

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        result = await self.exec_as_agent(
            environment,
            command="git rev-parse --verify HEAD",
            cwd=self.repo_dir,
            timeout_sec=30,
        )
        base_commit = (result.stdout or "").strip()
        if not re.fullmatch(r"[0-9a-f]{40,64}", base_commit):
            raise ValueError("Cannot determine the task repository base commit")
        try:
            await super().run(instruction, environment, context)
        except BaseException:
            try:
                await asyncio.wait_for(
                    self.export_prediction(environment, base_commit), timeout=60
                )
            except Exception as error:
                self.logger.error("Failed to preserve partial prediction: %s", error)
            raise
        else:
            await asyncio.wait_for(
                self.export_prediction(environment, base_commit), timeout=60
            )
