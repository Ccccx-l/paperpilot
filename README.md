# PaperPilot

**面向科研论文的本地长程研究 Agent。** PaperPilot 将 PDF 导入本地索引，按关键词检索证据，读取命中页面的原文，再由你配置的模型回答问题并给出论文标题与页码。项目基于仓库内的 [mini-harness](https://github.com/mini-harness/mini-harness)，复用其 Agent 主循环、工具执行器和模型配置机制。

PaperPilot 提供 **本地 Web 工作台** 和 **CLI**。PDF 解析、索引和检索在本机执行；模型推理是否离开本机，取决于你配置的是云端 API 还是本地模型服务。

## 功能与边界

| 能力 | 当前实现 |
| --- | --- |
| PDF 导入 | 抽取逐页文本；使用 SHA-256 检测文件变化；损坏或加密 PDF 会报告失败 |
| 全文检索 | SQLite FTS5；中文分词与 NFKC 规范化；严格匹配与宽松召回相结合 |
| 原文核验 | 检索片段用于定位论文和页码；引用原文前应读取对应页面全文 |
| Web 工作台 | 论文列表、导入、研究对话、原文阅读、工具执行轨迹和会话恢复 |
| 本地持久化 | SQLite 索引和会话保存在工作目录中，重启后可继续使用 |

**重要限制**

- 检索是词法检索，不是向量检索或语义搜索。它不能保证找出只在语义上相关、但没有共享关键词的段落。
- 检索片段只用于定位证据。需要逐字引用时，必须以 `read_paper_page` 返回的页面全文为准。
- 模型服务必须支持 OpenAI 兼容的工具调用（`tool_calls`）。仅能进行普通文本生成的模型不能可靠驱动导入、检索和原文读取流程。
- 论文内容会进入模型上下文。使用云端模型时，相关问题和工具返回的论文片段或页面正文会发送给对应的模型服务。详见[隐私边界](#隐私边界)。

## Quickstart：从零启动 Web 工作台

下面使用 **DeepSeek 云端 API** 作为首次运行路径。先按此流程验证安装、模型连接和论文检索，再根据需要切换到本地模型。

### 1. 准备环境

需要：

- Linux、macOS 或 Windows。
- Python 3.12 或更高版本。仓库的 `.python-version` 指定 3.12。
- Git。
- [uv](https://docs.astral.sh/uv/getting-started/installation/)。
- 一个支持工具调用的模型 API，以及可用的 API Key。

安装 uv：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell：在 PowerShell 中执行这一行
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后重新打开终端，并确认命令可用：

```bash
uv --version
git --version
```

### 2. 克隆仓库并安装锁定依赖

以下命令请在终端中逐行执行：

```bash
git clone https://github.com/Ccccx-l/paperpilot.git
cd paperpilot
uv python install 3.12
uv sync --locked
uv run --locked paperpilot --help
```

最后一条命令应显示 `paperpilot` 的命令行帮助。若失败，请先解决安装问题，不要继续配置模型。运行命令时建议始终留在仓库根目录；PaperPilot 默认把当前工作目录当作研究工作目录。

`uv sync --locked` 根据仓库中的 `uv.lock` 安装依赖。首次执行需要联网下载 Python 运行时（若本机尚无 Python 3.12）和项目依赖。

### 3. 配置模型 API

先在 [DeepSeek API Keys](https://platform.deepseek.com/api_keys) 创建 API Key，然后在仓库根目录创建一个名为 `.env` 的文件，内容如下：

```dotenv
MINI_HARNESS_PROVIDER=deepseek
MINI_HARNESS_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=sk-替换为你自己的真实密钥
```

请把最后一行替换成真实密钥，不要保留示例值。DeepSeek 的默认 API 地址由项目配置提供，以上配置不需要额外填写 Base URL。

如果你选择复制示例文件：

```bash
# macOS / Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

**复制后必须检查并修改 `.env`。** 当前 `.env.example` 默认启用的是本地模型配置（`provider=local`，地址为 `127.0.0.1:8000`）。如果没有启动对应的本地模型服务，直接使用该默认配置会导致连接失败。首次运行云端模型时，请将其活动配置替换为上面的 DeepSeek 三行配置。

`.env` 中的变量会在启动时加载。修改配置后，请重启正在运行的 PaperPilot 进程。不要把真实密钥提交到 Git 或粘贴到公开的 issue 中；仓库的 `.gitignore` 已忽略 `.env`。

使用 OpenAI 官方 API 时，可以将配置改为：

```dotenv
MINI_HARNESS_PROVIDER=openai
MINI_HARNESS_MODEL=gpt-4.1
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-替换为你自己的真实密钥
```

无论选择哪家服务，都要确认账号有权访问所填模型，并且模型/API 端点支持工具调用。

### 4. 放入 PDF

在仓库根目录执行：

```bash
mkdir -p papers
cp "/absolute/path/to/your-paper.pdf" papers/
```

Windows PowerShell 示例：

```powershell
New-Item -ItemType Directory -Force papers
Copy-Item "C:/path/to/your-paper.pdf" papers/
```

将示例路径替换为你机器上真实存在的 PDF 路径。请把 PDF **直接放在 `papers/` 目录下**；当前导入工具只扫描指定目录中的顶层 `*.pdf`，不会递归扫描更深层的子目录。PaperPilot 不会因为文件被复制进目录就自动建立索引，下一步还需要显式导入。

### 5. 启动 Web 工作台

在仓库根目录运行：

```bash
uv run --locked python -m paperpilot.web
```

启动成功时，终端会打印类似以下信息（具体路径取决于你的工作目录）：

```text
[paperpilot-web]: http://127.0.0.1:8001
[paperpilot-web]: work_space = ...
[paperpilot-web]: db = .../sandbox/paperpilot/index.db
[paperpilot-web]: web_sessions = .../sandbox/paperpilot/web_sessions
```

保持这个终端运行，在浏览器打开：

**http://127.0.0.1:8001/**

在工作台中：

1. 在「论文与证据」面板确认导入目录为 `papers`。
2. 点击「导入论文」，等待导入结果。确认至少有一篇 PDF 显示为已导入；若失败，先处理页面显示的具体错误。
3. 在研究对话框中提出一个确实能在这些 PDF 中找到的主题问题。例如，若论文讨论 LoRA，可以输入：

   `请查找论文中关于 LoRA 的证据。先检索关键词，再读取命中页面的全文；引用时注明准确的论文标题和页码。如果原文不支持结论，请明确说明。`

4. 检查回答下方的工具执行轨迹。正常流程应包含论文导入（首次使用时）、关键词检索，以及引用原文前的页面读取。若语料里没有相关证据，正确结果应是报告未找到，而不是生成虚构的页码或引文。

**注意：** 上面的示例问题假定你的 PDF 讨论 LoRA。请将主题替换成实际论文中出现的概念。PaperPilot 只检索已导入的论文，不会自动从互联网下载论文。

### 6. 使用 CLI（可选）

Web 工作台运行后，也可以在另一个终端的仓库根目录中使用 CLI。先导入 PDF：

```bash
uv run --locked paperpilot --task "import the papers dir"
```

再针对已导入的论文提问：

```bash
uv run --locked paperpilot --task "Find evidence about LoRA. Search with concise keywords, read the full original page before quoting, and cite the exact paper title and page number. If no supporting evidence is found, say so."
```

请把 `LoRA` 换成你的论文主题。若不传入 `--task`，会进入交互模式：

```bash
uv run --locked paperpilot
```

需要保存本次运行的遥测报告时，可添加 `--telemetry-out run.json`：

```bash
uv run --locked paperpilot --task "import the papers dir" --telemetry-out run.json
```

## 使用本地模型

如果不希望把论文内容发送给云端模型，可以连接本机的 OpenAI 兼容模型服务。模型服务需要先单独安装、启动，并且必须支持工具调用。

### 配置 PaperPilot

将仓库根目录的 `.env` 配置为与你实际启动的模型服务一致，例如：

```dotenv
MINI_HARNESS_PROVIDER=local
MINI_HARNESS_MODEL=qwen3-vl-8b
LOCAL_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_API_KEY=EMPTY
```

这里的模型名必须与模型服务公开的模型 ID 一致。若你的服务使用不同端口、模型 ID 或鉴权方式，请相应修改配置；不要同时保留相互冲突的活动配置。

### 使用仓库提供的 vLLM 启动脚本

仓库的 `start_local_model.sh` 适用于**已经安装 vLLM、已经准备好模型权重和 vLLM 虚拟环境**的机器。它不会替你下载模型或安装 vLLM。示例：

```bash
MODEL_PATH=/absolute/path/to/Qwen3-VL-8B-Instruct VLLM_VENV=/absolute/path/to/vllm-venv MODEL_NAME=qwen3-vl-8b bash start_local_model.sh
```

请替换两个绝对路径。脚本默认在 `127.0.0.1:8000` 启动服务，并启用 vLLM 的工具调用选项。启动脚本后，另开终端检查服务是否可访问：

```bash
curl http://127.0.0.1:8000/v1/models
```

确认返回的模型 ID 与 `MINI_HARNESS_MODEL` 相符，再按 Quickstart 第 5 步启动 PaperPilot。若你的模型或推理服务不支持兼容的工具调用格式，即使 `/v1/models` 可访问，Agent 仍可能无法执行论文工具。

## 数据存放位置

默认工作目录是**执行命令时的当前目录**。如果按 Quickstart 从仓库根目录启动，默认位置如下：

| 数据 | 默认位置 |
| --- | --- |
| PDF 文件 | `<工作目录>/papers/` |
| SQLite 论文索引 | `<工作目录>/sandbox/paperpilot/index.db` |
| CLI 会话 | `<工作目录>/sandbox/paperpilot/session.json` |
| Web 会话 | `<工作目录>/sandbox/paperpilot/web_sessions/` |

可以用 `MINI_HARNESS_WORK_SPACE` 指定独立的研究目录。例如：

```bash
MINI_HARNESS_WORK_SPACE=/data/my-research uv run --locked python -m paperpilot.web
```

设置独立工作目录后，请把 PDF 放在该目录的 `papers/` 中，并在 Web 界面中导入该目录。所有相对路径均以工作目录为基准。不同工作目录会使用不同的索引库和会话文件；建议每个研究项目固定使用一个工作目录。

## 隐私边界

- **本地执行：** PDF 解析、SQLite 索引和关键词检索在本机完成。
- **使用云端模型：** 系统提示词、研究问题、对话历史，以及工具返回的论文标题、页码、检索片段和被读取页面的正文，可能通过 API 发送给所配置的模型服务。
- **使用本地模型：** 如果配置的模型端点确实运行在本机且没有额外转发，推理请求可以留在本机；请自行核实推理服务、代理和日志的部署方式。
- Web 服务默认只绑定 `127.0.0.1:8001`，供本机浏览器访问，不是面向公网的部署配置。

对隐私敏感的论文，请优先评估本地推理方案，并确认模型服务没有将请求转发到第三方。

## 常见问题排查

| 现象 | 检查方式 |
| --- | --- |
| `uv run paperpilot --help` 失败 | 确认已进入仓库根目录，并重新运行 `uv sync --locked`。 |
| API 返回 401/403 | 检查 `.env` 中的密钥、变量名和模型访问权限；确认修改后已重启进程。不要把密钥发到 issue。 |
| 连接 `127.0.0.1:8000` 失败 | 这通常意味着当前配置在调用本地模型，但本地服务未启动或端口不匹配。检查 `MINI_HARNESS_PROVIDER`、`LOCAL_BASE_URL`，并用 `curl http://127.0.0.1:8000/v1/models` 检查服务。 |
| Agent 没有调用论文工具 | 确认模型/API 服务支持 OpenAI 兼容的 `tool_calls`，并检查服务端工具调用配置。 |
| 导入时找不到 PDF | 确认 PDF 位于当前工作目录的 `papers/` 中，且文件直接放在目录顶层；确认在 Web 中执行了导入。 |
| 检索不到相关内容 | 确认论文已经导入；尝试 2–6 个简洁、明确的关键词。检索为词法匹配，不保证命中只有语义相关但没有共同关键词的内容。 |
| 请求超过模型上下文长度 | 使用支持更大上下文的模型，减少单次问题涉及的材料，并检查模型服务实际上下文限制。配置细节见 [CONFIGURATION.md](docs/CONFIGURATION.md)。 |
| Web 启动时报端口占用 | 确认是否已有 PaperPilot Web 进程在运行；不要误停使用 8000 端口的本地模型服务。 |

## 开发与测试

在仓库根目录执行：

```bash
# 安装项目及开发依赖
uv sync --locked

# 单元测试
uv run --locked python -m unittest discover -s tests -t .

# 静态检查
uv run --locked ruff check src tests

# 检查 uv.lock 是否与项目配置一致
uv lock --check
```

## 进一步阅读

| 文档 | 内容 |
| --- | --- |
| [docs/INSTALL.md](docs/INSTALL.md) | 安装、启动和基础排错 |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | 模型服务、环境变量、上下文预算与重试配置 |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | 论文导入、数据管理、备份恢复与运维 |
| [mini-harness](https://github.com/mini-harness/mini-harness) | PaperPilot 使用的 Agent harness 基础项目 |
