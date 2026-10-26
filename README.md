# PaperPilot

**面向科研论文的本地长程研究 Agent**——导入 PDF、全文检索、返回可追溯的原文证据
（论文标题 + 页码 + 原文片段），再由你配置的大模型依据证据作答。

PaperPilot 基于本仓库内置的 `mini-harness`（一个最小编码 Agent harness）构建，
复用它的 Agent 主循环、工具执行器与配置机制；论文工具与检索链路是独立模块。

> **定位**：开发者/科研人员的仓库式本地工具。PaperPilot 是面向开发者与科研人员的本地研究工具，提供 CLI 和本地 Web 工作台，不依赖云端托管的应用界面。

---

## 核心能力

| 能力 | 说明 |
|---|---|
| PDF 导入 | 逐页抽取文本，SHA-256 变更检测，损坏/加密文件明确报错 |
| 全文检索 | SQLite FTS5 + 中文分词（NFKC 规范化）+ 双路召回（严格 AND + 宽松 OR） |
| 受控扩展 | 词形归一、有出处的术语别名、查询级扩展、少数查询的子查询融合 |
| 证据可追溯 | 返回论文标题、页码、原文片段；高亮只标原文真实位置，绝不编造 |
| 本地持久化 | SQLite 索引 + 会话文件，重启不丢 |

**边界如实说明**：
- 检索是**词法匹配**（lexical），不是语义搜索；未解决"主题相关 vs 字面提及"问题。
- 证据片段会随工具结果进入模型上下文（见下方[隐私边界](#隐私边界)）。
- 模型必须支持 **OpenAI 兼容的 function calling（工具调用）**，否则无法驱动工具链。

---

## Quickstart

### 1. 前置条件

- **Python ≥ 3.12**
- **[uv](https://docs.astral.sh/uv/)**（推荐的包管理器）
- 一个**支持 function calling 的 OpenAI 兼容模型服务**（云端或本地均可）

### 2. 克隆并安装

```bash
git clone <本仓库地址>
cd <仓库目录>
uv sync --locked
```

### 3. 配置模型服务

```bash
cp .env.example .env
# 用编辑器打开 .env，填入你的模型服务配置
```

配置有两种典型形态——**云端 API** 与 **本地模型服务**，详见 `.env.example` 内注释。

配置变量的真实名称与优先级（以源码 `src/mini_harness/config.py` 为准）：

| 环境变量 | 作用 | 必需性 |
|---|---|---|
| `MINI_HARNESS_PROVIDER` | `openai` / `deepseek` / `local` 之一；不设置时按 `MINI_HARNESS_MODEL` 前缀推断 | 建议显式设置 |
| `MINI_HARNESS_MODEL` | 模型名；可用 `provider/model` 形式 | 是 |
| `MINI_HARNESS_API_KEY` | API 密钥（优先读取） | 视服务而定 |
| `{PROVIDER}_API_KEY` | 备选密钥变量名（如 `DEEPSEEK_API_KEY`、`LOCAL_API_KEY`、`OPENAI_API_KEY`） | 二选一 |
| `MINI_HARNESS_API_KEY_ENV` | 自定义密钥变量名 | 可选 |
| `OPENAI_BASE_URL` | `provider=openai` 时的 API 地址 | 按需 |
| `LOCAL_BASE_URL` | `provider=local` 时的 API 地址（默认 `http://127.0.0.1:8000/v1`） | 按需 |
| `MINI_HARNESS_WORK_SPACE` | 工作目录（决定数据位置，见下） | 可选 |

`.env` 由 `python-dotenv` 自动加载（从当前目录向上查找）。**密钥不会写入日志或
telemetry**，请勿将 `.env` 提交到 Git（`.gitignore` 已覆盖）。

### 4. 准备论文

把 PDF 放进**工作目录**下的 `papers/`（默认工作目录是启动 `paperpilot` 时所在的目录）：

```bash
mkdir -p papers
cp /path/to/your/*.pdf papers/
```

### 5. 启动 Web 工作台（推荐）

```bash
uv run python -m paperpilot.web
```

浏览器打开 **http://127.0.0.1:8001/**，获得：
- 研究对话区（Markdown 渲染）
- 论文列表、元数据、原文阅读器
- 底部工具执行流水线（导入/检索/读原文/引文核验）
- 论文导入、多轮对话、历史恢复

### 6. 或使用 CLI 模式

先看帮助：

```bash
uv run paperpilot --help
```

执行一个真实研究任务（Agent 会自主调用导入 → 检索 → 读原文 → 作答）：

```bash
uv run paperpilot --task \
  "import the papers dir, then find evidence about low-rank updates and quote the exact source"
```

预期行为：Agent 先调用 `import_papers` 导入 `papers/`，再用 `search_papers` 检索，
必要时用 `read_paper_page` 读取原文，最后给出**带页码的引用**。若无命中，会如实说明，
不会编造引用。

需要审计工具调用记录时，可追加 telemetry 输出：

```bash
uv run paperpilot --task "..." --telemetry-out run.json
```

不带 `--task` 时进入交互模式，逐条输入研究问题。

### 6. 直接使用工具（可选）

三个工具也可以作为 Python 函数直接导入调用，用于调试或批量导入。它们不是 `paperpilot` CLI 子命令：

```python
from paperpilot.tools import (
    import_papers,
    search_papers,
    read_paper_page,
)
```

| 工具 | 作用 |
|---|---|
| `import_papers` | 导入目录（或单个 PDF）到本地索引 |
| `search_papers` | 关键词检索，返回 paper_id、页码、原文片段 |
| `read_paper_page` | 按 paper_id + 页码读取整页原文 |

---

## 数据存放位置

| 数据 | 默认位置 |
|---|---|
| 索引数据库 | `<工作目录>/sandbox/paperpilot/index.db` |
| 会话文件 | `<工作目录>/sandbox/paperpilot/session.json` |
| 论文 PDF | `<工作目录>/papers/`（按工作目录相对路径引用） |

**工作目录**默认是启动命令时的当前目录；可用 `MINI_HARNESS_WORK_SPACE` 指定：

```bash
MINI_HARNESS_WORK_SPACE=/path/to/research-space uv run paperpilot --task "..."
```

> **注意**：数据跟着工作目录走。换目录启动会得到不同的索引库。建议为每个研究项目
> 固定一个工作目录，并在该目录下运行 `paperpilot`。

`.gitignore` 已覆盖 `.env`、`sandbox/`、`/papers/`、`session.json`，本地数据不会被误提交。

---

## 隐私边界

**本地保存论文 ≠ 推理全在本地。**

- **完全本地的部分**：PDF 解析、全文索引、检索排序——全程在你的机器上，零外发。
- **会外发的部分**：当你使用**云端模型**时，下列内容会通过 API 发送给模型服务：
  1. 系统提示词（角色与工具说明）
  2. 你输入的研究问题
  3. **工具返回结果**——包括论文标题、页码，以及 `search_papers` 的原文片段和
     `read_paper_page` 的整页正文
  4. 上述内容构成的对话历史

对隐私敏感的论文，请配置**本地模型服务**（vLLM / Ollama 的 OpenAI 兼容模式等），
使推理也留在本机。

---

## 常见问题

**Q: 模型没有调用工具，直接编造了答案？**
该模型可能未正确支持 function calling。换用支持工具调用的模型/服务，并确认
服务端已开启 OpenAI 兼容的 tool_calls 格式。

**Q: 中文论文检索不到？**
PaperPilot 使用 CJK 感知的分词与 NFKC 规范化（兼容汉字可正常检索）。若仍漏检，
可尝试更简洁的关键词组合——检索是词法匹配，不理解同义改写。

**Q: 如何更换模型服务？**
改 `.env` 中的 `MINI_HARNESS_PROVIDER` / `MINI_HARNESS_MODEL` 与对应地址、密钥即可，
无需改动代码。

---

## 开发

```bash
uv run python -m unittest discover -s tests -t .   # 测试
uv run ruff check src tests                        # 静态检查
uv lock --check                                    # 锁文件一致性
```

代码结构见 `src/paperpilot/`（论文工具 + Web 工作台）与 `src/mini_harness/`（Agent harness）。

详细文档：

| 文档 | 内容 |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | 安装、启动、常见问题排查 |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | 全量配置项、模型预算、Provider 对比 |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | 论文导入、数据管理、备份恢复、运维 |
