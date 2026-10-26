# PaperPilot 安装指南

## 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux、macOS 或 Windows（建议 Linux） |
| Python | **≥ 3.12**（项目 `.python-version` 固定 3.12） |
| 包管理器 | [uv](https://docs.astral.sh/uv/)（推荐，锁定依赖版本） |
| 磁盘空间 | 依赖约 100MB；论文 PDF 与索引库按实际数据量 |
| 模型服务 | 支持 OpenAI 兼容 function calling（tool_calls）的模型 |

## 安装 uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装后验证：

```bash
uv --version
# 预期输出: uv 0.x.x ...
```

## 从干净环境安装

```bash
# 1. 克隆仓库
git clone <本仓库地址>
cd paperpilot

# 2. 安装依赖（使用锁文件，确保版本一致）
uv sync --locked

# 3. 验证安装
uv run paperpilot --help
```

`uv sync --locked` 会严格按照 `uv.lock` 安装依赖（`openai==2.45.0`、`httpx==0.28.1`、`pydantic==2.13.4`、`python-dotenv==1.2.2`、`pypdf==6.21.2`）。

## 配置模型服务

```bash
cp .env.example .env
# 用编辑器打开 .env，填入你的模型服务配置
```

配置两种典型形态：

### 形态 A：本地模型（推荐隐私敏感场景）

以本地 vLLM 服务为例：

```bash
MINI_HARNESS_PROVIDER=local
MINI_HARNESS_MODEL=qwen3-vl-8b
LOCAL_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_API_KEY=EMPTY
```

### 形态 B：云端 API

以 DeepSeek 为例：

```bash
MINI_HARNESS_PROVIDER=deepseek
MINI_HARNESS_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=sk-你的密钥
```

详细配置项说明见 [CONFIGURATION.md](CONFIGURATION.md)。

## 启动 PaperPilot Web 服务

```bash
uv run python -m paperpilot.web
```

启动后输出：

```
[paperpilot-web]: http://127.0.0.1:8001
[paperpilot-web]: work_space = <你的工作目录>
[paperpilot-web]: db = <工作目录>/sandbox/paperpilot/index.db
[paperpilot-web]: web_sessions = <工作目录>/sandbox/paperpilot/web_sessions
```

浏览器打开 **http://127.0.0.1:8001/** 即可使用工作台。

### 指定工作目录

```bash
MINI_HARNESS_WORK_SPACE=/path/to/research-space uv run python -m paperpilot.web
```

> **注意**：数据跟着工作目录走。换目录启动会得到不同的索引库。

## 启动 CLI 模式

```bash
# 单次任务
uv run paperpilot --task "import the papers dir, then find evidence about LoRA"

# 交互模式（不带 --task）
uv run paperpilot
```

## 停止服务

在运行服务的终端按 `Ctrl+C`。

## 常见问题排查

### 端口 8001 被占用

```bash
# 查看占用进程
ss -tlnp | grep 8001

# 如果是旧的 paperpilot.web 进程，确认后 kill
# 注意：不要动 8000 端口的 vLLM 服务
```

### 依赖安装失败

```bash
# 确认 Python 版本
python3 --version  # 应 ≥ 3.12

# 清理缓存后重试
uv cache clean
uv sync --locked
```

### 模型连接失败

```bash
# 测试模型服务是否可用（以本地 vLLM 为例）
curl http://127.0.0.1:8000/v1/models

# 检查 .env 配置
cat .env
```

### 中文论文检索不到

PaperPilot 使用 CJK 感知分词。若仍漏检，尝试更简洁的关键词组合。
详细运维指南见 [OPERATIONS.md](OPERATIONS.md)。
