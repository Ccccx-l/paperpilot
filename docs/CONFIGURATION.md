# PaperPilot 配置指南

所有配置项以源码 `src/mini_harness/config.py` 的实际实现为准。
配置通过环境变量或 `.env` 文件传入（环境变量优先于 `.env`）。

`.env` 由 `python-dotenv` 自动加载（`find_dotenv(usecwd=True)` 从当前目录向上查找）。

---

## 一、核心配置（必需）

### MINI_HARNESS_MODEL

| 项目 | 说明 |
|---|---|
| 作用 | 模型名称 |
| 类型 | 字符串 |
| 默认值 | 无（必须设置） |
| 必需 | **是** |
| 示例 | `qwen3-vl-8b`、`deepseek-v4-flash`、`gpt-4.1` |

也可用 `provider/model` 形式（如 `local/qwen3-vl-8b`），此时 provider 从模型名前缀推断。

### MINI_HARNESS_PROVIDER

| 项目 | 说明 |
|---|---|
| 作用 | 模型服务商类型 |
| 可选值 | `openai` / `deepseek` / `local` |
| 默认值 | 按 `MINI_HARNESS_MODEL` 前缀推断 |
| 必需 | 建议显式设置 |

### API 密钥（按服务商）

| 环境变量 | 优先级 | 适用 |
|---|---|---|
| `MINI_HARNESS_API_KEY` | 最高 | 通用 |
| `{PROVIDER}_API_KEY`（如 `DEEPSEEK_API_KEY`、`OPENAI_API_KEY`、`LOCAL_API_KEY`） | 次之 | 对应服务商 |
| `MINI_HARNESS_API_KEY_ENV` | 可选 | 自定义密钥变量名 |

**本地模型服务**：通常不需要真实密钥（`LOCAL_API_KEY=EMPTY`）。
**云端 API**：必须设置真实密钥，否则请求会被拒绝。

---

## 二、模型服务地址

### OPENAI_BASE_URL

| 项目 | 说明 |
|---|---|
| 作用 | `provider=openai` 时的 API 地址 |
| 默认值 | `https://api.openai.com/v1` |
| 必需 | 按需 |

### LOCAL_BASE_URL

| 项目 | 说明 |
|---|---|
| 作用 | `provider=local` 时的 API 地址 |
| 默认值 | `http://127.0.0.1:8000/v1` |
| 必需 | 按需 |

适用于 vLLM、Ollama（OpenAI 兼容模式）、llama.cpp server 等本地模型服务。

---

## 三、工作目录

### MINI_HARNESS_WORK_SPACE

| 项目 | 说明 |
|---|---|
| 作用 | 工作目录，决定所有数据的存放位置 |
| 类型 | 路径 |
| 默认值 | 启动命令时的当前目录 |
| 必需 | 可选 |

工作目录下的数据结构：

```
<工作目录>/
├── papers/          # 论文 PDF（用户放入）
├── sandbox/
│   └── paperpilot/
│       ├── index.db          # SQLite 索引库
│       ├── session.json      # CLI 会话文件
│       └── web_sessions/     # Web 会话数据
```

---

## 四、模型上下文与预算

以下参数决定请求的输入/输出预算。**必须满足：输入预算 + 输出预算 + 协议开销 < 模型真实上下文长度**。

### context_limit

| 项目 | 说明 |
|---|---|
| 作用 | 模型真实总上下文长度（输入 + 输出） |
| 类型 | 整数 |
| 默认值 | `0`（未知 / 不强制） |
| 本地默认 | `16384`（vLLM `--max-model-len 16384`） |

### compact_limit

| 项目 | 说明 |
|---|---|
| 作用 | 触发上下文压缩的输入 token 阈值 |
| 类型 | 整数 |
| 默认值 | `300000` |
| 本地默认 | `9000`（须远小于 context_limit） |

### max_tokens_main

| 项目 | 说明 |
|---|---|
| 作用 | 单次请求最大输出 token 数 |
| 类型 | 整数 |
| 默认值 | `100000` |
| 本地默认 | `4096` |

### max_tokens_sub

| 项目 | 说明 |
|---|---|
| 作用 | 子任务（压缩摘要等）最大输出 token 数 |
| 类型 | 整数 |
| 默认值 | `50000` |
| 本地默认 | `2048` |

### 参数关系（以本地 vLLM 16384 为例）

```
compact_limit(9000) + max_tokens_main(4096) + 协议开销(~500) = 13596 < 16384 ✓
```

**预算配置示例**：

| 场景 | context_limit | compact_limit | max_tokens_main | max_tokens_sub |
|---|---|---|---|---|
| vLLM `--max-model-len 16384` | 16384 | 9000 | 4096 | 2048 |
| vLLM `--max-model-len 32768` | 32768 | 20000 | 8192 | 4096 |
| 云端 DeepSeek（较大上下文） | 0 | 300000 | 100000 | 50000 |

---

## 五、重试配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MINI_HARNESS_MAX_RETRY` | 5 | 非限流异常重试次数 |
| `MINI_HARNESS_RATE_RETRY` | 6 | 限流异常重试次数 |

---

## 六、其他配置

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `MINI_HARNESS_PROFILE` | 设为 `bench` 时启用基准测试配置 | 无 |
| `MINI_HARNESS_SUB_MODEL` | 子任务使用不同模型 | 同 `MINI_HARNESS_MODEL` |
| `MINI_HARNESS_REASONING_EFFORT` | 推理努力级别（OpenAI provider） | `default` |

---

## 七、开发机专用 vs 通用配置

| 配置 | 适用范围 | 说明 |
|---|---|---|
| `LOCAL_BASE_URL=http://127.0.0.1:8000/v1` | 仅开发机 | 需按实际 vLLM 端口调整 |
| `LOCAL_API_KEY=EMPTY` | 本地开发 | 本地服务通常无需真实密钥 |
| `context_limit=16384` | 仅当前 vLLM 部署 | 按 `--max-model-len` 调整 |
| `compact_limit=9000` / `max_tokens_main=4096` | 仅当前预算 | 按实际模型调整 |
| `MINI_HARNESS_WORK_SPACE` | 仅当前机器 | 路径因环境而异 |

**不要**将开发机的本地路径、端口、预算值作为通用默认值提交。
