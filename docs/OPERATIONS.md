# PaperPilot 运维指南

## 一、论文 PDF 导入

### 把 PDF 放进工作目录

```bash
mkdir -p papers
cp /path/to/your/*.pdf papers/
```

### 导入索引

**方式 A：Web 界面**（推荐）

浏览器打开 `http://127.0.0.1:8001/`，在「论文与证据」面板输入目录（默认 `papers`），点击「导入论文」。

**方式 B：CLI**

```bash
uv run paperpilot --task "import the papers dir"
```

**方式 C：Python 函数调用**

```python
from paperpilot.tools import import_papers, ImportPapersInput
import_papers(ImportPapersInput(paper_dir='papers'))
```

### 导入行为

- 逐页抽取文本，SHA-256 变更检测
- 同路径同 SHA → 幂等跳过
- 同路径异 SHA → 删旧重建
- 损坏/加密 PDF → 明确报错，不产生半成品

## 二、数据存放位置

| 数据 | 默认位置 | 是否入 Git |
|---|---|---|
| 论文 PDF | `<工作目录>/papers/` | **否**（`.gitignore` 已忽略） |
| 索引数据库 | `<工作目录>/sandbox/paperpilot/index.db` | **否** |
| CLI 会话 | `<工作目录>/sandbox/paperpilot/session.json` | **否** |
| Web 会话 | `<工作目录>/sandbox/paperpilot/web_sessions/` | **否** |
| 压缩日志 | `<工作目录>/sandbox/paperpilot/mini_harness_history.jsonl` | **否** |
| uv 缓存 | `.cache/uv/` | **否** |

> **这些数据属于本地数据，不应随源码公开。** `.gitignore` 已覆盖 `sandbox/`、`/papers/`、`session.json`。

## 三、测试与检查

### 运行测试

```bash
# 完整测试套件
uv run --locked python -m unittest discover -s tests -t .

# 静态检查
uv run --locked ruff check src tests

# 锁文件一致性
uv lock --check
```

### 检查服务状态

```bash
# 检查 Web 服务是否运行
ss -tlnp | grep 8001

# 检查模型服务
curl http://127.0.0.1:8000/v1/models
```

### 查看执行详情

在 Web 工作台中，每次回答下方的工具轨迹可展开「查看执行详情」，包含工具名、参数、调用与结果的配对关系、真实执行状态。

## 四、备份与恢复

### 备份

```bash
# 备份索引库和会话
cp -r sandbox/paperpilot/ /path/to/backup/

# 备份论文（如果需要）
cp -r papers/ /path/to/backup/papers/
```

### 恢复

```bash
cp -r /path/to/backup/paperpilot/ sandbox/
```

### 清空重新导入

```bash
rm -rf sandbox/paperpilot/index.db
# 然后重新导入论文
```

## 五、常见问题排查

### 模型请求超限（BadRequestError）

**症状**：`maximum context length is ... tokens`

**原因**：输入+输出预算超过模型上下文

**解决**：调整 `.env` 中的 `compact_limit`、`max_tokens_main`，确保两者之和 + 500 协议开销 < `context_limit`。详见 [CONFIGURATION.md](CONFIGURATION.md)。

### 端口被占用

```bash
ss -tlnp | grep 8001  # 查看占用
kill <PID>            # 确认是旧 web 进程后清理
```

### 论文检索不到

- 确认已导入（Web 界面或 CLI 检查论文列表）
- 尝试更简洁的关键词（检索是词法匹配）
- 中文论文使用 CJK 分词，兼容汉字可正常检索

### 模型服务连接失败

```bash
# 测试服务可用性
curl http://127.0.0.1:8000/v1/models

# 检查 .env 配置
cat .env
```

### 会话历史异常

Web 会话存储在 `sandbox/paperpilot/web_sessions/` 下，每个浏览器会话一个目录。清空对话只影响当前会话，不影响其他会话或论文索引。

## 六、隐私与安全

- **完全本地**：PDF 解析、全文索引、检索排序
- **可能外发**：使用云端模型时，系统提示词、研究问题、工具返回结果（论文标题/页码/原文片段）会发送给模型服务
- **敏感文件**：`.env`、API 密钥、模型权重、论文 PDF 不应公开
- **个人笔记**：个人工作记录文件保持本地，不上传
