# Qwen API Calling - 围棋棋局语义标注工具

使用 Qwen API 自动生成围棋对局的语义描述，为棋局步骤添加结构化的中文语义标注。

## 功能特性

- **生成 Prompt** - 创建 API 请求的 prompt 和 payload（不调用 API）
- **单文件处理** - 为单个棋局步骤文件调用 API 生成语义描述
- **批量处理** - 批量处理整个目录的棋局步骤文件
- **重试机制** - 支持失败自动重试
- **进度显示** - 实时显示处理进度
- **试运行模式** - 不调用 API，仅复制文件用于测试

## 安装依赖

```bash
pip install requests
```

## 快速开始

### 1. 查看帮助

```bash
python3 simpleproto_qwen_tool.py --help
```

### 2. 测试运行（推荐）

先处理少量文件测试是否正常：

```bash
python3 simpleproto_qwen_tool.py fill-batch \
  --input-root ./simpleproto_from_sgf \
  --output-root ./prompted_json \
  --limit-files 5
```

### 3. 完整批量处理

处理所有待处理文件：

```bash
python3 simpleproto_qwen_tool.py fill-batch \
  --input-root ./simpleproto_from_sgf \
  --output-root ./prompted_json \
  --model qwen-plus \
  --temperature 0.2 \
  --retries 1 \
  --retry-wait 1.5
```

## 使用方法

### 模式一：生成 Prompt（不调用 API）

生成 API 请求的 prompt 和 payload，用于调试或测试：

```bash
python3 simpleproto_qwen_tool.py prompt \
  --input-json ./simpleproto_from_sgf/2025-06-01a/step_0001.json \
  --output-prompt ./api_prompt.txt \
  --output-payload ./api_payload.json
```

**参数：**
- `--input-json`: 输入的 step JSON 文件路径（必选）
- `--output-prompt`: 输出 prompt 文件路径（默认：`./api_prompt.txt`）
- `--output-payload`: 输出 payload 文件路径（默认：`./api_payload.json`）
- `--model`: 模型名称（默认：`qwen-plus`）
- `--temperature`: 温度参数（默认：`0.2`）

### 模式二：处理单个文件

为单个棋局步骤文件调用 API 生成语义描述：

```bash
python3 simpleproto_qwen_tool.py fill-one \
  --input-json ./simpleproto_from_sgf/2025-06-01a/step_0001.json \
  --output-json ./output.json \
  --model qwen-plus \
  --temperature 0.2
```

**参数：**
- `--input-json`: 输入的 step JSON 文件路径（必选）
- `--output-json`: 输出文件路径（默认：覆盖输入文件）
- `--model`: 模型名称（默认：`qwen-plus`）
- `--temperature`: 温度参数（默认：`0.2`）
- `--timeout`: 超时时间（秒，默认：`120`）
- `--api-key`: API 密钥（可选，默认使用代码中的密钥或环境变量）
- `--base-url`: API 基础 URL（默认：阿里云 DashScope）

### 模式三：批量处理（推荐）

批量处理整个目录的棋局步骤文件：

```bash
python3 simpleproto_qwen_tool.py fill-batch \
  --input-root ./simpleproto_from_sgf \
  --output-root ./prompted_json \
  --model qwen-plus \
  --temperature 0.2 \
  --retries 1 \
  --retry-wait 1.5
```

**参数：**
- `--input-root`: 输入目录路径（默认：`./simpleproto_from_sgf`）
- `--output-root`: 输出目录路径（默认：`./prompted_json`）
- `--model`: 模型名称（默认：`qwen-plus`）
- `--temperature`: 温度参数（默认：`0.2`）
- `--timeout`: 超时时间（秒，默认：`120`）
- `--api-key`: API 密钥（可选）
- `--retries`: 失败重试次数（默认：`1`）
- `--retry-wait`: 重试等待时间（秒，默认：`1.5`）
- `--limit-files`: 限制处理文件数量（用于测试，默认：不限制）
- `--dry-run`: 试运行模式（不调用 API，仅复制文件）

### 试运行模式

不调用 API，仅复制文件用于测试文件结构：

```bash
python3 simpleproto_qwen_tool.py fill-batch \
  --input-root ./simpleproto_from_sgf \
  --output-root ./prompted_json \
  --dry-run
```

## API 密钥配置

API 密钥可以通过以下方式提供（优先级从高到低）：

1. 命令行参数 `--api-key`
2. 环境变量 `DASHSCOPE_API_KEY`
3. 代码中的默认值 `USER_API_KEY`

```bash
# 方式一：命令行参数
python3 simpleproto_qwen_tool.py fill-batch --api-key "your-api-key"

# 方式二：环境变量
export DASHSCOPE_API_KEY="your-api-key"
python3 simpleproto_qwen_tool.py fill-batch
```

## 目录结构

```
QWEN_API_Calling/
├── simpleproto_qwen_tool.py    # 主程序
├── simpleproto_from_sgf/       # 输入目录（从 SGF 转换的棋局数据）
│   ├── games_map.json
│   ├── 2025-06-01a/
│   │   ├── index.json
│   │   ├── step_0001.json
│   │   ├── step_0002.json
│   │   └── ...
│   └── ...
└── prompted_json/              # 输出目录（经过 API 填充语义描述）
    ├── games_map.json
    ├── 2025-06-01a/
    │   ├── index.json
    │   ├── step_0001.json
    │   └── ...
    └── _failures.log           # 失败日志
```

## 输出格式说明

每个棋局步骤会被补充以下语义信息：

### semantic_description（全局语义描述）

```json
{
  "global_tags": ["战斗", "中盘", "黑棋优势"],
  "strategic_focus": ["左上角战斗", "中腹势力争夺"],
  "global_summary_cn": "黑棋在左上角发起进攻，白棋防守并寻找反击机会",
  "phase_explanation": "中盘阶段，双方在局部展开激烈战斗"
}
```

### regions（九宫格区域描述）

包含 9 个区域（`top_left`, `top_center`, `top_right`, `middle_left`, `center`, `middle_right`, `bottom_left`, `bottom_center`, `bottom_right`）：

```json
{
  "region_id": "top_left",
  "summary": "黑棋形成外势，白棋做活",
  "shapes": ["小飞", "跳"],
  "local_tags": ["进攻", "防守"],
  "key_points": ["D4", "Q16"],
  "group_status": ["黑棋优势", "白棋受攻"]
}
```

## 注意事项

1. **首次使用建议**：先用 `--dry-run` 测试文件结构，再用 `--limit-files 5` 测试少量文件
2. **API 调用限制**：请注意 API 的调用频率限制
3. **失败处理**：失败的文件会记录在输出目录的 `_failures.log` 中
4. **输出覆盖**：使用 `fill-one` 时如果不指定 `--output-json`，会覆盖原文件
5. **坐标格式**：所有坐标统一使用 GTP 格式（如 D4、Q16、PASS）

## 当前数据统计

| 目录 | step 文件数量 | 状态 |
|------|-------------|------|
| simpleproto_from_sgf | 1,071 | 待处理 |
| prompted_json | 2 | 已处理 |

## 许可证

MIT License