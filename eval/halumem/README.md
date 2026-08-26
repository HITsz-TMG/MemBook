# Evaluation — Memory Systems with HaluMem

This folder contains the evaluation toolkit for measuring hallucination and memory performance of different memory systems on the **HaluMem** benchmark.

---

## 🚀 Quickstart

```bash
# 1. enter eval folder
cd eval

# 2. install dependencies (Poetry)
poetry install --with eval
```

> If you prefer `pip` or `conda`, install the packages listed in `pyproject.toml` accordingly.

---

## ⚙️ Configuration

1. Copy the example environment file and fill in your own keys and service endpoints:

```bash
cp .env-example .env
```

2. Edit `.env` to include the required configurations for your setup.
   The key parameters are grouped as follows:

#### 🔑 Model Configuration

| Variable             | Description                                     | Example                     |
| -------------------- | ----------------------------------------------- | --------------------------- |
| `OPENAI_API_KEY`     | API key for OpenAI models                       | `sk-xxxx`                   |
| `OPENAI_BASE_URL`    | Base URL for custom OpenAI-compatible endpoints | `https://api.openai.com/v1` |
| `OPENAI_MODEL`       | Model name used for evaluation                  | `gpt-4o`                    |
| `OPENAI_MAX_TOKENS`  | Maximum token limit per request                 | `16384`                     |
| `OPENAI_TEMPERATURE` | Sampling temperature for generation             | `0.0`                       |
| `OPENAI_TIMEOUT`     | Timeout (seconds) for API calls                 | `300`                       |

#### 🔁 Retry Policy

| Variable                              | Description                                 | Example     |
| ------------------------------------- | ------------------------------------------- | ----------- |
| `RETRY_TIMES`                         | Maximum retry attempts                      | `3`         |
| `WAIT_TIME_LOWER` / `WAIT_TIME_UPPER` | Random wait range (seconds) between retries | `10` / `30` |

#### 🧠 Memory Systems

| Variable                                                     | Description                      | Example                                                      |
| ------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------ |
| `MEM0_API_KEY`                                               | API key for Mem0                 | `xxx`                                                        |
| `ZEP_API_KEY`                                                | API key for Zep Memory System    | `xxx`                                                        |
| `MEMOS_KEY`                                                  | Auth token for local MemOS       | `"Token mpg-xxxxx"`                                          |
| `SUPERMEMORY_API_KEY`                                        | API key for Supermemory          | `xxx`                                                        |
| `MEMOS_URL`                                                  | Local MemOS endpoint (optional)  | `"http://127.0.0.1:8001"`                                    |
| `MEMOS_ONLINE_URL`                                           | Online MemOS endpoint            | `"https://memos.memtensor.cn/api/openmem/v1"`                |
| `MEMOBASE_PROJECT_URL`                                       | Memobase project endpoint        | `"http://127.0.0.1:8001"`                                    |
| `MEMOBASE_PROJECT_TOKEN`                                     | Auth token for Memobase          | `"secret"`                                                   |
| `MEMOBASE_DB_HOST` / `PORT` / `USER` / `PASSWORD` / `DB_NAME` | Local DB config for Memobase     | `"127.0.0.1"`, `8002`, `"user_name"`, `"password"`, `"database_name"` |

> 💡 Tip:
>
> * If you're evaluating only one specific memory system (e.g., Mem0), you can omit the configuration settings for all other systems.
> * For local deployments, ensure corresponding services are running before evaluation.

---

## 🔧 Local deployments (Memobase)

For Memobase we rely on local/service deployments. See these projects for install & runtime instructions:

* Memobase: [https://github.com/memodb-io/memobase](https://github.com/memodb-io/memobase)

Make sure the services are running and the corresponding endpoints are reachable before running the evaluation scripts.

---

## 📁 Supported Memory System Wrappers

The repo includes evaluation adapters for multiple memory systems. Filenames map to system wrappers:

* `eval_memzero.py` — Mem0 (default)
* `eval_memzero_graph.py` — Mem0 (Graph variant)
* `eval_memos.py` — MemOS
* `eval_memobase.py` — Memobase
* `eval_zep.py` — Zep
* `eval_supermemory.py` — SuperMemory
* `eval_baseline.py` — Legacy page-summary baseline (`MemoryAgent`)
* `eval_membook.py` — MemBook (full ToC pipeline in `membook.memory.toc`)

Each adapter follows the same input / output contract so the downstream scorer can compare systems fairly.

#### Legacy page-summary baseline 评估

数据统一从 `data/halumem/` 加载，支持 Medium 和 Long 两个数据集：

```bash
# 1. 处理 Medium 数据集
python eval_baseline.py --medium

# 2. 处理 Long 数据集
python eval_baseline.py --long

# 使用 ToC memory backend（fragment 粒度 page）
python eval_baseline.py --medium --memory_agent toc

# 重新测试前删除旧结果
python eval_baseline.py --medium --clean
python eval_baseline.py --long --clean
```

`--memory_agent` 可选：
- `default`（默认）：原始 `MemoryAgent`，session 级 message page
- `toc`：`MemoryAgent`，使用 `membook.memory.toc` 完整 ToC 流水线；默认 `general_qa_mode=nominal`，即 **仅 `user_durable` fragment** 写入 `page_store`（BM25/向量索引与 Planning 目录）；`general_qa` / `interaction_meta` 不进检索页表。nominal 下这两类 lane **不调用** fragment/merge 的 title+summary LLM，滑窗 `PREVIOUS_TOC` 仅含 `user_durable`。Research Planning 使用 `MEMORY_CATALOG`（Session Title + `[k]` fragment title/summary，`[k]` 为**全局** `chapter_index`，跨 session 单调递增）。**与旧版按 session 从 0 编号的 `pages.json` 不兼容**：换代码后请用新结果目录或清空 `pages.json` 后全量重跑。

可选环境变量（与 `tests/test_membook` 一致，节选）：
- `MEMBOOK_GENERAL_QA_FULL_PIPELINE`：由适配器按 `general_qa_mode` 写入（`full`=>1, 其它=>0）；HaluMem 默认 `nominal`（不落盘 `general_qa`）。
- `MEMBOOK_SHIFTS_*`、`MEMBOOK_SESSION_RECONCILE`、`MEMBOOK_RECONCILE_MAX_CHARS` 等：见 `membook/memory/toc/toc_builder.py` 中 `os.getenv`
- `MEMBOOK_BUILD_VERBOSE` / `MEMBOOK_LIVE_LLM`：ToC 构建控制台日志（默认关闭）
- `MEMBOOK_EVAL_VERBOSE`：`eval_membook.MemBookAdapter` 每 session 构建的简要打印（默认关闭）

输出目录：`results/gam-medium/` 或 `results/baseline-long/`。之后运行评估与采样：

```bash
# 2. 评估并汇总
python evaluation.py --frame baseline --version medium   # 或 --version long

# 3. 抽取样本
python sample_cases.py --medium_stat_file results/baseline-medium/baseline_eval_stat_result.json \
  --medium_results_file results/baseline-medium/baseline_eval_results.jsonl \
  --long_stat_file results/baseline-long/gam_eval_stat_result.json \
  --long_results_file results/baseline-long/gam_eval_results.jsonl
```

---

#### MemBook 评估（推荐：一键复现）

MemBook 复用 `eval_baseline` 的整体流程，仅把记忆写入路径替换为完整 ToC 流水线
（`MemoryAgent` + `membook.memory.toc`）。三步即可复现：

**1. 配置环境变量**

只需在 `.env`（从 `.env-example` 复制）里填三项核心配置，memory / research / working 三个 LLM 共用同一套：

```bash
cp .env-example .env
# 编辑 .env：
#   OPENAI_API_KEY=sk-xxxx
#   OPENAI_BASE_URL=https://api.openai.com/v1   # 或任意 OpenAI 兼容端点
#   OPENAI_MODEL=gpt-4o
```

**2. 准备数据**

把 HaluMem 数据放到仓库根目录的 `data/halumem/`（该目录已被 `.gitignore` 忽略，需自行下载）：

```
data/halumem/HaluMem-Medium.jsonl
data/halumem/HaluMem-Long.jsonl
```

**3. 一键运行**

`run_membook.bash` 会依次执行「记忆流水线（`eval_membook.py`）→ 打分汇总（`evaluation.py`）」：

```bash
cd eval/halumem
bash run_membook.bash
```

脚本顶部「可调参数区」是唯一需要改的地方，常用项：

| 参数 | 默认 | 说明 |
| ---- | ---- | ---- |
| `dataset_list` | `("medium")` | 要跑的数据集，可写 `("medium" "long")` |
| `max_users` | `20` | **快速迭代默认只跑 20 个 user；全量复现请改为 `0`** |
| `top_k` | `5` | QA 检索 top_k |
| `model_memory_research` / `model_qa_eval` | `gpt-4o` | 记忆/研究 与 QA/打分 模型 |
| `max_workers` | `10` | 记忆流水线并行 user 进程数 |
| `clean` | `true` | 非时间戳模式下运行前清理旧目录 |
| `append_timestamp` | `true` | 给结果目录加时间戳，避免覆盖旧结果 |

输出目录：`results/membook-{medium|long}[-时间戳]/`，最终结果见 `membook_eval_results.jsonl`；LLM 消耗见 `membook_llm_usage.json`（记忆+Research+QA，不含向量索引与打分），打分汇总见 `membook_eval_stat_result.json` 的 `overall_score.llm_usage`。

> 想脱离脚本手动跑：
> ```bash
> python eval_membook.py --medium --max_users 0 --top_k 5 --model gpt-4o --max_workers 10
> python evaluation.py --frame membook --version medium
> ```

**关键 `MEMBOOK_*` 环境变量**（默认值即推荐配置，一般无需改动）：

| 变量 | 默认 | 说明 |
| ---- | ---- | ---- |
| `MEMBOOK_EVAL_EXTRACT_MODE` | `confirmed_list` | 写入 HaluMem `extracted_memories` 的粒度：`confirmed_list`（每条确认记忆一行）/ `session_summary` / `chapters` |
| `MEMBOOK_PENDING_IN_RETRIEVAL` | 关 | 开启后在抽取结果末尾追加一行 `[PENDING] ...`（与 QA 索引对齐） |
| `MEMBOOK_GENERAL_QA_FULL_PIPELINE` | `0` | 由适配器按 `general_qa_mode` 设置；`nominal` 时仅 `user_durable` fragment 进检索页表 |
| `MEMBOOK_USE_PAGE_INDEX` | `1` | 按 `MEMORY_CATALOG` 的 `[k]` 整页/确认点读取；`0` 时仅 keyword/vector |
| `MEMBOOK_USE_BM25` / `MEMBOOK_USE_DENSE_RETRIEVER` | `1` | 是否构建 BM25 / 向量检索器 |
| `MEMBOOK_QA_INTEGRATE_MAX_HITS` | `20` | 合并去重后传入 integrate 的记忆条数上限 |
| `MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE` / `_STEP` / `_ENABLE_MIN_TURNS` | `15` / `10` / `15` | ToC 动态切分滑窗参数 |
| `MEMBOOK_EVAL_VERBOSE` / `MEMBOOK_BUILD_VERBOSE` | 关 | 每 session 构建的调试打印 |
| `MEMBOOK_USAGE_TIKTOKEN_ENCODING` | `cl100k_base` | API 未返回 `usage` 时 tiktoken 兜底编码 |

**LLM 消耗统计**（`eval_membook.py` 阶段，不含向量 embedding 与 `evaluation.py` 打分）：

- 每 user 的 `llm_usage` 写入 `membook_eval_results.jsonl`
- Run 级汇总：`membook_llm_usage.json`（`api_calls`、`prompt_tokens`、`completion_tokens`、`total_tokens`）
- 打分后 `membook_eval_stat_result.json` → `overall_score.llm_usage` 从 jsonl 再聚合
- API 无 `usage` 时用 tiktoken 估算，见 `usage_missing_calls` / `token_source`

> ⚠️ 全局 `[k]` 章节编号（跨 session 单调递增）与旧版按 session 从 0 编号的 `pages.json` **不兼容**。换代码后请用新结果目录，或清空 `pages.json` 后全量重跑。
>
> 更多变量见 `membook/memory/toc/toc_builder.py` 中的 `os.getenv`，以及 `eval/halumem/eval_membook.py` 顶部 docstring。

---

## 🧪 Running the Evaluation

To evaluate memory systems using **HaluMem**, execute the following commands step by step:

1. Generate run artifacts (extract memories & generate predictions): using **Mem0** as an example, first run the following command to process dialogue input, extract memory points, and perform QA retrieval.

   ```bash
   python eval_memzero.py
   ```

   Please set data_path and version in eval_memzero.py as needed:
   * `data_path` : path to dataset
   * `version` : evaluation version identifier (used to tag outputs)

   Specifically, for Zep, due to its asynchronous design, we provide an additional `run_task` argument in `eval_zep.py` to specify either the dialogue addition task (`run_task="add"`) or the memory retrieval task (`run_task="search"`, which includes both memory update and QA retrieval). The dialogue addition task must be executed first, and once the dialogue processing is complete (progress can be monitored via threads and episodes on the official platform), the memory retrieval task can then be performed.

2. Score & aggregate results: run the following command to evaluate **Mem0** on memory extraction, memory update, and memory QA tasks, and aggregate the results.

   ```bash
   python evaluation.py --frame memzero --version default
   ```

   * `--frame` : which adapter/system frame to evaluate (e.g., `memzero`, `memos`, `memobase`)
   * `--version` : same identifier used during run generation

3. Outputs: the final results will be saved in the `results` folder.
   

These artifacts enable reproducibility and further analysis (error inspection, evidence linking, per-memory-point breakdowns).

---

## Special Configurations for Some Memory Systems

While the experimental setup strives to maintain consistent configurations across all evaluated systems, certain memory systems exhibit unique API constraints that necessitate specific adjustments or workarounds.
Each subsection below outlines these system-specific configurations to ensure reproducibility.

#### Memobase

Since Memobase does not provide a Get Dialogue Memory API, we adopted a localized deployment approach and directly accessed the corresponding dialogue memories from its underlying database. Additionally, the Retrieve Memory API of Memobase only supports controlling the maximum length of the returned memory text. Based on test results, we set the maximum length for memory recall in the memory updating task to 250 tokens and the recall length for the memory question answering task to 500 tokens.

#### Zep

According to our current understanding, the official APIs provided by Zep do not support retrieving all memory points within a specific session, meaning they do not offer functionality equivalent to a Get Dialogue Memory API. Consequently, we were unable to evaluate Zep’s performance on the memory extraction task. We attempted to use the function `thread.get_user_context()` offered by Zep to obtain all memories under a given thread; however, this method only returns recent memories rather than the complete set, which does not meet the evaluation requirements. Moreover, since Zep’s memory processing workflow operates entirely asynchronously, we could not accurately measure the time consumption in the dialogue addition phase and instead recorded only the time cost associated with memory retrieval.

Therefore, the evaluation metrics of the *memory extraction* task (`memory_accuracy` and `memory_recall`) and the *dialogue addition time* obtained after running `eval_zep.py` and `evaluation.py` cannot accurately reflect the actual performance of Zep.

---
