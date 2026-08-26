#!/usr/bin/env bash
# LongMemEval-S (cleaned) + HaluMem MYMEMAdapter：生成 hypotheses.jsonl，再可选调用官方 evaluate_qa 判分。
#
# 数据：默认使用项目根下 data/longmemeval/（可将参考仓库同名目录拷入；本仓库 .gitignore 通常忽略 data/）。
#
# 用法：
#   export OPENAI_API_KEY='...'
#   export OPENAI_BASE_URL='...'   # 可选
#   ./scripts/eval_longmemeval_membook.sh
#
# 默认：MYMEM（记忆/检索/生成 hypothesis）用 gpt-4o-mini；判分（evaluate_qa）用 gpt-4o。
#   生成子进程会强制 OPENAI_MODEL=MEMBOOK_MODEL，避免与判分模型混用。
#
# 可选环境变量：
#   MEMBOOK_MODEL=gpt-4o-mini   # MYMEM + hypothesis 生成（默认 gpt-4o-mini）
#   METRIC_MODEL=gpt-4o       # LLM judge 短名，须 ∈ evaluate_qa.py 的 model_zoo
#   EVAL_QA_MODEL=gpt-4o      # judge 请求体里的实际 model；默认与 METRIC_MODEL 相同
# Reading 默认 hybrid：temporal-reasoning(133) 用 CoT，其余 5 类用 plain（一次 API/题，直接写入 hypotheses.jsonl）
#   LONGMEMEVAL_READING_MODE=hybrid|cot|plain  或  --reading-mode（透传 longmemeval_membook.py）
# 查看示例：python3 eval/baseline/longmemeval_membook.py --show-reading-prompt
# 事后合并两次判分（旧流程 CoT+noncot 各跑一遍）仍可用：scripts/merge_longmemeval_hybrid_metrics.py
# 每条打印 gold/hypothesis：默认 --print-answers；关闭：LONGMEMEVAL_PRINT_ANSWERS=0 或 --no-print-answers
#   RUN_JUDGE=0          只生成 hypotheses
#   判分依赖：LongMemEval 的 evaluate_qa.py 需要 Python 包 backoff（已写入 requirements.txt）。
#   若未安装，脚本在判分前会尝试：${PYTHON} -m pip install -q 'backoff>=2.2.0'
#   也可手动：${PYTHON} -m pip install 'backoff>=2.2.0'
#   MAX_ITEMS=5          只跑前 5 条（调试）
#   START=0
#   TOP_K=20
#   RESUME=1             传 --resume（memory_point_id 改为 {session_id}/mp_n 后勿 resume 旧 mymem_state）
#   OUT_DIR=...          自定义结果目录（默认 results/longmemeval_membook-<模型slug>-<时间戳>）
#   WORKERS=4            并发样本数；MYMEM ToC 较重，默认较低（见下 LONGMEMEVAL_DEFAULT_WORKERS）
# ToC 动态滑窗 + 窗内 reconcile（见 gam/memory/toc/toc_builder.py；session 轮数 < min_turns 时仍为单窗 reconcile）：
#   MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS=15
#   MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE=15
#   MEMBOOK_SHIFTS_SLIDING_STEP=10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

: "${OPENAI_API_KEY:?请先 export OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"

# 与 eval/halumem/run_membook.bash、eval_longmemeval_membook_dashscope_qwen.sh 对齐；可在命令行前覆盖。
export MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS="${MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS:-15}"
export MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE="${MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE:-20}"
export MEMBOOK_SHIFTS_SLIDING_STEP="${MEMBOOK_SHIFTS_SLIDING_STEP:-15}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

MEMBOOK_MODEL="${MEMBOOK_MODEL:-gpt-4o-mini}"
METRIC_MODEL="${METRIC_MODEL:-gpt-4o}"
EVAL_QA_MODEL="${EVAL_QA_MODEL:-${METRIC_MODEL}}"

# MYMEM ToC + 检索比纯 GAM 更重：默认并发保守；可用 WORKERS=8 ./scripts/eval_longmemeval_membook.sh 覆盖
LONGMEMEVAL_DEFAULT_WORKERS="${LONGMEMEVAL_DEFAULT_WORKERS:-20}"
WORKERS="${WORKERS:-${LONGMEMEVAL_DEFAULT_WORKERS}}"
LONGMEMEVAL_READING_MODE="${LONGMEMEVAL_READING_MODE:-hybrid}"
# LongMemEval: single-session-assistant 自动启用 general_qa full pipeline；
# 其他题型保持 nominal（不落盘 general_qa chapter/MP）。

TS="$(date +%Y%m%d_%H%M%S)"
# 默认目录名带上模型，便于多模型结果并列对比（非法路径字符压成下划线）
LONGMEMEVAL_MODEL_SLUG="$(printf '%s' "${MEMBOOK_MODEL}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
OUT_DIR="${OUT_DIR:-${ROOT}/results/longmemeval_membook-${LONGMEMEVAL_MODEL_SLUG}-${TS}}"
mkdir -p "${OUT_DIR}"

HYP="${OUT_DIR}/hypotheses.jsonl"
REF="${ROOT}/data/longmemeval/longmemeval_s_cleaned.json"
EVAL_DIR="${ROOT}/data/longmemeval/LongMemEval-main/src/evaluation"
EVAL_QA="${EVAL_DIR}/evaluate_qa.py"
PRINT_METRICS="${EVAL_DIR}/print_qa_metrics.py"

EXTRA=()
if [[ -n "${MAX_ITEMS:-}" ]]; then
  EXTRA+=(--max_items "${MAX_ITEMS}")
fi
if [[ -n "${START:-}" ]]; then
  EXTRA+=(--start "${START}")
fi
if [[ -n "${TOP_K:-}" ]]; then
  EXTRA+=(--top_k "${TOP_K}")
fi
if [[ "${RESUME:-}" == "1" ]]; then
  EXTRA+=(--resume)
fi
EXTRA+=(--workers "${WORKERS}")
EXTRA+=(--reading-mode "${LONGMEMEVAL_READING_MODE}")

cmd=(
  "${PYTHON}" eval/baseline/longmemeval_membook.py
  --data "${REF}"
  --out "${HYP}"
  --save_path "${OUT_DIR}/mymem_state"
  --model "${MEMBOOK_MODEL}"
  --print-answers
)
if ((${#EXTRA[@]} > 0)); then
  cmd+=("${EXTRA[@]}")
fi
if [[ "${LONGMEMEVAL_PRINT_ANSWERS:-}" =~ ^(0|false|no|off)$ ]]; then
  cmd+=(--no-print-answers)
fi

_shifts_step="${MEMBOOK_SHIFTS_SLIDING_STEP:-10}"
echo "🚀 生成：python=${PYTHON} | model=${MEMBOOK_MODEL} | workers=${WORKERS} | reading=${LONGMEMEVAL_READING_MODE} | out=${HYP}"
echo "   ToC shifts: min_turns=${MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS} window=${MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE} step=${_shifts_step} overlap=$((MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE - _shifts_step))"
if [[ ! -f "${REF}" ]]; then
  echo "❌ 未找到数据文件：${REF}" >&2
  echo "   请将 longmemeval_s_cleaned.json 与 LongMemEval-main 放到 data/longmemeval/，或设置 OUT_DIR 前先用 --data 指向有效路径（需自行改本脚本 REF 或命令行）。" >&2
  exit 1
fi

env OPENAI_MODEL="${MEMBOOK_MODEL}" "${cmd[@]}"

echo ""
echo "📄 Hypotheses: ${HYP}"
echo "📄 Errors (if any): ${OUT_DIR}/hypotheses.errors.jsonl"

RUN_JUDGE="${RUN_JUDGE:-1}"
if [[ "${RUN_JUDGE}" == "1" ]]; then
  if [[ ! -f "${EVAL_QA}" ]]; then
    echo "❌ 未找到 ${EVAL_QA}，跳过判分。"
    exit 0
  fi
  if ! "${PYTHON}" -c "import backoff" 2>/dev/null; then
    echo "📦 未检测到 backoff（evaluate_qa.py 需要），正在安装…"
    "${PYTHON}" -m pip install -q 'backoff>=2.2.0'
  fi
  echo ""
  echo "⚖️  判分：evaluate_qa.py + metric「${METRIC_MODEL}」/ request_model「${EVAL_QA_MODEL}」（MYMEM 已用「${MEMBOOK_MODEL}」）"
  (
    cd "${EVAL_DIR}"
    env EVAL_QA_MODEL="${EVAL_QA_MODEL}" "${PYTHON}" evaluate_qa.py "${METRIC_MODEL}" "${HYP}" "${REF}"
  )
  RESULT_JSONL="${HYP}.eval-results-${METRIC_MODEL}"
  if [[ -f "${PRINT_METRICS}" && -f "${RESULT_JSONL}" ]]; then
    echo ""
    echo "📊 按题型 / 总体（print_qa_metrics.py）"
    "${PYTHON}" "${PRINT_METRICS}" "${RESULT_JSONL}" "${REF}" || true
  fi
  echo ""
  echo "📄 带 autoeval_label 的逐条结果: ${RESULT_JSONL}"
fi
