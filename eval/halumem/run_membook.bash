#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 优先用项目 .venv（Python 3.10+）；避免误用系统/base 的 3.8
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

########################
# 可调参数区（优先改这里）
########################



# 要跑的数据集：按数组顺序依次执行；每个数据集会先跑完记忆流水线再打分，再进入下一个。
# 正确写法必须是两个元素，不要写成 ("medium long")（那会变成一个非法名字）。
# dataset_list=("medium")
dataset_list=("medium" "long")

# QA 检索 top_k（越大召回更高，成本也更高）
top_k=5
# QA：合并去重排序后、传入 integrate 之前的记忆条数上限（ResearchAgent 读 MEMBOOK_QA_INTEGRATE_MAX_HITS）
qa_integrate_max_hits=20
# 记忆+研究阶段模型
model_memory_research="gpt-4o"
# QA 作答 + evaluation 打分模型（llms.py 读取 OPENAI_MODEL）
model_qa_eval="gpt-4o"
# 输出目录前缀：results/{frame}-{version}/
frame="membook"
# 自定义保存路径；留空则走默认 results/{frame}-{version}
save_path=""
# 非时间戳模式下，运行前是否清理旧目录
clean=true
# 并行处理 user 的进程数（eval_membook 跑记忆流水线）
max_workers=10
# 打分阶段 evaluation.py：同时评测多少个 user（与上面独立）
eval_user_workers=4
# ResearchAgent 常规阶段最大反思/迭代轮数
research_max_iters=1
# ResearchAgent 展开到原文最大迭代轮数
research_max_deep_iters=0
# 记忆更新评测（memory update）检索 top_k
update_retrieve_top_k=10
# 快速迭代：只跑 N 个 user（<=0 表示全量）
max_users=20
# PageIndex：按 MEMORY_CATALOG 的 [k] 整页/确认点读取；0 时仅 keyword/vector（与 ResearchAgent 一致）
use_page_index=1
# 跳过前 N 个 user（与 max_users 组合可切片；例如 15+5 表示第 16-20）
user_offset=0
# 只跑指定 uuid；留空表示全部
target_uuid=""
# true 时给 version 加统一时间戳，避免覆盖旧结果
append_timestamp=true

# 断点续跑：指向已有结果目录（留空则新开一轮）
resume=false
resume_dir=""

# 日志：true=终端+写入 logs/mymem_<时间戳>.log；false=仅终端
save_log=true
# 更详细的 MEMBOOK 构建/评测打印（fragment title+summary 等）；并行时日志会交错
verbose_logs=false

# ===== ToC 动态切分滑窗参数 =====
# 对话长度达到该阈值后启用动态滑窗切分
shifts_sliding_enable_min_turns=20
# 话题探测窗口长度（turn）
shifts_sliding_window_size=20
# 滑窗步长（turn）
shifts_sliding_step=15

########################
# 环境：加载 .env、实时日志、可选 verbose
########################
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

export PYTHONUNBUFFERED=1
if [[ "$verbose_logs" == true ]]; then
  export MEMBOOK_EVAL_VERBOSE=1
  export MEMBOOK_BUILD_VERBOSE=1
  unset MEMBOOK_LOCOMO_QUIET_BUILD 2>/dev/null || true
else
  unset MEMBOOK_EVAL_VERBOSE MEMBOOK_BUILD_VERBOSE 2>/dev/null || true
  export MEMBOOK_LOCOMO_QUIET_BUILD=1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
if [[ "$save_log" == true ]]; then
  mkdir -p "$SCRIPT_DIR/logs"
  LOG_FILE="$SCRIPT_DIR/logs/mymem_${RUN_TS}.log"
  echo "[run_membook.bash] logging to ${LOG_FILE}"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

########################
# QA 与打分模型（llms.py 读取 OPENAI_MODEL）
########################
export OPENAI_MODEL="${OPENAI_MODEL:-$model_qa_eval}"
export MEMBOOK_QA_INTEGRATE_MAX_HITS="${MEMBOOK_QA_INTEGRATE_MAX_HITS:-$qa_integrate_max_hits}"
export MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS="${MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS:-$shifts_sliding_enable_min_turns}"
export MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE="${MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE:-$shifts_sliding_window_size}"
export MEMBOOK_SHIFTS_SLIDING_STEP="${MEMBOOK_SHIFTS_SLIDING_STEP:-$shifts_sliding_step}"
export MEMBOOK_USE_PAGE_INDEX="${MEMBOOK_USE_PAGE_INDEX:-$use_page_index}"
# general_qa 记忆点抽取与 BM25/向量索引默认关闭（仅 user_durable 进 page_store）。需开启时取消下一行 unset 并 export MEMBOOK_GENERAL_QA_FULL_PIPELINE=1
unset MEMBOOK_GENERAL_QA_FULL_PIPELINE 2>/dev/null || true

if [[ "$append_timestamp" == true ]]; then
  echo "[run_membook.bash] append_timestamp=true → 结果将写入 results/${frame}-{medium|long}-${RUN_TS}/"
fi

if [[ "$resume" == true ]]; then
  if [[ -z "${resume_dir:-}" || ! -d "$resume_dir" ]]; then
    echo "[run_membook.bash] resume=true 需要有效 resume_dir 目录"
    exit 1
  fi
  save_path="$resume_dir"
  append_timestamp=false
  clean=false
  resume_version="$(basename "$resume_dir")"
  resume_version="${resume_version#${frame}-}"
  echo "[run_membook.bash] resume=true → 续跑目录: ${save_path} (version=${resume_version})"
fi

# 超长单行 echo 易被编辑器/换行符破坏（曾出现裸行 e 导致 set -e 在打分前退出）；拆成多行输出。
print_mymem_run_summary() {
  local dataset="$1"
  local version_suffix="$2"
  local st="${MEMBOOK_SHIFTS_SLIDING_STEP:-15}"
  echo "[run_membook.bash] dataset=${dataset} version=${version_suffix}"
  echo "[run_membook.bash] memory/research=${model_memory_research} QA/eval=${OPENAI_MODEL}"
  echo "[run_membook.bash] research iters: normal=${research_max_iters} deep=${research_max_deep_iters} page_index=${MEMBOOK_USE_PAGE_INDEX:-1}"
  echo "[run_membook.bash] max_users=${max_users:-all} user_offset=${user_offset:-0} uuid=${target_uuid:-all} frame=${frame}"
  echo "[run_membook.bash] shifts: min_turns=${MEMBOOK_SHIFTS_SLIDING_ENABLE_MIN_TURNS} window=${MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE} step=${st} overlap=$((MEMBOOK_SHIFTS_SLIDING_WINDOW_SIZE - st))"
}

validate_dataset() {
  local dataset="$1"
  if [[ "$dataset" != "medium" && "$dataset" != "long" ]]; then
    echo "dataset 必须为 medium 或 long，当前为: $dataset"
    exit 1
  fi
}

run_eval_membook_for_dataset() {
  local dataset="$1"
  echo "[run_membook.bash] ========== 数据集 ${dataset} · 阶段 1/2: eval_membook.py（记忆流水线）=========="
  local version_suffix
  local -a args=()
  if [[ "$dataset" == "medium" ]]; then
    args+=("--medium")
  else
    args+=("--long")
  fi

  if [[ "$append_timestamp" != true && "$clean" == true ]]; then
    args+=("--clean")
  fi

  if [[ "$append_timestamp" == true ]]; then
    version_suffix="${dataset}-${RUN_TS}"
    args+=("--version" "$version_suffix")
  elif [[ "$resume" == true ]]; then
    version_suffix="$resume_version"
    args+=("--version" "$version_suffix")
  else
    version_suffix="$dataset"
  fi

  args+=("--top_k" "$top_k")

  if [[ -n "$max_workers" ]]; then
    args+=("--max_workers" "$max_workers")
  fi

  if [[ -n "$research_max_iters" ]]; then
    args+=("--research_max_iters" "$research_max_iters")
  fi
  if [[ -n "$research_max_deep_iters" ]]; then
    args+=("--research_max_deep_iters" "$research_max_deep_iters")
  fi

  if [[ -n "$update_retrieve_top_k" ]]; then
    args+=("--update_retrieve_top_k" "$update_retrieve_top_k")
  fi

  if [[ -n "$model_memory_research" ]]; then
    args+=("--model" "$model_memory_research")
  fi

  args+=("--frame" "$frame")

  if [[ -n "$save_path" ]]; then
    args+=("--save_path" "$save_path")
  fi

  if [[ "$resume" == true ]]; then
    args+=("--resume")
  fi

  if [[ -n "$max_users" && "$max_users" -gt 0 ]]; then
    args+=("--max_users" "$max_users")
  fi

  if [[ -n "$user_offset" && "$user_offset" -gt 0 ]]; then
    args+=("--user_offset" "$user_offset")
  fi

  if [[ -n "$target_uuid" ]]; then
    args+=("--uuid" "$target_uuid")
  fi

  echo "[run_membook.bash] python=$("$PYTHON" --version 2>&1)"
  print_mymem_run_summary "$dataset" "${version_suffix:-$dataset}"
  "$PYTHON" eval_membook.py "${args[@]}"
  echo "[run_membook.bash] ========== 数据集 ${dataset} · 阶段 1/2 结束 =========="
}

run_evaluation_for_dataset() {
  local dataset="$1"
  echo "[run_membook.bash] ========== 数据集 ${dataset} · 阶段 2/2: evaluation.py（打分与汇总）=========="
  local eval_version
  if [[ "$append_timestamp" == true ]]; then
    eval_version="${dataset}-${RUN_TS}"
  elif [[ "$resume" == true ]]; then
    eval_version="$resume_version"
  else
    eval_version="$dataset"
  fi
  local -a eval_args=("--frame" "$frame" "--version" "$eval_version")
  if [[ -n "$max_users" && "$max_users" -gt 0 ]]; then
    eval_args+=("--user-num" "$max_users")
  fi
  if [[ -n "$eval_user_workers" && "$eval_user_workers" -gt 0 ]]; then
    eval_args+=("--user-workers" "$eval_user_workers")
  fi
  echo "[run_membook.bash] 打分: evaluation.py ${eval_args[*]}"
  "$PYTHON" evaluation.py "${eval_args[@]}"
  echo "[run_membook.bash] ========== 数据集 ${dataset} · 阶段 2/2 结束 =========="
}

for dataset in "${dataset_list[@]}"; do
  validate_dataset "$dataset"
  run_eval_membook_for_dataset "$dataset"
  run_evaluation_for_dataset "$dataset"
done
echo "[run_membook.bash] 全部完成（各数据集：记忆流水线 → 打分汇总）"

