#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LongMemEval-S (cleaned) + HaluMem MYMEMAdapter（与「Indexing / Retrieval / Reading」三阶段一致）：

1. **Indexing**：逐 session ``add_dialogue`` → ``MemoryAgent``（MemBook ToC）→ page-store / 索引。
2. **Retrieval**：``retrieve_memory(..., use_research_agent=True)`` → ResearchAgent → integrated context。
3. **Reading**：``replace`` + ``json``；默认 **hybrid**（``temporal-reasoning`` 用 CoT，其余用 plain）。
   可用 ``--reading-mode cot|plain|hybrid`` 或 ``LONGMEMEVAL_READING_MODE``。
   模板来自 LongMemEval ``longmemeval_reader_prompt_templates``。
   查看拼装示例：``python3 eval/baseline/longmemeval_membook.py --show-reading-prompt``
4. 输出 jsonl：``question_id``、``hypothesis``、``reading_cot``、``reading_mode``，以及
   ``mymem_integrated_memory``（ResearchAgent 的 ``integrated_memory`` 原文）。

数据默认路径（可将参考仓库 ``data/longmemeval`` 拷到项目根下，或用 ``--data`` 指定）：
  ``data/longmemeval/longmemeval_s_cleaned.json``
  ``data/longmemeval/LongMemEval-main/src/generation/``（答题模板 import）

在项目根执行：
  python3 eval/baseline/longmemeval_membook.py --data data/longmemeval/longmemeval_s_cleaned.json --out results/.../hypotheses.jsonl

环境变量：``OPENAI_API_KEY``；可选 ``OPENAI_BASE_URL`` / ``MEMBOOK_LLM_BASE_URL``。
模型：``--model`` / ``MEMBOOK_MODEL`` / ``GAM_MODEL`` / ``OPENAI_MODEL``（默认 gpt-4o-mini）。
判分：``METRIC_MODEL``（默认 gpt-4o）与记忆侧独立。

若本机无默认数据目录，可从其它已配置仓库复制 ``data/longmemeval/``，或仅传 ``--data <你的 longmemeval_s_cleaned.json>``，
并保证存在 ``.../LongMemEval-main/src/generation/longmemeval_reader_prompt_templates.py``（与 ``--data`` 同树根下的相对路径一致）。

其它 CLI 行为与 ``longmemeval_gam`` 对齐：``--resume``、``--workers``、tqdm、``LONGMEMEVAL_*`` 等。
general_qa mode 按样本题型自动切换：
- ``single-session-assistant``: ``general_qa_mode=full``（general_qa chapter + MP 落盘并可检索）
- 其它题型: ``general_qa_mode=nominal``（仅名义合并，不落盘 general_qa）
Pyserini/Lucene 为共享 JVM，多线程不稳定时请 ``--workers 1``。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Literal


def _native_thread_env_defaults() -> None:
    for key, val in (
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("TOKENIZERS_PARALLELISM", "false"),
    ):
        os.environ.setdefault(key, val)


_native_thread_env_defaults()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_HALUMEM = os.path.join(_ROOT, "eval", "halumem")
if _HALUMEM not in sys.path:
    sys.path.insert(0, _HALUMEM)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_LM_GEN = os.path.join(_ROOT, "data", "longmemeval", "LongMemEval-main", "src", "generation")
if _LM_GEN not in sys.path:
    sys.path.insert(0, _LM_GEN)

from tqdm import tqdm  # noqa: E402

from longmemeval_reader_prompt_templates import (  # type: ignore[import-untyped]  # noqa: E402
    ANSWER_TEMPLATE_NONE_COT,
    ANSWER_TEMPLATE_NONE_PLAIN,
    ANSWER_TEMPLATE_REPLACE_COT,
    ANSWER_TEMPLATE_REPLACE_PLAIN,
)

READING_FIXED_MERGE_MODE: str = "replace"
READING_FIXED_HISTORY_FORMAT: str = "json"
# 兼容旧逻辑；hybrid 模式下按 question_type 逐条决定
READING_FIXED_COT: bool = True
READING_DEFAULT_MODE: str = "hybrid"
READING_COT_QUESTION_TYPES: Set[str] = frozenset({"temporal-reasoning"})

ReadingMode = Literal["hybrid", "cot", "plain"]
READING_PROMPT_TEMPLATE_COT_REPLACE: str = ANSWER_TEMPLATE_REPLACE_COT


def _normalize_reading_mode(mode: str) -> ReadingMode:
    m = (mode or READING_DEFAULT_MODE).strip().lower()
    if m in ("hybrid", "cot", "plain"):
        return m  # type: ignore[return-value]
    raise ValueError(f"reading_mode must be hybrid|cot|plain, got {mode!r}")


def resolve_reading_cot(entry: Dict[str, Any], reading_mode: ReadingMode) -> bool:
    """hybrid：仅 temporal-reasoning 用 CoT；cot/plain 为全量统一模板。"""
    if reading_mode == "cot":
        return True
    if reading_mode == "plain":
        return False
    qtype = str(entry.get("question_type") or "")
    return qtype in READING_COT_QUESTION_TYPES


def format_reading_config(reading_mode: ReadingMode) -> str:
    if reading_mode == "hybrid":
        types = ",".join(sorted(READING_COT_QUESTION_TYPES))
        return f"hybrid({READING_FIXED_MERGE_MODE}+CoT|plain+{READING_FIXED_HISTORY_FORMAT}; CoT types={types})"
    cot = reading_mode == "cot"
    return f"{reading_mode}({READING_FIXED_MERGE_MODE}+{('CoT' if cot else 'plain')}+{READING_FIXED_HISTORY_FORMAT})"


def _build_integrated_history_string(
    integrated_context: str,
    *,
    history_format: str,
    reading_merge_mode: str,
) -> str:
    """将 Researcher 整合上下文包成与 run_generation.prepare_prompt 相同的外层 flat-session 结构。"""
    text = (integrated_context or "").strip()

    if reading_merge_mode == "replace":
        if history_format == "json":
            sess_string = "\n" + json.dumps(text, ensure_ascii=False)
        elif history_format == "nl":
            sess_string = "\n\n" + text
        else:
            raise ValueError(f"history_format must be 'json' or 'nl', got {history_format!r}")
    elif reading_merge_mode == "none":
        if history_format == "json":
            chunk_entry: List[Dict[str, Any]] = [
                {"role": "mymem_researcher", "content": text},
            ]
            sess_string = "\n" + json.dumps(chunk_entry, ensure_ascii=False)
        elif history_format == "nl":
            chunk_nl = [{"role": "mymem_researcher", "content": text}]
            sess_string = ""
            for turn_entry in chunk_nl:
                sess_string += "\n\n{}: {}".format(
                    turn_entry["role"],
                    str(turn_entry.get("content", "")).strip(),
                )
        else:
            raise ValueError(f"history_format must be 'json' or 'nl', got {history_format!r}")
    else:
        raise ValueError(f"reading_merge_mode must be 'none' or 'replace', got {reading_merge_mode!r}")

    chunk_date = "(MYMEM Researcher integrated context)"
    return "\n### Session 1:\nSession Date: {}\nSession Content:\n{}\n".format(chunk_date, sess_string)


def build_longmemeval_official_reading_prompt(
    entry: Dict[str, Any],
    integrated_context: str,
    *,
    reading_cot: bool,
    reading_history_format: str,
    reading_merge_mode: str,
) -> str:
    """LongMemEval 官方 Reading 模板 + 注入 MYMEM 检索整合上下文。"""
    history_string = _build_integrated_history_string(
        integrated_context,
        history_format=reading_history_format,
        reading_merge_mode=reading_merge_mode,
    )
    question_date_string = str(entry.get("question_date") or "")
    question_string = str(entry.get("question") or "")
    if reading_merge_mode == "replace":
        tpl = ANSWER_TEMPLATE_REPLACE_COT if reading_cot else ANSWER_TEMPLATE_REPLACE_PLAIN
    else:
        tpl = ANSWER_TEMPLATE_NONE_COT if reading_cot else ANSWER_TEMPLATE_NONE_PLAIN
    return tpl.format(history_string, question_date_string, question_string)


def format_fixed_reading_prompt_demo(
    integrated_context: str,
    question_date: str,
    question: str,
    *,
    reading_cot: bool,
) -> str:
    entry = {"question_date": question_date, "question": question}
    return build_longmemeval_official_reading_prompt(
        entry,
        integrated_context,
        reading_cot=reading_cot,
        reading_history_format=READING_FIXED_HISTORY_FORMAT,
        reading_merge_mode=READING_FIXED_MERGE_MODE,
    )


def _load_halumem_eval_mymem():
    path = os.path.join(_HALUMEM, "eval_membook.py")
    spec = importlib.util.spec_from_file_location("halumem_eval_mymem", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mymem_adapter_cls: Optional[type] = None


def _get_mymem_adapter_cls():
    """惰性加载，便于 ``--show-reading-prompt`` 在无 API 依赖时退出。"""
    global _mymem_adapter_cls
    if _mymem_adapter_cls is None:
        _mymem_adapter_cls = _load_halumem_eval_mymem().MYMEMAdapter
    return _mymem_adapter_cls


_RE_HAYSTACK_DATE = re.compile(
    r"^(\d{4}/\d{2}/\d{2})\s*(?:\([^)]+\))?\s*(\d{1,2}:\d{2})\s*$"
)


def parse_haystack_timestamp(date_str: str, fallback: int) -> int:
    s = (date_str or "").strip()
    if not s:
        return fallback
    m = _RE_HAYSTACK_DATE.match(s)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M")
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    try:
        dt = datetime.strptime(s, "%Y/%m/%d (%a) %H:%M")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return fallback


def _resolve_openai_base_url() -> Optional[str]:
    raw = (os.getenv("OPENAI_BASE_URL") or os.getenv("MEMBOOK_LLM_BASE_URL") or "").strip()
    return raw if raw else None


def _preview_for_terminal(value: Any, max_chars: int) -> str:
    s = str(value) if value is not None else ""
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars].rstrip() + "…(truncated)"


def _env_flag_true(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _log_line(msg: str, *, prefer_tqdm_write: bool) -> None:
    if prefer_tqdm_write:
        tqdm.write(msg)
    else:
        print(msg)


def load_done_question_ids(out_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.isfile(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("question_id")
                if qid and "hypothesis" in obj:
                    done.add(str(qid))
            except json.JSONDecodeError:
                continue
    return done


def process_one_entry(
    entry: Dict[str, Any],
    *,
    save_path: str,
    top_k: int,
    memory_api_key: str,
    memory_base_url: Optional[str],
    memory_model: str,
    research_api_key: str,
    research_base_url: Optional[str],
    research_model: str,
    working_api_key: str,
    working_base_url: Optional[str],
    working_model: str,
    research_max_iters: int,
    research_max_deep_iters: int,
    reading_mode: ReadingMode,
    show_progress: bool,
    sample_pbar: Any = None,
    on_session_indexed: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    qid = str(entry["question_id"])
    question = entry["question"]
    sessions: List[List[Dict[str, Any]]] = entry["haystack_sessions"]
    dates: List[str] = entry.get("haystack_dates") or []

    user_dir = os.path.join(save_path, "users", qid)
    os.makedirs(user_dir, exist_ok=True)

    mb = memory_base_url or ""
    rb = research_base_url or ""
    wb = working_base_url or ""

    adapter = _get_mymem_adapter_cls()(
        user_id=qid,
        user_dir=user_dir,
        memory_api_key=memory_api_key,
        memory_base_url=mb,
        memory_model=memory_model,
        research_api_key=research_api_key,
        research_base_url=rb,
        research_model=research_model,
        working_api_key=working_api_key,
        working_base_url=wb,
        working_model=working_model,
        research_max_iters=research_max_iters,
        research_max_deep_iters=research_max_deep_iters,
        general_qa_mode=(
            "full" if str(entry.get("question_type") or "").strip() == "single-session-assistant" else "nominal"
        ),
    )

    if sample_pbar is not None:
        sample_pbar.set_postfix_str(f"{qid} · add_dialogue", refresh=True)

    base_ts = int(time.time())
    n_sess = len(sessions)
    session_iter: Any = sessions
    if on_session_indexed is None and show_progress and n_sess > 0:
        session_iter = tqdm(
            sessions,
            total=n_sess,
            desc=f"Indexing sessions ({qid})",
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
        )
    for i, session in enumerate(session_iter):
        date_str = dates[i] if i < len(dates) else ""
        ts = parse_haystack_timestamp(date_str, base_ts + i)
        formatted = [
            {
                "role": turn["role"],
                "content": turn.get("content", ""),
                "timestamp": date_str or "",
            }
            for turn in session
            if isinstance(turn, dict) and turn.get("role")
        ]
        session_id = f"{qid}_sess_{i}"
        adapter.add_dialogue(
            dialogue=formatted,
            timestamp=ts,
            session_id=session_id,
            session_start_time=date_str,
            session_end_time=date_str,
        )
        if on_session_indexed is not None:
            on_session_indexed()

    if sample_pbar is not None:
        sample_pbar.set_postfix_str(f"{qid} · retrieve+answer", refresh=True)

    context, memories, _ = adapter.retrieve_memory(
        query=question,
        top_k=top_k,
        use_research_agent=True,
    )
    integrated_memory = memories[0] if memories else ""
    use_cot = resolve_reading_cot(entry, reading_mode)
    prompt = build_longmemeval_official_reading_prompt(
        entry,
        context,
        reading_cot=use_cot,
        reading_history_format=READING_FIXED_HISTORY_FORMAT,
        reading_merge_mode=READING_FIXED_MERGE_MODE,
    )
    from llms import llm_request  # type: ignore[import-untyped]

    # HaluMem llms.llm_request 仅接受 prompt；模型读模块级 MODEL（环境变量 OPENAI_MODEL）。
    # 生成阶段请由 shell 在启动前设置 OPENAI_MODEL（eval_longmemeval_membook.sh 已对齐 MEMBOOK_MODEL）。
    hypothesis = llm_request(prompt)
    if not isinstance(hypothesis, str):
        hypothesis = str(hypothesis)

    return {
        "question_id": qid,
        "hypothesis": hypothesis.strip(),
        "mymem_integrated_memory": integrated_memory,
        "reading_cot": use_cot,
        "reading_mode": reading_mode,
    }


def _process_one_entry_worker(
    task: Tuple[int, Dict[str, Any], Dict[str, Any]],
) -> Tuple[int, Dict[str, Any], str, Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    idx, entry, p_kw = task
    qid = str(entry.get("question_id", ""))
    try:
        row = process_one_entry(entry, **p_kw)
        return (idx, entry, qid, row, None)
    except Exception as e:
        return (
            idx,
            entry,
            qid,
            None,
            {"error": str(e), "traceback": traceback.format_exc()},
        )


def main() -> None:
    default_data = os.path.join(_ROOT, "data", "longmemeval", "longmemeval_s_cleaned.json")

    parser = argparse.ArgumentParser(description="LongMemEval-S + MYMEM(ToC) → hypotheses jsonl")
    parser.add_argument("--data", type=str, default=default_data, help="longmemeval_s_cleaned.json 路径")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(_ROOT, "results", "longmemeval_membook", "hypotheses.jsonl"),
        help="输出 jsonl（每行 question_id + hypothesis）",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="MYMEM 状态根目录；默认与 --out 同目录下的 mymem_state/",
    )
    parser.add_argument("--top_k", type=int, default=20, help="retrieve_memory 的 top_k")
    parser.add_argument("--start", type=int, default=0, help="起始样本下标（0-based）")
    parser.add_argument("--max_items", type=int, default=None, help="最多处理条数；不设则直到文件末尾")
    parser.add_argument(
        "--research_max_iters",
        type=int,
        default=int(os.getenv("GAM_RESEARCH_MAX_ITERS", "3")),
    )
    parser.add_argument(
        "--research_max_deep_iters",
        type=int,
        default=int(os.getenv("GAM_RESEARCH_MAX_DEEP_ITERS", "3")),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="统一指定 memory/research/working 与 hypothesis 生成模型；默认 MEMBOOK_MODEL / GAM_MODEL / OPENAI_MODEL / gpt-4o-mini",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若 out 已存在，跳过其中已有 question_id",
    )
    _reading_mode_default = _normalize_reading_mode(
        os.getenv("LONGMEMEVAL_READING_MODE", READING_DEFAULT_MODE)
    )
    parser.add_argument(
        "--reading-mode",
        type=str,
        choices=["hybrid", "cot", "plain"],
        default=_reading_mode_default,
        help="Reading 模板：hybrid=temporal-reasoning 用 CoT 其余 plain（默认）；cot/plain=全量统一",
    )
    parser.add_argument(
        "--show-reading-prompt",
        action="store_true",
        help="打印 Reading 示例（按 --reading-mode）后退出，不调 API",
    )
    _pa_default = os.getenv("LONGMEMEVAL_PRINT_ANSWERS", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    parser.add_argument(
        "--print-answers",
        action=argparse.BooleanOptionalAction,
        default=_pa_default,
        help="每条成功后打印 reference answer 与 hypothesis；可用 LONGMEMEVAL_PRINT_ANSWERS=false 或 --no-print-answers",
    )
    parser.add_argument(
        "--print-answer-max-chars",
        type=int,
        default=int(os.getenv("LONGMEMEVAL_PRINT_ANSWER_MAX_CHARS", "2000")),
        help="终端展示时每段最多字符数；0=不截断。可用 LONGMEMEVAL_PRINT_ANSWER_MAX_CHARS",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 tqdm；可用 LONGMEMEVAL_PROGRESS=0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(os.getenv("LONGMEMEVAL_WORKERS", "1"))),
        help="并发样本数（默认 1）；>1 为线程池。可用 LONGMEMEVAL_WORKERS",
    )
    args = parser.parse_args()
    workers = max(1, int(args.workers))

    reading_mode = _normalize_reading_mode(args.reading_mode)

    if args.show_reading_prompt:
        demo_ctx = (
            "[MYMEM Integrated Memory 示例]\n"
            "- User prefers using Suica and TripIt when navigating Tokyo public transport.\n"
            "- …（真实运行时此处为 Researcher.integrated_memory 全文）"
        )
        print(f"=== LongMemEval Reading（{format_reading_config(reading_mode)}）===\n")
        if reading_mode == "hybrid":
            for label, cot, q in (
                ("temporal-reasoning → CoT", True, "How many days ago did I start training?"),
                ("other types → plain", False, "What degree did I graduate with?"),
            ):
                print(f"--- {label} ---\n")
                print(
                    format_fixed_reading_prompt_demo(
                        demo_ctx,
                        "2023/05/30 (Tue) 23:40",
                        q,
                        reading_cot=cot,
                    )
                )
                print()
        else:
            print(
                format_fixed_reading_prompt_demo(
                    demo_ctx,
                    "2023/05/30 (Tue) 23:40",
                    "I'm a bit anxious about getting around Tokyo. Do you have any helpful tips?",
                    reading_cot=(reading_mode == "cot"),
                )
            )
        print("=== 以上即 llm_request 前的 user 全文（占位仅为演示）===")
        return

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ 请设置环境变量 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    base_url = _resolve_openai_base_url()
    model = (
        args.model
        or os.getenv("MEMBOOK_MODEL")
        or os.getenv("GAM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )

    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "mymem_state")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    n = len(data)
    end = n if args.max_items is None else min(n, args.start + args.max_items)
    done = load_done_question_ids(args.out) if args.resume else set()

    err_path = os.path.splitext(args.out)[0] + ".errors.jsonl"
    mode = "a" if args.resume and os.path.isfile(args.out) else "w"

    use_progress = (not args.no_progress) and _env_flag_true("LONGMEMEVAL_PROGRESS", default=True)

    print(
        f"📂 data={args.data} | samples [{args.start}, {end}) | out={args.out} | "
        f"save_path={save_path} | model={model} | base_url={base_url!r} | resume={args.resume} | "
        f"reading={format_reading_config(reading_mode)} | "
        f"progress={'tqdm' if use_progress else 'off'} | workers={workers}"
    )

    tqdm_write = use_progress
    sample_indices = range(args.start, end)
    sample_pbar: Any = None
    outer_iter: Any = sample_indices
    if use_progress and workers <= 1:
        sample_pbar = tqdm(
            sample_indices,
            total=end - args.start,
            desc="LongMemEval-S samples (MYMEM)",
            dynamic_ncols=True,
            file=sys.stderr,
            bar_format=(
                "{desc}: {n_fmt}/{total_fmt} ({percentage:4.1f}%)|{bar}| "
                "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
            ),
        )
        outer_iter = sample_pbar

    _bar_fmt_parallel = (
        "{desc}: {n_fmt}/{total_fmt} ({percentage:4.1f}%)|{bar}| "
        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    )

    def _write_success(idx: int, entry: Dict[str, Any], qid: str, row: Dict[str, Any]) -> None:
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        _log_line(f"✅ [{idx + 1}/{end}] {qid}", prefer_tqdm_write=tqdm_write)
        if args.print_answers:
            gold = entry.get("answer", "")
            hyp = row.get("hypothesis", "")
            mc = args.print_answer_max_chars
            _log_line(f"\n--- question_id={qid} ---", prefer_tqdm_write=tqdm_write)
            _log_line(
                f"reference (gold):\n{_preview_for_terminal(gold, mc)}",
                prefer_tqdm_write=tqdm_write,
            )
            _log_line(
                f"hypothesis (system):\n{_preview_for_terminal(hyp, mc)}",
                prefer_tqdm_write=tqdm_write,
            )
            _log_line("---\n", prefer_tqdm_write=tqdm_write)

    def _write_err(idx: int, qid: str, err: str, tb: str) -> None:
        err_f.write(
            json.dumps(
                {"question_id": qid, "error": err, "traceback": tb},
                ensure_ascii=False,
            )
            + "\n"
        )
        err_f.flush()
        msg = f"❌ [{idx + 1}/{end}] {qid}: {err}"
        if tqdm_write:
            tqdm.write(msg, file=sys.stderr)
        else:
            print(msg, file=sys.stderr)

    p_kw_parallel_base: Dict[str, Any] = {
        "save_path": save_path,
        "top_k": args.top_k,
        "memory_api_key": api_key,
        "memory_base_url": base_url,
        "memory_model": model,
        "research_api_key": api_key,
        "research_base_url": base_url,
        "research_model": model,
        "working_api_key": api_key,
        "working_base_url": base_url,
        "working_model": model,
        "research_max_iters": args.research_max_iters,
        "research_max_deep_iters": args.research_max_deep_iters,
        "reading_mode": reading_mode,
        "show_progress": False,
        "sample_pbar": None,
    }

    with open(args.out, mode, encoding="utf-8") as out_f, open(
        err_path, "a", encoding="utf-8"
    ) as err_f:
        if workers > 1:
            tasks: List[Tuple[int, Dict[str, Any]]] = []
            total_sessions_in_batch = 0
            for idx in range(args.start, end):
                entry = data[idx]
                qid = str(entry.get("question_id", ""))
                if not qid:
                    continue
                if qid in done:
                    _log_line(f"⏭️  skip {qid} (already in output)", prefer_tqdm_write=tqdm_write)
                    continue
                total_sessions_in_batch += len(entry.get("haystack_sessions") or [])
                tasks.append((idx, entry))
            if tasks:
                session_pbar: Any = None
                sess_lock = threading.Lock()
                p_kw: Dict[str, Any] = dict(p_kw_parallel_base)
                if use_progress and total_sessions_in_batch > 0:
                    session_pbar = tqdm(
                        total=total_sessions_in_batch,
                        desc="Indexing sessions (all workers)",
                        position=1,
                        leave=True,
                        dynamic_ncols=True,
                        file=sys.stderr,
                        bar_format=_bar_fmt_parallel,
                    )

                    def _on_session_indexed() -> None:
                        with sess_lock:
                            session_pbar.update(1)

                    p_kw["on_session_indexed"] = _on_session_indexed
                try:
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        mapped = ex.map(
                            _process_one_entry_worker,
                            [(idx, entry, p_kw) for idx, entry in tasks],
                        )
                        if use_progress:
                            mapped = tqdm(
                                mapped,
                                total=len(tasks),
                                desc="LongMemEval-S samples (MYMEM, parallel)",
                                position=0,
                                leave=True,
                                dynamic_ncols=True,
                                file=sys.stderr,
                                bar_format=_bar_fmt_parallel,
                            )
                        for idx, entry, qid, row, err_payload in mapped:
                            if err_payload is not None:
                                _write_err(idx, qid, err_payload["error"], err_payload["traceback"])
                            else:
                                assert row is not None
                                _write_success(idx, entry, qid, row)
                finally:
                    if session_pbar is not None:
                        session_pbar.close()
        else:
            for idx in outer_iter:
                entry = data[idx]
                qid = str(entry.get("question_id", ""))
                if not qid:
                    continue
                if qid in done:
                    _log_line(f"⏭️  skip {qid} (already in output)", prefer_tqdm_write=tqdm_write)
                    continue
                try:
                    row = process_one_entry(
                        entry,
                        save_path=save_path,
                        top_k=args.top_k,
                        memory_api_key=api_key,
                        memory_base_url=base_url,
                        memory_model=model,
                        research_api_key=api_key,
                        research_base_url=base_url,
                        research_model=model,
                        working_api_key=api_key,
                        working_base_url=base_url,
                        working_model=model,
                        research_max_iters=args.research_max_iters,
                        research_max_deep_iters=args.research_max_deep_iters,
                        reading_mode=reading_mode,
                        show_progress=use_progress,
                        sample_pbar=sample_pbar,
                    )
                    _write_success(idx, entry, qid, row)
                except Exception as e:
                    _write_err(idx, qid, str(e), traceback.format_exc())

    print("✅ 完成。判分：./scripts/eval_longmemeval_membook.sh（默认 RUN_JUDGE=1）或：")
    print(
        "   cd data/longmemeval/LongMemEval-main/src/evaluation && "
        "python3 evaluate_qa.py gpt-4o-mini <hypotheses.jsonl> <longmemeval_s_cleaned.json>"
    )


if __name__ == "__main__":
    main()
