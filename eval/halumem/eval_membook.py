"""
HaluMem runner for MemBook (full ToC in ``membook.memory.toc``).

See ``eval/halumem/README.md`` for ``MEMBOOK_*`` environment variables.
QA prompt: ``PROMPT_MEMBOOK``.
"""
import os
import sys
import time
import json
import copy
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import eval_baseline
from eval_baseline import (
    RESEARCH_MAX_ITERS,
    RESEARCH_MAX_DEEP_ITERS,
    _env_bm25_build_enabled,
    _resolve_data_path,
    extract_user_name,
)
from llms import llm_request
from prompts import PROMPT_MEMBOOK
from base_adapter import HaluMemBaseAdapter
from membook import MemoryAgent
from membook.compat import env
from membook.memory.toc import GeneralQaMode, normalize_general_qa_mode
from membook.memory.toc.schemas import TurnTimeRange
from membook.memory.toc.toc_builder import (
    _parse_time_value,
    normalized_session_pending_text,
    pending_retrieval_evidence_enabled,
)
from membook.usage.llm_usage import LLMUsageStats


def _attach_llm_usage_tracker(adapter: "MemBookAdapter", tracker: LLMUsageStats) -> None:
    memory_gen = adapter.memory_agent._llm._generator
    memory_gen.usage_tracker = tracker
    memory_gen.usage_category = "memory"
    research_gen = adapter.research_agent.generator
    research_gen.usage_tracker = tracker
    research_gen.usage_category = "research"


def _user_result_cached(tmp_dir: str, uuid: str) -> bool:
    """True when tmp/{uuid}.json exists and parses as JSON."""
    path = os.path.join(tmp_dir, f"{uuid}.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def _prepare_user_retry(save_path: str, uuid: str) -> None:
    """Drop partial artifacts before re-running a failed user."""
    import shutil

    tmp_dir = os.path.join(save_path, "tmp")
    error_path = os.path.join(tmp_dir, f"{uuid}_error.log")
    if os.path.isfile(error_path):
        os.remove(error_path)
    user_dir = os.path.join(save_path, "users", uuid)
    if os.path.isdir(user_dir):
        shutil.rmtree(user_dir)


def _merge_llm_usage_from_tmp_dir(tmp_dir: str, tracker: LLMUsageStats) -> int:
    """Merge llm_usage from all completed tmp/*.json files; return count merged."""
    if not os.path.isdir(tmp_dir):
        return 0
    merged = 0
    for name in os.listdir(tmp_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(tmp_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            usage = data.get("llm_usage")
            if usage:
                tracker.merge(LLMUsageStats.from_dict(usage))
                merged += 1
        except Exception:
            continue
    return merged


def _session_pending_line_from_pages(pages: List[Any]) -> str:
    for p in pages:
        meta = getattr(p, "meta", None) or {}
        if meta.get("session_pending") is None:
            continue
        pend = normalized_session_pending_text(meta.get("session_pending"))
        if pend:
            return pend
    return ""


def _append_pending_line_if_enabled(lines: List[str], pages: List[Any]) -> List[str]:
    """If ``MEMBOOK_PENDING_IN_RETRIEVAL`` is on and session has pending text, append one line."""
    if not pending_retrieval_evidence_enabled():
        return list(lines)
    pend = _session_pending_line_from_pages(pages)
    if not pend:
        return list(lines)
    extra = f"[PENDING] {pend}"
    out = list(lines)
    if extra not in out:
        out.append(extra)
    return out


def _membook_eval_extract_mode() -> str:
    raw = env("MEMBOOK_EVAL_EXTRACT_MODE")
    if raw is None or not str(raw).strip():
        return "confirmed_list"
    key = str(raw).strip().lower()
    if key in ("confirmed_list", "confirmed", "confirmed_points"):
        return "confirmed_list"
    if key in ("chapters", "chapter", "fragments", "fragment"):
        return "chapters"
    return "session_summary"


def _membook_extract_line_from_page(page: Any) -> Optional[str]:
    """ToC chapter → one evaluation line (same shape as ``add_dialogue`` return items)."""
    # Point pages must not be treated as chapters during extraction.
    pmeta = (getattr(page, "meta", None) or {}) if page is not None else {}
    if pmeta.get("memory_point_id"):
        return None
    title = str(page.meta.get("chapter_title") or page.header.replace("[CHAPTER]", "").strip())
    summary = str(page.meta.get("chapter_summary") or "").strip()
    text = f"{title}: {summary}".strip(": ").strip()
    return text if text else None


def _confirmed_lines_from_pages(pages: List[Any]) -> List[str]:
    lines: List[str] = []
    seen: set[str] = set()
    for p in pages:
        raw_items = (p.meta or {}).get("confirmed_memories") or []
        if not isinstance(raw_items, list):
            continue
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(text)
    return lines


class MemBookAdapter(HaluMemBaseAdapter):
    """MemBook HaluMem adapter: MemoryAgent write path + shared research/retrieve."""

    def __init__(self, *args, **kwargs):
        raw_mode = kwargs.pop("general_qa_mode", "nominal")
        os.environ["MEMBOOK_GENERAL_QA_FULL_PIPELINE"] = (
            "1" if normalize_general_qa_mode(raw_mode) == "full" else "0"
        )
        super().__init__(*args, **kwargs)
        self.general_qa_mode: GeneralQaMode = normalize_general_qa_mode(raw_mode)
        self.memory_agent = MemoryAgent(
            memory_store=self.memory_store,
            page_store=self.page_store,
            generator=self._memory_generator,
            research_agent=self.research_agent,
            general_qa_mode=self.general_qa_mode,
        )
        if (env("MEMBOOK_EVAL_VERBOSE", "") or "").strip().lower() in ("1", "true", "yes", "on"):
            print("[MemBook] MemoryAgent initialized (full MemBook ToC)")

    def add_dialogue(
        self,
        dialogue: List[Dict[str, Any]],
        timestamp: int,
        session_id: Optional[str] = None,
        *,
        session_start_time: str = "",
        session_end_time: str = "",
    ) -> Tuple[List[str], float]:
        start_time = time.time()
        if session_id is None:
            session_id = f"{self.user_id}_session_{self.session_counter}"
            self.session_counter += 1

        verbose = env("MEMBOOK_EVAL_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")
        if verbose:
            print(f"[MemBook][ToC] Building session {session_id} with {len(dialogue)} turns...")
        pages_before = len(self.page_store.load())
        if hasattr(self.memory_agent, "set_next_session_id"):
            self.memory_agent.set_next_session_id(session_id)
        time_span: Optional[TurnTimeRange] = None
        if session_start_time or session_end_time:
            time_span = TurnTimeRange(
                start=_parse_time_value(session_start_time),
                end=_parse_time_value(session_end_time),
            )
        if hasattr(self.memory_agent, "memorize_dialogue"):
            _ = self.memory_agent.memorize_dialogue(
                dialogue=dialogue,
                session_id=session_id,
                user_id=self.user_id,
                time_span=time_span,
            )
        else:
            dialogue_text = eval_baseline.format_dialogue_to_text(dialogue)
            _ = self.memory_agent.memorize(dialogue_text)

        all_pages = self.page_store.load()
        new_pages = all_pages[pages_before:]
        for p in new_pages:
            p.meta.update(
                {
                    "user_id": self.user_id,
                    "session_id": session_id,
                    "timestamp": timestamp,
                }
            )
        if new_pages:
            self.page_store.save(all_pages)

        extracted_memories: List[str] = []
        mode = _membook_eval_extract_mode()
        if mode == "confirmed_list":
            extracted_memories = _confirmed_lines_from_pages(new_pages)
        elif mode == "session_summary" and new_pages:
            s = str(new_pages[0].meta.get("session_summary") or "").strip()
            if s:
                extracted_memories = [s]
        else:
            for p in new_pages:
                line = _membook_extract_line_from_page(p)
                if line:
                    extracted_memories.append(line)

        extracted_memories = _append_pending_line_if_enabled(extracted_memories, new_pages)

        if verbose:
            pend = (new_pages[0].meta.get("session_pending") if new_pages else "") or ""
            if isinstance(pend, list):
                pend_s = ". ".join(str(x).strip() for x in pend if str(x).strip())
            else:
                pend_s = str(pend).strip()
            print(
                f"[MemBook][ToC] Session {session_id} done: chapters={len(new_pages)}, "
                f"pending_chars={len(pend_s)}"
            )

        for name, retriever in self.retrievers.items():
            try:
                retriever.update(self.page_store)
            except Exception as e:
                print(f"[WARN] 更新{name}检索器失败: {e}")

        return extracted_memories, (time.time() - start_time) * 1000

    def get_dialogue_memory(self, session_id: str) -> List[str]:
        """
        ``eval_baseline`` persists ``extracted_memories`` via this API. Baseline ``BaselineAdapter`` only
        reads ``[ABSTRACT]`` pages; MYMEM uses ToC pages, so we override.

        Lines written per session follow ``MEMBOOK_EVAL_EXTRACT_MODE`` (see module docstring).
        """
        all_pages = self.page_store.load()
        session_pages = [p for p in all_pages if p.meta.get("session_id") == session_id]
        if not session_pages:
            return []
        mode = _membook_eval_extract_mode()
        if mode == "confirmed_list":
            extracted_memories = _confirmed_lines_from_pages(session_pages)
        elif mode == "session_summary":
            s = str(session_pages[0].meta.get("session_summary") or "").strip()
            extracted_memories = [s] if s else []
        else:
            extracted_memories = []
            for p in session_pages:
                line = _membook_extract_line_from_page(p)
                if line:
                    extracted_memories.append(line)

        return _append_pending_line_if_enabled(extracted_memories, session_pages)


def _process_pool_mp_context(workers: int):
    """pyserini/pyjnius JVM is unsafe after fork; use spawn when BM25 runs in parallel."""
    if workers <= 1 or not _env_bm25_build_enabled():
        return None
    return mp.get_context("spawn")


def process_membook_user(
    user_data: Dict[str, Any],
    top_k: int,
    update_retrieve_top_k: int,
    save_path: str,
    memory_api_key: str,
    memory_base_url: str,
    memory_model: str,
    research_api_key: str,
    research_base_url: str,
    research_model: str,
    working_api_key: str,
    working_base_url: str,
    working_model: str,
    memory_api_type: str = "openai",
    research_api_type: str = "openai",
    working_api_type: str = "openai",
    research_max_iters: int = RESEARCH_MAX_ITERS,
    research_max_deep_iters: int = RESEARCH_MAX_DEEP_ITERS,
    resume: bool = False,
):
    """
    与 ``eval_baseline.process_user`` 等价（约 471–617 行），仅适配器为 ``MemBookAdapter``。
    若 HaluMem 形状有变，请同步修改 ``eval/halumem/eval_baseline.py`` 中 ``process_user``。
    """
    user_name = extract_user_name(user_data["persona_info"])
    sessions = user_data["sessions"]
    uuid = user_data["uuid"]

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{uuid}.json")

    user_dir = os.path.join(save_path, "users", uuid)
    os.makedirs(user_dir, exist_ok=True)

    new_user_data: Dict[str, Any] = {
        "uuid": uuid,
        "user_name": user_name,
        "sessions": [],
    }

    try:
        if resume:
            _prepare_user_retry(save_path, uuid)

        llm_usage = LLMUsageStats()
        adapter = MemBookAdapter(
            user_id=user_name,
            user_dir=user_dir,
            memory_api_key=memory_api_key,
            memory_base_url=memory_base_url,
            memory_model=memory_model,
            research_api_key=research_api_key,
            research_base_url=research_base_url,
            research_model=research_model,
            working_api_key=working_api_key,
            working_base_url=working_base_url,
            working_model=working_model,
            memory_api_type=memory_api_type,
            research_api_type=research_api_type,
            working_api_type=working_api_type,
            research_max_iters=research_max_iters,
            research_max_deep_iters=research_max_deep_iters,
        )
        _attach_llm_usage_tracker(adapter, llm_usage)

        for sid, session in enumerate(
            tqdm(sessions, total=len(sessions), desc=f"Processing user {user_name}")
        ):
            new_session = {
                "memory_points": session["memory_points"],
                "dialogue": session["dialogue"],
            }

            dialogue = session["dialogue"]
            formatted_dialogue = [
                {
                    "role": turn["role"],
                    "content": turn["content"],
                    "timestamp": turn.get("timestamp", ""),
                }
                for turn in dialogue
            ]

            date_format = "%b %d, %Y, %H:%M:%S"
            try:
                dt = datetime.strptime(session["start_time"], date_format).replace(tzinfo=timezone.utc)
                timestamp = int(dt.timestamp())
            except Exception:
                timestamp = int(time.time())

            session_id = f"{uuid}_session_{sid}"

            extracted_from_add, duration_ms = adapter.add_dialogue(
                dialogue=formatted_dialogue,
                timestamp=timestamp,
                session_id=session_id,
                session_start_time=str(session.get("start_time") or ""),
                session_end_time=str(session.get("end_time") or ""),
            )

            if session.get("is_generated_qa_session", False):
                new_session["add_dialogue_duration_ms"] = duration_ms
                new_session["is_generated_qa_session"] = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            new_session["add_dialogue_duration_ms"] = duration_ms

            # 与 eval_gam 一致先调 get_dialogue_memory；若因 meta/查询差异为空而 add_dialogue 已有行，则回退避免 jsonl 里全空导致 integrity/accuracy 0it
            persisted = adapter.get_dialogue_memory(session_id)
            new_session["extracted_memories"] = persisted if persisted else (extracted_from_add or [])

            for memory in new_session["memory_points"]:
                if memory["is_update"] == "False" or not memory.get("original_memories"):
                    continue
                _, memories_from_system, _ = adapter.retrieve_memory(
                    query=memory["memory_content"],
                    top_k=update_retrieve_top_k,
                    use_research_agent=False,
                )
                memory["memories_from_system"] = memories_from_system

            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []

            for qa in session["questions"]:
                context, _, search_duration_ms = adapter.retrieve_memory(
                    query=qa["question"],
                    top_k=top_k,
                    use_research_agent=True,
                )

                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = search_duration_ms

                prompt = PROMPT_MEMBOOK.format(
                    context=context,
                    question=qa["question"],
                )

                qa_start = time.time()
                response = llm_request(prompt, tracker=llm_usage, category="qa")
                new_qa["system_response"] = response
                new_qa["response_duration_ms"] = (time.time() - qa_start) * 1000

                new_session["questions"].append(new_qa)

            new_user_data["sessions"].append(new_session)

        new_user_data["llm_usage"] = llm_usage.to_dict()

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved user {user_name} to {tmp_file}")
        return {"uuid": uuid, "status": "ok", "path": tmp_file, "llm_usage": new_user_data["llm_usage"]}

    except Exception as e:
        error_path = os.path.join(tmp_dir, f"{uuid}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"❌ Error in user {user_name}: {e}")
        traceback.print_exc()
        return {"uuid": uuid, "status": "error", "path": error_path, "llm_usage": None}


def _membook_process_user_worker(
    user_data: Dict[str, Any],
    top_k: int,
    update_retrieve_top_k: int,
    save_path: str,
    memory_api_key: str,
    memory_base_url: str,
    memory_model: str,
    research_api_key: str,
    research_base_url: str,
    research_model: str,
    working_api_key: str,
    working_base_url: str,
    working_model: str,
    memory_api_type: str,
    research_api_type: str,
    working_api_type: str,
    research_max_iters: int,
    research_max_deep_iters: int,
    resume: bool,
):
    """Child-process entry: runs ``process_membook_user`` (no ``eval_baseline`` global patch)."""
    return process_membook_user(
        user_data,
        top_k,
        update_retrieve_top_k,
        save_path,
        memory_api_key,
        memory_base_url,
        memory_model,
        research_api_key,
        research_base_url,
        research_model,
        working_api_key,
        working_base_url,
        working_model,
        memory_api_type,
        research_api_type,
        working_api_type,
        research_max_iters,
        research_max_deep_iters,
        resume=resume,
    )


def run_membook_main(
    data_path: str,
    version: str = "default",
    frame: str = "membook",
    top_k: int = 20,
    update_retrieve_top_k: int = 10,
    max_workers: int = 2,
    max_users: Optional[int] = None,
    user_offset: int = 0,
    target_uuid: Optional[str] = None,
    research_max_iters: int = RESEARCH_MAX_ITERS,
    research_max_deep_iters: int = RESEARCH_MAX_DEEP_ITERS,
    save_path: Optional[str] = None,
    memory_api_key: Optional[str] = None,
    memory_base_url: Optional[str] = None,
    memory_model: str = "gpt-4o-mini",
    research_api_key: Optional[str] = None,
    research_base_url: Optional[str] = None,
    research_model: str = "gpt-4o-mini",
    working_api_key: Optional[str] = None,
    working_base_url: Optional[str] = None,
    working_model: str = "gpt-4o-mini",
    memory_api_type: str = "openai",
    research_api_type: str = "openai",
    working_api_type: str = "openai",
    resume: bool = False,
):
    if save_path is None:
        save_path = f"results/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    _base = os.getenv("OPENAI_BASE_URL")
    _model = os.getenv("OPENAI_MODEL")
    if _model:
        memory_model = research_model = working_model = _model
    if memory_api_key is None:
        memory_api_key = os.getenv("OPENAI_API_KEY", "")
    if memory_base_url is None:
        memory_base_url = _base
    if research_api_key is None:
        research_api_key = os.getenv("OPENAI_API_KEY", "")
    if research_base_url is None:
        research_base_url = _base
    if working_api_key is None:
        working_api_key = os.getenv("OPENAI_API_KEY", "")
    if working_base_url is None:
        working_base_url = _base

    workers = max(1, int(max_workers))
    run_mode = f"parallel(max_workers={workers})" if workers > 1 else "in_process"
    run_llm_usage = LLMUsageStats()
    print(
        f"[MemBook] OpenAI base_url: {memory_base_url}, model: {memory_model}, "
        f"research_max_iters: {research_max_iters}, "
        f"research_max_deep_iters: {research_max_deep_iters}, run_mode={run_mode}"
    )
    start_time = time.time()

    selected_users: List[Dict[str, Any]] = []
    for idx, user_data in enumerate(eval_baseline.iter_jsonl(data_path), 1):
        if idx <= user_offset:
            continue
        if max_users is not None and idx > user_offset + max_users:
            break
        uuid = user_data["uuid"]
        if target_uuid is not None and uuid != target_uuid:
            continue
        selected_users.append(user_data)

    if resume:
        skipped = 0
        pending_users: List[Dict[str, Any]] = []
        for user_data in selected_users:
            uuid = user_data["uuid"]
            if _user_result_cached(tmp_dir, uuid):
                skipped += 1
                continue
            pending_users.append(user_data)
        print(
            f"[MemBook] resume=true: skip {skipped} completed user(s), "
            f"pending {len(pending_users)} user(s)"
        )
        selected_users = pending_users
        _merge_llm_usage_from_tmp_dir(tmp_dir, run_llm_usage)

    total_users = len(selected_users)
    process_args = (
        top_k,
        update_retrieve_top_k,
        save_path,
        memory_api_key,
        memory_base_url,
        memory_model,
        research_api_key,
        research_base_url,
        research_model,
        working_api_key,
        working_base_url,
        working_model,
        memory_api_type,
        research_api_type,
        working_api_type,
        research_max_iters,
        research_max_deep_iters,
        resume,
    )

    if total_users == 0:
        print("[MemBook] No pending users — rebuilding jsonl / usage from cached tmp only.")
    elif workers > 1:
        mp_context = _process_pool_mp_context(workers)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
            futures = {
                executor.submit(_membook_process_user_worker, ud, *process_args): ud["uuid"]
                for ud in selected_users
            }
            for i, future in enumerate(as_completed(futures), 1):
                uuid = futures[future]
                try:
                    result = future.result()
                    if result.get("llm_usage"):
                        run_llm_usage.merge(LLMUsageStats.from_dict(result["llm_usage"]))
                    print(f"[{i}/{total_users}] ✅ Finished {uuid} ({result['status']})")
                except Exception as e:
                    print(f"[{i}/{total_users}] ❌ Error processing {uuid}: {e}")
                    traceback.print_exc()
    else:
        for i, user_data in enumerate(selected_users, 1):
            uuid = user_data["uuid"]
            try:
                result = process_membook_user(user_data, *process_args)
                if result.get("llm_usage"):
                    run_llm_usage.merge(LLMUsageStats.from_dict(result["llm_usage"]))
                print(f"[{i}/{total_users}] ✅ Finished {uuid} ({result['status']})")
            except Exception as e:
                print(f"[{i}/{total_users}] ❌ Error processing {uuid}: {e}")
                traceback.print_exc()

    with open(output_file, "w", encoding="utf-8") as f_out:
        for file in os.listdir(tmp_dir):
            if file.endswith(".json") and not file.endswith("_error.log"):
                file_path = os.path.join(tmp_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f_in:
                        data = json.load(f_in)
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"⚠️ Skipped {file}: {e}")

    elapsed = time.time() - start_time
    run_llm_usage = LLMUsageStats()
    _merge_llm_usage_from_tmp_dir(tmp_dir, run_llm_usage)
    usage_path = os.path.join(save_path, f"{frame}_llm_usage.json")
    usage_dict = run_llm_usage.to_dict()
    with open(usage_path, "w", encoding="utf-8") as f_usage:
        json.dump(usage_dict, f_usage, ensure_ascii=False, indent=2)
    print(
        f"[MemBook] LLM usage: api_calls={usage_dict['api_calls']}, "
        f"prompt={usage_dict['prompt_tokens']}, "
        f"completion={usage_dict['completion_tokens']}, "
        f"total={usage_dict['total_tokens']} "
        f"(source={usage_dict['token_source']}, "
        f"usage_missing_calls={usage_dict['usage_missing_calls']})"
    )
    print(f"✅ All done in {elapsed:.2f}s")
    print(f"✅ Final results saved to: {output_file}")
    print(f"✅ LLM usage saved to: {usage_path}")


# Public aliases
process_halumem_user = process_membook_user  # preferred name for MemBook HaluMem path
MYMEMAdapter = MemBookAdapter  # deprecated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HaluMem MemBook(ToC) evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--medium", action="store_true", help="处理 Medium 数据集")
    group.add_argument("--long", action="store_true", help="处理 Long 数据集")

    parser.add_argument("--clean", action="store_true", help="运行前删除旧结果目录")
    parser.add_argument("--top_k", type=int, default=5, help="Retrieval top_k for QA")
    parser.add_argument(
        "--update_retrieve_top_k",
        type=int,
        default=10,
        help="记忆更新评测时 Retrieve Memory API 的 top_k",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="并行处理用户数（>1 时用多进程；=1 时单进程顺序跑，便于看日志）",
    )
    parser.add_argument("--max_users", type=int, default=None, help="只处理前 N 个 user")
    parser.add_argument("--user_offset", type=int, default=0, help="跳过前 N 个 user")
    parser.add_argument("--uuid", type=str, default=None, help="只处理指定 uuid 的 user")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="统一指定 memory/research/working 模型，不传则保持默认",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default="membook",
        help='结果目录前缀，默认 "membook"，输出到 results/{frame}-{version}/',
    )
    parser.add_argument("--save_path", type=str, default=None, help="结果保存目录")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：跳过 tmp/{uuid}.json 已完成的 user，重跑失败/未完成的",
    )
    parser.add_argument("--version", type=str, default=None, help="版本后缀")
    parser.add_argument(
        "--research_max_iters",
        type=int,
        default=RESEARCH_MAX_ITERS,
        help="ResearchAgent 常规阶段最大反思/迭代轮数",
    )
    parser.add_argument(
        "--research_max_deep_iters",
        type=int,
        default=RESEARCH_MAX_DEEP_ITERS,
        help="ResearchAgent deep 阶段最大迭代轮数",
    )
    args = parser.parse_args()

    dataset = "medium" if args.medium else "long"
    data_path = _resolve_data_path(dataset)
    version = args.version if args.version is not None else dataset

    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        sys.exit(1)

    if args.save_path is not None:
        save_path = args.save_path
    else:
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        save_path = os.path.join(_project_root, "results", f"{args.frame}-{version}")

    if args.clean and os.path.exists(save_path):
        import shutil

        shutil.rmtree(save_path)
        print(f"🗑️ 已删除旧结果: {save_path}")

    main_kwargs = dict(
        data_path=data_path,
        version=version,
        frame=args.frame,
        top_k=args.top_k,
        update_retrieve_top_k=args.update_retrieve_top_k,
        max_workers=args.max_workers,
        max_users=args.max_users,
        user_offset=args.user_offset,
        target_uuid=args.uuid,
        research_max_iters=args.research_max_iters,
        research_max_deep_iters=args.research_max_deep_iters,
        save_path=save_path,
        resume=args.resume,
    )

    if args.model:
        main_kwargs["memory_model"] = args.model
        main_kwargs["research_model"] = args.model
        main_kwargs["working_model"] = args.model

    print(
        f"📂 数据集: {dataset} | 数据路径: {data_path} | 输出: {save_path} | "
        f"uuid: {args.uuid}"
    )
    run_membook_main(**main_kwargs)
