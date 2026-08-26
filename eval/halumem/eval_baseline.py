import os
import re
import sys
import time
import json
import copy
import traceback
from datetime import datetime, timezone
try:
    from dotenv import load_dotenv
except ImportError:  # 精简环境可省略 python-dotenv（run_membook.bash 会 source .env）
    def load_dotenv(*_a, **_k):
        return False
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from tqdm import tqdm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed
)

# 添加项目根到 sys.path
gam_path = os.path.join(os.path.dirname(__file__), "..", "..")
if gam_path not in sys.path:
    sys.path.insert(0, gam_path)

from membook import MemoryAgent

from llms import llm_request
# QA 阶段：用 HaluMem 评测统一的「context + question → answer」模板，与 MemZero/MemoBase 等一致，便于公平对比
from prompts import PROMPT_MEMZERO
from base_adapter import (
    HaluMemBaseAdapter,
    RESEARCH_MAX_ITERS,
    RESEARCH_MAX_DEEP_ITERS,
    RETRY_TIMES,
    WAIT_TIME,
    TEMPLATE_MEMORY,
    _env_bm25_build_enabled,
)

load_dotenv()


def format_dialogue_to_text(dialogue: List[Dict[str, Any]]) -> str:
    """
    将HaluMem的对话格式转换为文本字符串
    
    输入：
    - dialogue: [{"role": "user", "content": "...", "timestamp": "..."}, ...]
    
    输出：
    - 格式化的对话文本
    """
    lines = []
    for turn in dialogue:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        timestamp = turn.get("timestamp", "")
        if timestamp:
            lines.append(f"[{timestamp}]{role}: {content}")
        else:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_user_name(persona_info: str) -> str:
    """从persona_info中提取用户名"""
    match = re.search(r'Name:\s*(.*?); Gender:', persona_info)
    if match:
        username = match.group(1).strip()
        return username
    else:
        raise ValueError("No name found in persona_info.")


class BaselineAdapter(HaluMemBaseAdapter):
    """
    Legacy page-summary baseline 适配器（论文复现）：MemoryAgent + page abstract 抽取。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_agent = MemoryAgent(
            memory_store=self.memory_store,
            page_store=self.page_store,
            generator=self._memory_generator,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_fixed(WAIT_TIME),
        stop=stop_after_attempt(RETRY_TIMES),
        reraise=True
    )
    def add_dialogue(
        self,
        dialogue: List[Dict[str, Any]],
        timestamp: int,
        session_id: Optional[str] = None,
        *,
        session_start_time: str = "",
        session_end_time: str = "",
    ) -> Tuple[List[str], float]:
        """
        Add Dialogue API

        输入：
        - dialogue: 对话列表 [{"role": "user", "content": "...", "timestamp": "..."}, ...]
        - timestamp: 时间戳
        - session_id: 会话ID（可选，如果不提供会自动生成）

        输出：
        - extracted_memories: 提取的记忆列表
        - duration_ms: 处理时间（毫秒）
        """
        start_time = time.time()

        if session_id is None:
            session_id = f"{self.user_id}_session_{self.session_counter}"
            self.session_counter += 1

        dialogue_text = format_dialogue_to_text(dialogue)
        memory_update = self.memory_agent.memorize(dialogue_text)

        all_pages = self.page_store.load()
        if all_pages:
            last_page = all_pages[-1]
            last_page.meta.update({
                "user_id": self.user_id,
                "session_id": session_id,
                "timestamp": timestamp
            })
            self.page_store.save(all_pages)

        abstract = memory_update.new_state.abstracts[-1] if memory_update.new_state.abstracts else ""
        extracted_memories = [abstract] if abstract else []

        for name, retriever in self.retrievers.items():
            try:
                retriever.update(self.page_store)
            except Exception as e:
                print(f"[WARN] 更新{name}检索器失败: {e}")

        duration_ms = (time.time() - start_time) * 1000
        return extracted_memories, duration_ms

    def get_dialogue_memory(self, session_id: str) -> List[str]:
        """Get Dialogue Memory API — extract [ABSTRACT] pages for a session."""
        all_pages = self.page_store.load()
        session_pages = [
            p for p in all_pages
            if p.meta.get("session_id") == session_id
        ]

        extracted_memories = []
        for page in session_pages:
            if page.header.startswith("[ABSTRACT]"):
                abstract = page.header.replace("[ABSTRACT]", "").strip()
                if abstract:
                    extracted_memories.append(abstract)
            elif page.meta.get("decorated"):
                decorated = page.meta.get("decorated", "")
                if "[ABSTRACT]" in decorated:
                    parts = decorated.split(";", 1)
                    if len(parts) > 0:
                        abstract = parts[0].replace("[ABSTRACT]", "").strip()
                        if abstract:
                            extracted_memories.append(abstract)

        return extracted_memories



def process_user(
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
):
    """
    处理单个用户的所有数据
    
    输入：
    - user_data: 用户数据（包含uuid, persona_info, sessions）
    - top_k: 检索时返回的记忆数量
    - save_path: 结果保存路径
    - 其他参数：API配置
    
    输出：
    - 保存结果到 tmp/{uuid}.json
    """
    
    user_name = extract_user_name(user_data["persona_info"])
    sessions = user_data["sessions"]
    uuid = user_data["uuid"]
    
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{uuid}.json")
    
    # 为每个用户创建独立的存储目录
    user_dir = os.path.join(save_path, "users", uuid)
    os.makedirs(user_dir, exist_ok=True)
    
    new_user_data = {
        "uuid": uuid,
        "user_name": user_name,
        "sessions": []
    }
    
    try:
        # 创建 BaselineAdapter
        adapter = BaselineAdapter(
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
        
        # 处理每个session
        for sid, session in enumerate(tqdm(sessions, total=len(sessions), desc=f"Processing user {user_name}")):
            new_session = {
                "memory_points": session["memory_points"],
                "dialogue": session["dialogue"]
            }
            
            # 格式化对话
            dialogue = session["dialogue"]
            formatted_dialogue = [
                {
                    "role": turn["role"],
                    "content": turn["content"],
                    "timestamp": turn.get("timestamp", "")
                }
                for turn in dialogue
            ]
            
            # 解析时间戳
            date_format = "%b %d, %Y, %H:%M:%S"
            try:
                dt = datetime.strptime(session["start_time"], date_format).replace(tzinfo=timezone.utc)
                timestamp = int(dt.timestamp())
            except Exception:
                timestamp = int(time.time())
            
            # 生成session_id
            session_id = f"{uuid}_session_{sid}"
            
            # API 1: Add Dialogue API — 输入对话，由记忆系统执行记忆提取
            _, duration_ms = adapter.add_dialogue(
                dialogue=formatted_dialogue,
                timestamp=timestamp,
                session_id=session_id,
                session_start_time=str(session.get("start_time") or ""),
                session_end_time=str(session.get("end_time") or ""),
            )
            
            if session.get('is_generated_qa_session', False):
                new_session["add_dialogue_duration_ms"] = duration_ms
                new_session["is_generated_qa_session"] = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue
            
            new_session["add_dialogue_duration_ms"] = duration_ms

            # API 2: Get Dialogue Memory API — 从该会话中取出系统提取的所有记忆点（用于记忆提取评测）
            new_session["extracted_memories"] = adapter.get_dialogue_memory(session_id)

            # API 3（用于记忆更新评测）：与 eval_memzero / eval_zep 一致，用 Retrieve Memory API，query=memory_content，top_k=10
            for memory in new_session["memory_points"]:
                if memory["is_update"] == "False" or not memory.get("original_memories"):
                    continue
                _, memories_from_system, _ = adapter.retrieve_memory(
                    query=memory["memory_content"],
                    top_k=update_retrieve_top_k,
                    use_research_agent=False,
                )
                memory["memories_from_system"] = memories_from_system

            # API 3（用于记忆问答 QA 评测）：Retrieve Memory API，query=question
            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue
            
            new_session["questions"] = []
            
            for qa in session["questions"]:
                # 检索记忆
                context, _, search_duration_ms = adapter.retrieve_memory(
                    query=qa["question"],
                    top_k=top_k,
                    use_research_agent=True  # 问答使用ResearchAgent
                )
                
                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = search_duration_ms
                
                # 生成答案
                prompt = PROMPT_MEMZERO.format(
                    context=context,
                    question=qa["question"]
                )
                
                start_time = time.time()
                response = llm_request(prompt)
                new_qa["system_response"] = response
                new_qa["response_duration_ms"] = (time.time() - start_time) * 1000
                
                new_session["questions"].append(new_qa)
            
            new_user_data["sessions"].append(new_session)
        
        # 保存结果
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved user {user_name} to {tmp_file}")
        return {"uuid": uuid, "status": "ok", "path": tmp_file}
    
    except Exception as e:
        error_path = os.path.join(tmp_dir, f"{uuid}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"❌ Error in user {user_name}: {e}")
        traceback.print_exc()
        return {"uuid": uuid, "status": "error", "path": error_path}


def iter_jsonl(file_path: str):
    """迭代读取JSONL文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(
    data_path: str,
    version: str = "default",
    frame: str = "baseline",
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
):
    """
    主函数：批量处理用户数据
    
    输入：
    - data_path: HaluMem数据集路径
    - version: 版本标识
    - frame: 结果目录前缀（默认 "baseline"）
    - top_k: 检索返回数量
    - max_workers: 并行处理数
    - save_path: 结果保存目录；若为 None，则使用 "results/{frame}-{version}/"
    - 其他参数：API配置
    """
    if save_path is None:
        save_path = f"results/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    
    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 从环境变量读取API配置（如果未提供）。统一使用 OPENAI_BASE_URL / OPENAI_API_KEY
    _base = os.getenv("OPENAI_BASE_URL")
    # 模型：OPENAI_MODEL 覆盖 memory/research/working，便于阿里等网关只支持 qwen 系列时统一配置
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

    print(
        f"[Baseline] OpenAI base_url: {memory_base_url}, model: {memory_model}, "
        f"research_max_iters: {research_max_iters}, research_max_deep_iters: {research_max_deep_iters}"
    )
    start_time = time.time()
    
    # 并行处理用户（BM25/pyserini JVM 不能在 fork 子进程中复用，需 spawn）
    mp_context = mp.get_context("spawn") if max_workers > 1 and _env_bm25_build_enabled() else None
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        futures = {}
        for idx, user_data in enumerate(iter_jsonl(data_path), 1):
            if idx <= user_offset:
                continue
            if max_users is not None and idx > user_offset + max_users:
                break
            uuid = user_data["uuid"]
            if target_uuid is not None and uuid != target_uuid:
                continue
            future = executor.submit(
                process_user,
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
            )
            futures[future] = uuid
        
        total_users = len(futures)
        
        for i, future in enumerate(as_completed(futures), 1):
            uuid = futures[future]
            try:
                result = future.result()
                print(f"[{i}/{total_users}] ✅ Finished {uuid} ({result['status']})")
            except Exception as e:
                print(f"[{i}/{total_users}] ❌ Error processing {uuid}: {e}")
                traceback.print_exc()
    
    # 合并所有结果（覆盖写入，避免重复）
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
    print(f"✅ All done in {elapsed:.2f}s")
    print(f"✅ Final results saved to: {output_file}")


def _resolve_data_path(dataset: str) -> str:
    """根据数据集类型返回完整路径"""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "halumem")
    if dataset == "medium":
        return os.path.join(base_dir, "HaluMem-Medium.jsonl")
    elif dataset == "long":
        return os.path.join(base_dir, "HaluMem-Long.jsonl")
    else:
        raise ValueError(f"Unknown dataset: {dataset}, use 'medium' or 'long'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HaluMem page-summary baseline evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--medium",
        action="store_true",
        help="处理 Medium 数据集 (HaluMem-Medium.jsonl)",
    )
    group.add_argument(
        "--long",
        action="store_true",
        help="处理 Long 数据集 (HaluMem-Long.jsonl)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="运行前删除该数据集的旧结果目录 (results/{frame}-{medium|long}/)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Retrieval top_k for QA",
    )
    parser.add_argument(
        "--update_retrieve_top_k",
        type=int,
        default=10,
        help="记忆更新评测时 Retrieve Memory API 的 top_k（与 eval_memzero/zep 一致）",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="并行处理的进程数（同时处理的用户数）",
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="只处理前 N 个 user，不设则处理全部（快速迭代时可设为 1）",
    )
    parser.add_argument(
        "--user_offset",
        type=int,
        default=0,
        help="跳过前 N 个 user，再按 max_users 处理（例如 user_offset=10 + max_users=10 表示只跑第 11–20 个 user）",
    )
    parser.add_argument(
        "--uuid",
        type=str,
        default=None,
        help="只处理指定 uuid 的 user（与 case study 一致时用）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "统一指定 memory/research/working 模型，如 gpt-4o。"
            "不传则用环境变量 OPENAI_MODEL，再否则用默认 gpt-4o-mini"
        ),
    )
    parser.add_argument(
        "--frame",
        type=str,
        default="baseline",
        help='结果目录前缀，默认为 "gam"，最终目录为 results/{frame}-{version}/',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="结果保存目录；若不指定，则使用 results/{frame}-{version}/",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="版本后缀；若指定则结果目录为 results/{frame}-{version}/，便于时间戳追加不覆盖（如 medium-20250228_143022）",
    )
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
        print(f"   请确认 data/halumem/ 下存在 HaluMem-{dataset.capitalize()}.jsonl")
        sys.exit(1)

    if args.save_path is not None:
        save_path = args.save_path
    else:
        # 默认写到项目根目录的 results/ 下，与 eval/gam 等一致
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        save_path = os.path.join(_project_root, "results", f"{args.frame}-{version}")

    if args.clean and os.path.exists(save_path):
        import shutil

        shutil.rmtree(save_path)
        print(f"🗑️  已删除旧结果: {save_path}")

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
    )

    if args.model:
        main_kwargs["memory_model"] = main_kwargs["research_model"] = main_kwargs["working_model"] = args.model

    print(f"📂 数据集: {dataset} | 数据路径: {data_path} | 输出: {save_path}")
    main(**main_kwargs)
