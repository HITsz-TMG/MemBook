# research_agent.py
# -*- coding: utf-8 -*-
"""
ResearchAgent Module

This module defines the ResearchAgent for the MemBook framework.

- ResearchAgent is responsible for research tasks, reasoning, and advanced information retrieval.
- It interacts with the MemoryAgent to store and access past knowledge as abstracts (memory is represented as a list[str], without events/tags).
- ResearchAgent uses explicit research functions to process queries and generate insights.
- Prompts within the module are placeholders for future extensions, such as customizable instructions or templates.

The module focuses on providing clear abstraction and extensible interfaces for research-related agent functionalities.
"""

from __future__ import annotations

from membook.compat import env

import os
import re
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

from membook.prompts import (
    Planning_PROMPT,
    Integrate_PROMPT,
    InfoCheck_PROMPT,
    GenerateRequests_PROMPT,
)
from membook.schemas import (
    MemoryState,
    SearchPlan,
    Hit,
    Result,
    Page,
    ReflectionDecision,
    ResearchOutput,
    MemoryStore,
    PageStore,
    Retriever,
    ToolRegistry,
    InMemoryMemoryStore,
    PLANNING_SCHEMA,
    INTEGRATE_SCHEMA,
    INFO_CHECK_SCHEMA,
    GENERATE_REQUESTS_SCHEMA,
)
from membook.generator import AbsGenerator
from membook.memory.toc.reconcile_context import format_integrate_evidence_block


def _general_qa_full_pipeline_enabled() -> bool:
    """When enabled, include general_qa chapter pages in planning/index retrieval."""
    raw = (env("MEMBOOK_GENERAL_QA_FULL_PIPELINE") or "").strip().lower()
    return raw in ("1", "true", "yes", "on", "full", "enabled")


def _research_debug_enabled() -> bool:
    return env("MEMBOOK_RESEARCH_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _chapter_body_for_page_index(content: Optional[str]) -> str:
    """
    Chap-ID / page_index 证据：去掉章节正文里为检索注入的 pending 后缀；
    pending 仍可由 BM25、向量与伪 MEMORY_POINT 命中（见 MemoryAgent）。
    """
    c = (content or "").strip()
    marker = "[SESSION_PENDING_FOR_RETRIEVAL]"
    i = c.find(marker)
    if i == -1:
        return c
    return c[:i].rstrip()


class ResearchAgent:
    """
    Public API:
      - research(request) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search(plan) -> SearchResults  (calls keyword/vector/page_id + tools)
      - _integrate(search_results, temp_memory) -> TempMemory
      - _reflection(request, memory_state, temp_memory) -> ReflectionDecision

    Note: Uses MemoryStore to dynamically load current memory state.
    This allows ResearchAgent to access the latest memory updates from MemoryAgent.
    """

    def __init__(
        self,
        page_store: PageStore,
        memory_store: MemoryStore | None = None,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        generator: AbsGenerator | None = None,  # 必须传入Generator实例
        max_iters: int = 3,
        max_deep_iters: int = 3,
        dir_path: Optional[str] = None,  # 新增：文件系统存储路径
        system_prompts: Optional[Dict[str, str]] = None,  # 新增：system prompts字典
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters
        self.max_deep_iters = max_deep_iters
        
        # 初始化 system_prompts，默认值为空字符串
        default_system_prompts = {
            "planning": "",
            "integration": "",
            "reflection": ""
        }
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:
            # 合并用户提供的 prompts 和默认值
            self.system_prompts = {**default_system_prompts, **system_prompts}

        # Build indices upfront (if retrievers are provided)
        for name, r in self.retrievers.items():
            try:
                # 调用 retriever 的 build 方法，传递 page_store
                r.build(self.page_store)
                print(f"Successfully built {name} retriever")
            except Exception as e:
                print(f"Failed to build {name} retriever: {e}")
                pass
        # 明确打印当前参与检索的检索器，便于确认 vector 是否真正启用
        active = list(self.retrievers.keys())
        print(f"[ResearchAgent] 参与检索的检索器: {active}" + (" (含 vector，将按 Planning 结果调用)" if "vector" in active else " (未包含 vector)"))
        print(f"[ResearchAgent] PageIndex（按 [k] 读页）: {'on' if self._use_page_index() else 'off (MEMBOOK_USE_PAGE_INDEX)'}")

    # ---- Public ----
    def research(self, request: str) -> ResearchOutput:
        # 在开始研究前，确保检索器索引是最新的
        self._update_retrievers()
        
        temp = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request
        last_hits: List[Hit] = []

        for step in range(self.max_iters):
            # Load current memory state dynamically
            memory_state = self.memory_store.load()
            plan = self._planning(next_request, memory_state)
            plan = self._finalize_search_plan_ablation(plan, request)

            temp, last_hits = self._search(plan, temp, request, deep_mode=False)

            decision = self._reflection(request, temp)

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
            })

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request

            # If phase-1 exhausted and still not enough, start deep phase.
            if step == self.max_iters - 1 and not decision.enough:
                break

        # Phase-2 deepresearch (3+3): only when phase-1 didn't reach enough.
        # We check by re-running reflection on the last temp content if phase-1 ended early.
        # If enough already, we won't enter deep mode.
        try:
            # If the last recorded decision was enough, we won't reach here.
            # When phase-1 loop ended via break (enough True), the raw decision isn't visible here.
            # So we recompute quickly with current temp.
            memory_state = self.memory_store.load()
            _ = memory_state  # silence unused warning
        except Exception:
            pass

        # Enter deep phase iff the last reflection decision in phase-1 was not enough.
        # We infer it by checking if the integrated content is empty or reflection would be false,
        # but we don't have that decision object; instead we use a heuristic:
        # deep phase is triggered if we exited phase-1 due to step == max_iters-1.
        # To keep behavior deterministic, we store whether step == max_iters-1 above.
        # (We do this by checking current step in iterations length.)
        if iterations:
            last_decision = iterations[-1].get("decision", {}).get("enough", False)
        else:
            last_decision = True

        if not last_decision and self.max_deep_iters > 0:
            # Candidates: chapter pages mapped from last_hits' chapter_id.
            candidate_chapter_page_indices = self._select_chapter_pages_from_hits(last_hits)
            deep_next_request = next_request
            for deep_step in range(self.max_deep_iters):
                memory_state = self.memory_store.load()

                # Let PlanningAgent choose tools/queries; keep three-route retrieval.
                plan_deep = self._planning(deep_next_request, memory_state)
                plan_deep = self._finalize_search_plan_ablation(plan_deep, request)
                # Deep phase: force/weight page_index for raw expansion (keep keyword/vector too).
                if self._use_page_index() and self._ablation_allows_tool("page_index"):
                    if "page_index" not in (plan_deep.tools or []):
                        plan_deep.tools = list(plan_deep.tools or []) + ["page_index"]
                    # Weight page_index by making it first in the tool list.
                    plan_deep.tools = ["page_index"] + [
                        t for t in (plan_deep.tools or []) if t != "page_index"
                    ]
                    if not plan_deep.page_index:
                        plan_deep.page_index = candidate_chapter_page_indices[:5]
                    else:
                        plan_deep.page_index = plan_deep.page_index[:5]

                temp, deep_last_hits = self._search(plan_deep, temp, request, deep_mode=True)
                decision = self._reflection(request, temp)

                iterations.append({
                    "step": f"deep_{deep_step}",
                    "plan": plan_deep.__dict__,
                    "temp_memory": temp.__dict__,
                    "decision": decision.__dict__,
                })

                if decision.enough:
                    break
                deep_next_request = decision.new_request or request


        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
        }
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    def _update_retrievers(self):
        """确保检索器索引是最新的"""
        # 检查是否有新的页面需要更新索引
        pages = self.page_store.load()
        current_page_count = len(pages)
        current_chapter_count = self._count_chapters(pages)
        
        # 如果页面数量发生变化，更新所有检索器索引
        if hasattr(self, '_last_page_count') and current_page_count != self._last_page_count:
            previous_chapter_count = getattr(self, "_last_chapter_count", current_chapter_count)
            print(
                "检测到页面数量变化 "
                f"({self._last_page_count} -> {current_page_count})，"
                "章节数量变化 "
                f"({previous_chapter_count} -> {current_chapter_count})，"
                "更新检索器索引..."
            )
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    print(f"✅ Updated {name} retriever index")
                except Exception as e:
                    print(f"❌ Failed to update {name} retriever: {e}")
        
        # 更新页面计数
        self._last_page_count = current_page_count
        self._last_chapter_count = current_chapter_count

    def _count_chapters(self, pages: List[Page]) -> int:
        """按 chapter_id 统计当前页面集合中的章节数（去重）。"""
        chapter_ids: set[str] = set()
        for page in pages:
            meta = page.meta or {}
            chapter_id = str(meta.get("chapter_id") or "").strip()
            if chapter_id:
                chapter_ids.add(chapter_id)
        return len(chapter_ids)

    def _use_page_index(self) -> bool:
        """
        是否启用 PageIndex 路径（按 MEMORY_CATALOG 的 chapter_index [k] 整页/确认点读取）。

        环境变量 ``MEMBOOK_USE_PAGE_INDEX``：未设置或 ``1``/``true``/``yes``/``on`` 为开启；
        ``0``/``false``/``no``/``off`` 为关闭（仅 keyword/vector）。
        """
        raw = (env("MEMBOOK_USE_PAGE_INDEX", "1") or "1").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        return True

    _RETRIEVAL_ABLATION_TABLE: Dict[str, frozenset[str]] = {
        "chap_id": frozenset({"page_index"}),
        "embedding": frozenset({"vector"}),
        "bm25": frozenset({"keyword"}),
        "emb_chap": frozenset({"vector", "page_index"}),
        "emb_bm25": frozenset({"vector", "keyword"}),
        "bm25_chap": frozenset({"keyword", "page_index"}),
    }

    def _retrieval_ablation_raw_mode(self) -> str:
        return (env("MEMBOOK_RETRIEVAL_ABLATION") or "").strip().lower()

    def _retrieval_ablation_allowed_tools(self) -> Optional[frozenset[str]]:
        key = self._retrieval_ablation_raw_mode()
        if not key or key in ("full", "none", "off"):
            return None
        allowed = self._RETRIEVAL_ABLATION_TABLE.get(key)
        if allowed is None:
            print(
                f"[ResearchAgent] MEMBOOK_RETRIEVAL_ABLATION={key!r} 未识别，不按消融过滤 tools"
            )
            return None
        return allowed

    def _ablation_allows_tool(self, tool: str) -> bool:
        allowed = self._retrieval_ablation_allowed_tools()
        if allowed is None:
            return True
        return tool in allowed

    def _finalize_search_plan_ablation(self, plan: SearchPlan, question: str) -> SearchPlan:
        allowed = self._retrieval_ablation_allowed_tools()
        if allowed is None:
            return plan
        seen: set[str] = set()
        filtered: List[str] = []
        for t in plan.tools or []:
            if t not in allowed or t in seen:
                continue
            filtered.append(t)
            seen.add(t)
        plan = plan.model_copy(update={"tools": filtered})
        if filtered:
            return plan
        mode = self._retrieval_ablation_raw_mode()
        if mode == "embedding":
            vq = list(plan.vector_queries) if plan.vector_queries else []
            if not vq:
                vq = [question]
            print("[ResearchAgent] 检索消融 embedding：规划 tools 为空，回退 vector")
            return plan.model_copy(update={"tools": ["vector"], "vector_queries": vq})
        if mode == "bm25":
            kc = list(plan.keyword_collection) if plan.keyword_collection else []
            if not kc:
                kc = [question]
            print("[ResearchAgent] 检索消融 bm25：规划 tools 为空，回退 keyword")
            return plan.model_copy(update={"tools": ["keyword"], "keyword_collection": kc})
        return plan

    def _qa_integrate_max_hits(self) -> Optional[int]:
        """
        QA：传入 ``_integrate`` 之前的命中条数上限。

        环境变量 ``MEMBOOK_QA_INTEGRATE_MAX_HITS``（默认 ``20``）。
        若解析为 ``<=0``，表示不截断。
        """
        raw = env("MEMBOOK_QA_INTEGRATE_MAX_HITS", "20").strip()
        try:
            n = int(raw)
        except ValueError:
            return 20
        if n <= 0:
            return None
        return n

    def _sort_hits_by_score(self, hits: List[Hit]) -> List[Hit]:
        """
        按 ``meta.score`` 降序排序；QA ``_search_collect_hits`` 与 Reconcile top-k 共用，避免两套 key 不一致。
        """
        if not hits:
            return []
        return sorted(
            hits,
            key=lambda h: h.meta.get("score", 0) if h.meta else 0,
            reverse=True,
        )

    # ---- Internal ----
    def _planning(
        self, 
        request: str, 
        memory_state: MemoryState,
        planning_prompt: Optional[str] = None
    ) -> SearchPlan:
        """
        Produce a SearchPlan:
          - what specific info is needed
          - which tools are useful + inputs
          - keyword/vector/page_id payloads
        """

        # 限制 memory 长度，避免 Planning prompt 超过模型 context 上限（128k tokens ≈ 512k chars）
        # 满打满算：预留 ~8k tokens 给 template+request+response，其余给 memory
        MAX_PLANNING_MEMORY_CHARS = 480_000  # ~120k tokens
        catalog = self._build_membook_planning_catalog(max_chars=MAX_PLANNING_MEMORY_CHARS)
        if catalog:
            memory_context = (
                "MEMORY_CATALOG (page_index must match global chapter_index [k] in brackets; "
                "[k] is unique across all sessions):\n" + catalog
            )
        elif not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
            if len(memory_context) > MAX_PLANNING_MEMORY_CHARS:
                memory_context = (
                    "[... earlier memory omitted for context length ...]\n\n"
                    + memory_context[-MAX_PLANNING_MEMORY_CHARS:]
                )
                if _research_debug_enabled():
                    print(f"[DEBUG] Planning memory truncated to {MAX_PLANNING_MEMORY_CHARS} chars (~{MAX_PLANNING_MEMORY_CHARS // 4} tokens)")
        
        system_prompt = self.system_prompts.get("planning")
        template_prompt = Planning_PROMPT.format(request=request, memory=memory_context)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt
        
        # 调试：打印prompt长度
        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4  # 粗略估算：1 token ≈ 4 字符
        if _research_debug_enabled():
            print(f"[DEBUG] Planning prompt length: {prompt_chars} chars (~{estimated_tokens} tokens)")

        try:
            response = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                # keyword_collection=[request],
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", [])
            )
        except Exception as e:
            print(f"Error in planning: {e}")
            return SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_index=[]
            )
    

    def _search(
        self, 
        plan: SearchPlan, 
        result: Result, 
        question: str,
        deep_mode: bool = False,
        searching_prompt: Optional[str] = None
    ) -> Tuple[Result, List[Hit]]:
        """
        Unified search with integration:
          1) Execute all search tools and collect all hits
          2) Deduplicate hits by page_id
          3) Integrate all deduplicated hits together with LLM
        Returns integrated Result.
        """
        _, sorted_hits = self._search_collect_hits(
            plan, result, question, deep_mode=deep_mode, top_k=5
        )
        if not sorted_hits:
            return result, []
        cap = self._qa_integrate_max_hits()
        if cap is not None:
            sorted_hits = sorted_hits[:cap]
        integrated = self._integrate(sorted_hits, result, question)
        return integrated, sorted_hits

    def _search_collect_hits(
        self,
        plan: SearchPlan,
        result: Result,
        question: str,
        *,
        deep_mode: bool = False,
        top_k: int = 5,
    ) -> Tuple[Result, List[Hit]]:
        """Collect and dedupe hits like _search, but without integration."""
        all_hits: List[Hit] = []

        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=top_k)
                    print(
                        f"[ResearchAgent] 已调用 keyword 检索，keywords={plan.keyword_collection[:5]}"
                        f"{'...' if len(plan.keyword_collection) > 5 else ''}"
                    )
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    print(f"[ResearchAgent] keyword 命中数: {len(hits)}")
                    all_hits.extend(hits)

            elif tool == "vector":
                vector_queries = plan.vector_queries if plan.vector_queries else [question]
                vector_results = self._search_by_vector(vector_queries, top_k=top_k)
                print(
                    f"[ResearchAgent] 已调用 vector 检索，queries={vector_queries[:3]}"
                    f"{'...' if len(vector_queries) > 3 else ''}"
                )
                if vector_results and isinstance(vector_results[0], list):
                    for result_list in vector_results:
                        hits.extend(result_list)
                else:
                    hits.extend(vector_results or [])
                all_hits.extend(hits)

            elif tool == "page_index":
                if self._use_page_index() and plan.page_index:
                    print(
                        f"[ResearchAgent] 已调用 page_index 检索，page_index={plan.page_index[:10]}"
                        f"{'...' if len(plan.page_index) > 10 else ''}，deep_mode={deep_mode}"
                    )
                    page_results = self._search_by_page_index(plan.page_index, deep_mode=deep_mode)
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    print(f"[ResearchAgent] page_index 命中数: {len(hits)}")
                    all_hits.extend(hits)

        if not all_hits:
            return result, []

        unique_hits: Dict[str, Hit] = {}
        hits_without_dedupe_key: List[Hit] = []
        for hit in all_hits:
            hmeta = hit.meta or {}
            mp_id = str(hmeta.get("memory_point_id") or "").strip()
            if mp_id:
                dedupe_key = f"mp:{mp_id}"
            elif hit.page_id:
                dedupe_key = f"page:{hit.page_id}"
            else:
                dedupe_key = ""

            if not dedupe_key:
                hits_without_dedupe_key.append(hit)
                continue

            if dedupe_key not in unique_hits:
                unique_hits[dedupe_key] = hit
            else:
                existing_hit = unique_hits[dedupe_key]
                existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                current_score = hmeta.get("score", 0) if hmeta else 0
                if current_score > existing_score:
                    unique_hits[dedupe_key] = hit

        all_unique_hits = list(unique_hits.values()) + hits_without_dedupe_key
        sorted_hits = self._sort_hits_by_score(all_unique_hits)
        return result, sorted_hits

    def _search_no_integrate(self, plan: SearchPlan, result: Result, question: str) -> Result:
        """
        Search without integration:
          1) Execute search tools
          2) Collect all hits without LLM integration
          3) Format hits as plain text results
        Returns Result with raw search hits formatted as content.
        """
        all_hits: List[Hit] = []

        # Execute each planned tool and collect hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    # 将多个关键词拼接成一个字符串进行搜索
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=5)
                    print(
                        f"[ResearchAgent] 已调用 keyword 检索，keywords={plan.keyword_collection[:5]}"
                        f"{'...' if len(plan.keyword_collection) > 5 else ''}"
                    )
                    # Flatten the results if they come as List[List[Hit]]
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    print(f"[ResearchAgent] keyword 命中数: {len(hits)}")
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                vector_queries = plan.vector_queries if plan.vector_queries else [question]
                vector_results = self._search_by_vector(vector_queries, top_k=5)
                print(f"[ResearchAgent] 已调用 vector 检索，queries={vector_queries[:3]}{'...' if len(vector_queries) > 3 else ''}")
                if vector_results and isinstance(vector_results[0], list):
                    for result_list in vector_results:
                        hits.extend(result_list)
                else:
                    hits.extend(vector_results or [])
                all_hits.extend(hits)
                    
            elif tool == "page_index":
                if self._use_page_index() and plan.page_index:
                    print(
                        f"[ResearchAgent] 已调用 page_index 检索，page_index={plan.page_index[:10]}"
                        f"{'...' if len(plan.page_index) > 10 else ''}，deep_mode=False"
                    )
                    page_results = self._search_by_page_index(plan.page_index)
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    print(f"[ResearchAgent] page_index 命中数: {len(hits)}")
                    all_hits.extend(hits)

        # Format all hits as text content without integration
        if not all_hits:
            return result
        
        # 按 page_id 去重 hits，避免同一个 page 被多个 tool 检索到时重复添加
        unique_hits: Dict[str, Hit] = {}  # page_id -> Hit
        hits_without_id: List[Hit] = []  # 没有 page_id 的 hits
        for hit in all_hits:
            if hit.page_id:
                # 如果这个 page_id 还没出现过，或者当前 hit 的得分更高（如果有的话），则更新
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    # 如果已有该 page_id 的 hit，比较得分（如果有的话），保留得分更高的
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                # 没有 page_id 的 hits 也保留
                hits_without_id.append(hit)
        
        evidence_text = []
        sources = []
        seen_sources = set()
        
        # 按得分排序（如果有的话），然后格式化
        # 合并有 page_id 和没有 page_id 的 hits
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = self._sort_hits_by_score(all_unique_hits)

        for i, hit in enumerate(sorted_hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            
            # Collect unique sources
            if hit.page_id and hit.page_id not in seen_sources:
                sources.append(hit.page_id)
                seen_sources.add(hit.page_id)
        
        formatted_content = "\n".join(evidence_text)
        
        return Result(
            content=formatted_content if formatted_content else result.content,
            sources=sources if sources else result.sources
        )

    def _integrate(
        self, 
        hits: List[Hit], 
        result: Result, 
        question: str,
        integration_prompt: Optional[str] = None
    ) -> Result:
        """
        Integrate search hits with LLM to generate question-relevant result.
        """
        
        sources = []
        for hit in hits:
            if hit.page_id:
                sources.append(hit.page_id)

        evidence_block = format_integrate_evidence_block(hits)
        evidence_context = evidence_block if evidence_block != "(none)" else "无搜索结果"
        
        system_prompt = self.system_prompts.get("integration")
        template_prompt = Integrate_PROMPT.format(question=question, evidence_context=evidence_context, result=result.content)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            response = self.generator.generate_single(prompt=prompt, schema=INTEGRATE_SCHEMA)
            raw_text = response.get("text") or ""
            data = response.get("json")
            if data is None and raw_text:
                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError as je:
                    # LLM 返回中常有非法 JSON 转义（如 content 里的路径 C:\、\x 等），导致 Invalid \escape
                    fixed = re.sub(r'\\(?![\\"/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', raw_text)
                    try:
                        data = json.loads(fixed)
                    except json.JSONDecodeError:
                        print(f"Error in integration: {je}")
                        return result
            if data is None:
                return result
            
            # 处理 sources：确保是字符串列表（如果LLM返回的是整数，转换为字符串）
            llm_sources = data.get("sources", sources)
            if llm_sources:
                # 将整数或混合类型转换为字符串列表
                sources_list = []
                for s in llm_sources:
                    if s is not None:
                        sources_list.append(str(s))
                sources = sources_list if sources_list else sources
            else:
                sources = sources
            
            return Result(
                content=data.get("content", ""),
                sources=sources
            )
        except Exception as e:
            print(f"Error in integration: {e}")
            return result

    def _build_membook_planning_catalog(self, max_chars: int = 480_000) -> str:
        """MYMEM: Session title + chapter title/summary; [k] == global chapter_index."""
        pages = self.page_store.load()
        lines: List[str] = []
        include_general = _general_qa_full_pipeline_enabled()
        current_sid: Optional[str] = None
        session_ord = -1
        fallback_chapter_index = -1
        for p in pages:
            meta = p.meta or {}
            if not meta.get("toc_mode"):
                continue
            lane = str(meta.get("memory_lane") or "")
            if lane == "general_qa" and not include_general:
                continue
            if lane not in ("user_durable", "general_qa"):
                continue
            chapter_index_raw = meta.get("chapter_index")
            try:
                chapter_index = int(chapter_index_raw)
            except Exception:
                fallback_chapter_index += 1
                chapter_index = fallback_chapter_index
            sid = str(meta.get("session_id") or "")
            stitle = str(meta.get("session_title") or sid).strip() or sid
            if sid != current_sid:
                current_sid = sid
                session_ord += 1
                lines.append(f"Session {session_ord}: {stitle}")
            ct = str(meta.get("chapter_title") or "").replace("\n", " ").strip()
            cs = str(meta.get("chapter_summary") or "").replace("\n", " ").strip()
            lines.append(f" - [{chapter_index}] ({lane}) {ct} - {cs}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = "[... truncated MEMORY_CATALOG ...]\n\n" + text[-max_chars:]
        return text

    def _select_chapter_pages_from_hits(self, hits: List[Hit], max_fragments: int = 5) -> List[int]:
        """
        Map point-level hits back to their chapter pages (toc_mode=True & user_durable).
        Used as candidate page_index list for deepresearch.
        Here page_index means chapter_index shown in MEMORY_CATALOG [k].
        """
        if not hits:
            return []
        chapter_ids: set[str] = set()
        for h in hits:
            meta = h.meta or {}
            chapter_id = meta.get("chapter_id")
            if chapter_id:
                chapter_ids.add(str(chapter_id))

        if not chapter_ids:
            return []

        # Scan main page_store and pick chapter_index by chapter_id.
        chapter_indices: List[int] = []
        pages = self.page_store.load()
        include_general = _general_qa_full_pipeline_enabled()
        fallback_chapter_index = -1
        for p in pages:
            meta = p.meta or {}
            if not meta.get("toc_mode"):
                continue
            lane = str(meta.get("memory_lane") or "")
            if lane == "general_qa" and not include_general:
                continue
            if lane not in ("user_durable", "general_qa"):
                continue
            chapter_index_raw = meta.get("chapter_index")
            try:
                chapter_index = int(chapter_index_raw)
            except Exception:
                fallback_chapter_index += 1
                chapter_index = fallback_chapter_index
            if str(meta.get("chapter_id") or "") in chapter_ids:
                chapter_indices.append(chapter_index)
        return chapter_indices[:max_fragments]

    def _resolve_chapter_page_by_index(self, index_value: int) -> Tuple[Optional[int], Optional[Page]]:
        """
        Resolve a page target from planner's page_index value.
        Preferred semantics: index_value == global chapter_index shown in MEMORY_CATALOG [k].
        Backward compatibility: if no chapter_index match, treat it as page_store index.
        """
        pages = self.page_store.load()
        include_general = _general_qa_full_pipeline_enabled()
        fallback_chapter_index = -1
        matches: List[Tuple[int, Page]] = []
        for store_idx, page in enumerate(pages):
            meta = page.meta or {}
            if not meta.get("toc_mode"):
                continue
            lane = str(meta.get("memory_lane") or "")
            if lane == "general_qa" and not include_general:
                continue
            if lane not in ("user_durable", "general_qa"):
                continue
            chapter_index_raw = meta.get("chapter_index")
            try:
                chapter_index = int(chapter_index_raw)
            except Exception:
                fallback_chapter_index += 1
                chapter_index = fallback_chapter_index
            if chapter_index == index_value:
                matches.append((store_idx, page))
        if len(matches) > 1:
            warnings.warn(
                f"Duplicate chapter_index={index_value} in page_store ({len(matches)} toc durable "
                f"matches); using the first (store_idx={matches[0][0]}).",
                UserWarning,
                stacklevel=2,
            )
        if matches:
            return matches[0]
        page = self.page_store.get(index_value)
        if page is None:
            return None, None
        return index_value, page

    def _find_toc_chapter_page_for_hit(self, hit: Hit) -> Optional[Page]:
        """Resolve user_durable chapter page from a point-level hit (BM25/vector meta has chapter_id)."""
        hm = hit.meta or {}
        chapter_id = str(hm.get("chapter_id") or "").strip()
        sid = str(hm.get("session_id") or "").strip()
        if not chapter_id and not sid:
            return None
        pages = self.page_store.load()
        include_general = _general_qa_full_pipeline_enabled()
        if chapter_id:
            for p in pages:
                pm = p.meta or {}
                lane = str(pm.get("memory_lane") or "")
                if pm.get("toc_mode") and lane in ("user_durable", "general_qa"):
                    if lane == "general_qa" and not include_general:
                        continue
                    if str(pm.get("chapter_id") or "") == chapter_id:
                        return p
        if sid:
            for p in pages:
                pm = p.meta or {}
                lane = str(pm.get("memory_lane") or "")
                if pm.get("toc_mode") and lane in ("user_durable", "general_qa"):
                    if lane == "general_qa" and not include_general:
                        continue
                    if str(pm.get("session_id") or "") == sid:
                        return p
        return None

    def _pending_hint_for_hit(self, hit: Hit, cache: Dict[str, str]) -> str:
        """
        Inject shared SessionTitle + SessionPending for integrate evidence.
        Works for toc_mode chapter hits and for point-level hits (resolves parent chapter page).
        """
        if not hit.page_id:
            return ""
        try:
            idx = int(hit.page_id)
        except Exception:
            return ""
        page = self.page_store.get(idx)
        pmeta = (page.meta or {}) if page else {}
        hit_meta = hit.meta or {}
        is_point = bool(str(hit_meta.get("memory_point_id") or "").strip())

        if pmeta.get("toc_mode"):
            source_meta = pmeta
        elif is_point or (page and not pmeta.get("toc_mode")):
            frag = self._find_toc_chapter_page_for_hit(hit)
            source_meta = (frag.meta or {}) if frag else {}
        else:
            return ""

        session_id = str(source_meta.get("session_id") or "").strip()
        if not session_id:
            return ""
        if session_id in cache:
            return cache[session_id]
        pending = source_meta.get("session_pending")
        if isinstance(pending, list):
            parts = [str(x).strip() for x in pending if str(x).strip()]
            text = ". ".join(parts) if parts else ""
        else:
            text = str(pending or "").strip()
        if not text:
            cache[session_id] = ""
            return ""
        session_title = str(source_meta.get("session_title") or "").strip()
        prefix = f"SessionTitle:\n{session_title}\n" if session_title else ""
        block = prefix + "SessionPending:\n" + text
        cache[session_id] = block
        return block

    # ---- search channels ----
    def _search_by_keyword(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:
                # BM25Retriever 返回 List[List[Hit]]
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in keyword search: {e}")
                return []
        # naive fallback: scan pages for substring
        out: List[List[Hit]] = []
        for query in query_list:
            query_hits: List[Hit] = []
            q = query.lower()
            for i, p in enumerate(self.page_store.load()):
                if q in p.content.lower() or q in p.header.lower():
                    snippet = p.content
                    query_hits.append(Hit(page_id=str(i), snippet=snippet, source="keyword", meta={}))
                    if len(query_hits) >= top_k:
                        break
            out.append(query_hits)
        return out

    def _search_by_vector(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in vector search: {e}")
                return []
        # fallback: none
        return []

    def _search_by_page_index(self, page_index: List[int], *, deep_mode: bool = False) -> List[List[Hit]]:
        # We intentionally bypass IndexRetriever here so we can control evidence granularity:
        # - normal mode: PageIndex returns ONLY confirmed memory points (not raw chapter text)
        # - deep mode: PageIndex expands raw chapter content (with chapter hint); pending 后缀不进入 Chap-ID
        out: List[Hit] = []
        include_general = _general_qa_full_pipeline_enabled()
        for idx in page_index:
            store_idx, p = self._resolve_chapter_page_by_index(idx)
            if store_idx is None or not p:
                continue
            meta = p.meta or {}
            if not meta.get("toc_mode"):
                # Only toc_mode chapter pages are valid page_index targets.
                continue
            lane = str(meta.get("memory_lane") or "")
            if lane == "general_qa" and not include_general:
                continue
            if lane not in ("user_durable", "general_qa"):
                continue

            chapter_title = str(meta.get("chapter_title") or "").strip()
            chapter_summary = str(meta.get("chapter_summary") or "").strip()
            hint = ""
            if chapter_title or chapter_summary:
                hint = f"[CHAPTER_HINT] {chapter_title}"
                if chapter_summary:
                    hint += f" | {chapter_summary}"
                hint += "\n"

            if deep_mode:
                body = _chapter_body_for_page_index(getattr(p, "content", None))
                snippet = f"{hint}{body}".strip()
            else:
                pts = meta.get("confirmed_memories") or []
                lines: List[str] = []
                seen: set[str] = set()
                if isinstance(pts, list):
                    for it in pts:
                        if not isinstance(it, dict):
                            continue
                        mp_id = str(it.get("memory_point_id") or "").strip()
                        text = str(it.get("text") or "").strip()
                        if not text:
                            continue
                        key = mp_id or text.lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        lines.append(f"- {text}")
                snippet = (hint + "\n".join(lines)).strip() if lines else (hint + "(no confirmed points)").strip()

            out.append(
                Hit(
                    page_id=str(store_idx),
                    snippet=snippet,
                    source="page_index",
                    meta={
                        "chapter_id": str(meta.get("chapter_id") or "").strip(),
                        "chapter_index": int(meta.get("chapter_index")) if str(meta.get("chapter_index") or "").strip().isdigit() else idx,
                        "session_id": str(meta.get("session_id") or "").strip(),
                        "memory_lane": str(meta.get("memory_lane") or "").strip(),
                        "deep_mode": bool(deep_mode),
                    },
                )
            )
        return [out]
        
        

    # ---- reflection & summarization ----
    def _reflection(
        self, 
        request: str, 
        result: Result,
        reflection_prompt: Optional[str] = None
    ) -> ReflectionDecision:
        """
        - "whether information is enough" 
        - "if not, generate remaining information as a new request"  
        """
        
        try:
            system_prompt = self.system_prompts.get("reflection")
            
            # 调试：打印reflection prompt长度
            result_content_chars = len(result.content)
            estimated_result_tokens = result_content_chars // 4
            if _research_debug_enabled():
                print(f"[DEBUG] Reflection result.content length: {result_content_chars} chars (~{estimated_result_tokens} tokens)")
            
            # Step 1: Check for completeness of information
            template_check_prompt = InfoCheck_PROMPT.format(request=request, result=result.content)
            if system_prompt:
                check_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_check_prompt}"
            else:
                check_prompt = template_check_prompt
            check_prompt_chars = len(check_prompt)
            estimated_check_tokens = check_prompt_chars // 4
            if _research_debug_enabled():
                print(f"[DEBUG] Reflection check_prompt length: {check_prompt_chars} chars (~{estimated_check_tokens} tokens)")
            
            check_response = self.generator.generate_single(prompt=check_prompt, schema=INFO_CHECK_SCHEMA)
            check_data = check_response.get("json")
            if check_data is None:
                raw_text = (check_response.get("text") or "").strip()
                try:
                    check_data = json.loads(raw_text) if raw_text else {}
                except json.JSONDecodeError:
                    print(f"[WARN] Reflection check 返回非 JSON，已忽略。raw_text 前 200 字: {raw_text[:200]!r}")
                    check_data = {}
            enough = check_data.get("enough", False)
            
            # If there is enough information, return directly
            if enough:
                return ReflectionDecision(enough=True, new_request=None)
            
            # Step 2: Generate follow-up retrieval questions from REQUEST + RESULT (no MEMORY_CATALOG).
            template_generate_prompt = GenerateRequests_PROMPT.format(
                request=request,
                result=result.content,
            )
            if system_prompt:
                generate_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_generate_prompt}"
            else:
                generate_prompt = template_generate_prompt
            generate_prompt_chars = len(generate_prompt)
            estimated_generate_tokens = generate_prompt_chars // 4
            if _research_debug_enabled():
                print(f"[DEBUG] Reflection generate_prompt length: {generate_prompt_chars} chars (~{estimated_generate_tokens} tokens)")
            
            generate_response = self.generator.generate_single(prompt=generate_prompt, schema=GENERATE_REQUESTS_SCHEMA)
            generate_data = generate_response.get("json")
            if generate_data is None:
                raw_text = (generate_response.get("text") or "").strip()
                try:
                    generate_data = json.loads(raw_text) if raw_text else {}
                except json.JSONDecodeError:
                    print(f"[WARN] Reflection generate_requests 返回非 JSON，已忽略。raw_text 前 200 字: {raw_text[:200]!r}")
                    generate_data = {}
            # Get the list of requests and convert to string
            new_requests_list = generate_data.get("new_requests", [])
            new_request = None
            
            if new_requests_list and isinstance(new_requests_list, list):
                new_request = " ".join(new_requests_list)
            
            return ReflectionDecision(
                enough=False,
                new_request=new_request
            )
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            return ReflectionDecision(enough=False, new_request=None)