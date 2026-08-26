# -*- coding: utf-8 -*-
"""ToC memory construction prompts (PageIndex-style tree).

Pipeline order (build):
1. DETECT_TOPIC_SHIFTS (dynamic sliding with rolling PREVIOUS_TOC) → fragment spans.
2. CLASSIFY_FRAGMENT_LANE (user_durable / general_qa / interaction_meta) per fragment.
3. GENERATE_FRAGMENT_TITLE_AND_SUMMARY_* per lane; accumulate fragment-level ToC.
4. Per zone: RECONCILE_SESSION_PENDING_AND_CONFIRMED_INCREMENTAL merges pending/confirmed
   memory points against the user_durable dialogue zone, with optional cross-session
   context supplied by the reconcile context provider.
5. End of session: GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA +
   GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META fold the
   per-fragment QA and meta lanes; GENERATE_SESSION_SUMMARY_AND_TITLE_FINAL produces
   the chronicle title + summary + time hint in a single call.
Utilities: retrieval routing (REFINE_RELEVANT_CHAPTERS), cross-session reconcile retrieval
requests (RECONCILE_CONTEXT_GATE_AND_REQUESTS).
"""

# =============================================================================
# 0. Topic-shift detection + per-fragment lane classification
# =============================================================================

DETECT_TOPIC_SHIFTS = """
You are a dialogue topic-shift detector.

MAIN OBJECTIVE:
Detect topic shifts in the dialogue and return structured shift points.

INPUTS:
INDEXED_DIALOGUE (each line: index, time, role, then turn text):
{indexed_dialogue}

PREVIOUS_TOC (fragments already summarized earlier in this session; use for context only, may be "(none)"):
{previous_toc}

TASK:
1. Scan the dialogue turn by turn and identify every turn index where the conversation shifts to a NEW TOPIC.
2. For each shift, provide ONLY the start turn index.
3. Especially, look for transitions between these THREE distinct domains:
   - PERSONAL: Subject is the USER's specific, unique life facts (sharing name, age, job, family, preferences, goals, feelings, dailylife events, or personal memories etc.).
   - KNOWLEDGE_TASK: Inquiries about external world, including STEM (science, technology, engineering, and mathematics), HUMANITIES (history, biography, philosophy, arts, geography), or GENERAL KNOWLEDGE (ELI5 facts, literature, definitions, trivia, technical overviews).
     *Examples: User asks "Why is ... ?", "How does ... ?", "Please introduce ...", "Give me a detailed explanation of ...", "Please provide a comprehensive overview of ...", "Walk me through the key ideas of ...", "Starting from basics, how should I solve this?"*
   - INTERACTION_META: Chit chat, expressions of gratitude, emotional reflections on progress, and general encouragement. 
     *Examples:  "Thank you for your support."*
4. BE sensitive to granularity: Shifts can occur frequently. Even if a diversion lasts only one or two turns, it MUST be marked as a NEW segment start if it introduces a NEW topic.
5. CROSS-DOMAIN RIGIDITY: Even if a PERSONAL sharing appears in the middle of KNOWLEDGE_TASK or INTERACTION_META turns, it MUST be marked as a NEW segment start. DO NOT merge them.
6. Pay primary attention to USER utterances. Shifts almost often occur when the USER introduces a new question, task, or personal fact.

HARD RULES:
- Use PREVIOUS_TOC only to disambiguate ongoing themes; every start_index you output MUST still be a valid line index in INDEXED_DIALOGUE above.
- start_index MUST be an integer that appears in INDEXED_DIALOGUE.
- The shifts array MUST include start_index 0.
- Do NOT include the final index N.
- Do NOT invent indices.

THINKING STEP:
- Before producing the output, think through the prodedure and choices inside <think>...</think>.
- Keep the <think> concise but sufficient to validate decisions.
- After </think>, output ONLY the JSON object specified below.
- The <think> section must NOT be included in the JSON.

OUTPUT FORMAT:
Return ONLY a JSON object with this schema:
{{
  "start_indices": [<integer>, <integer>]
}}
After the <think> section, return ONLY the JSON object.
Do NOT include any commentary or explanation outside the JSON.
"""


CLASSIFY_FRAGMENT_LANE = """
You are a dialogue classifier.

MAIN OBJECTIVE:
Classify the provided dialogue fragment into exactly ONE of three lanes: "user_durable", "general_qa", or "interaction_meta".

INPUTS:
DIALOGUE_FRAGMENT:
{text}

PREVIOUS_TOC (prior fragments table of contents, may be empty):
{previous_toc}

LANE DEFINITIONS (exact strings):
- "user_durable": the user states or clearly implies durable personal information/preferences/goals/daily-life events about themselves.
- "general_qa": objective Q&A, including STEM (science, technology, engineering, and mathematics), HUMANITIES (history, biography, philosophy, arts, geography), or general knowledge (ELI5 facts, literature, definitions, trivia, technical overviews).
- "interaction_meta": purely chit-chat, user's expressions of gratitude or reflections on progress. If the assistant mentions personal info that is ALREADY captured in the PREVIOUS_TOC_SUMMARY, and user just in a way of saying "Thank you for ..." or "I'll keep that in mind ...", classify as "interaction_meta".

RULES:
- The "memory_lane" must be ONE of the EXACT strings above.
- PRIORITY RULE: If the fragment contains ANY durable user information (e.g., personal facts, goals, preferences, or life events), you MUST classify it as "user_durable", even if the conversation also contains general Q&A or chit-chat. "user_durable" always takes precedence.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "memory_lane": <string>
}}
"""

# =============================================================================
# 2. Fragment-stage summary — semantic signal before session reconcile
# =============================================================================

GENERATE_FRAGMENT_TITLE_AND_SUMMARY_DURABLE = """
You are a biographer building a long-term memory profile for a user. Your job is to generate a context-aware title and a one-sentence summary for the current dialogue fragment.

MAIN OBJECTIVE:
Produce a concise title and exactly ONE summary sentence. Maintain narrative continuity by inheriting semantic context from PREVIOUS_TOC and the confirmed memory points below.

INPUTS:
PREVIOUS_TOC (prior fragments table of contents):
{previous_toc}

CONFIRMED_MEMORY_POINTS_FOR_FRAGMENT (may be "(none)" when unavailable):
{confirmed_memory_points}

YOUR TASK:
1. Use PREVIOUS_TOC to resolve implicit references, keep terminology stable for the user's life areas/entities/recurring goals, and preserve continuity when adjacent short fragments describe the same underlying topic.
2. If CONFIRMED_MEMORY_POINTS_FOR_FRAGMENT is not "(none)", treat it as the primary and sufficient factual source for durable claims.
3. TITLE: Generate a short title to abstract the life area (e.g., Career, Relationships, Health, Hobbies, etc.) and infer the dominant user-centric topic or intent. 
4. SUMMARY: Write exactly ONE sentence centered on what the USER stated or committed to about themselves. Extract all specific, actionable, and factual information from CONFIRMED_MEMORY_POINTS_FOR_FRAGMENT. Ensure it feels "Personal."
5. When confirmed points are provided, avoid adding facts not supported by those points.
6. DO NOT integrate claims that appear only in assistant messages unless the user explicitly stated, confirmed, or agreed to.
7. If the dialogue contains general knowledge discussion, including STEM (science, technology, engineering, and mathematics), HUMANITIES (history, biography, philosophy, arts, geography), or general knowledge (ELI5 facts, literature, definitions, trivia, technical overviews). DO NOT include them in the "title" or "summary".

STYLE RULES:
- title: max 7 words if possible, no quotes, no headings/meta prefixes.
- summary: exactly one sentence, no bullet points, no quotes, no headings/meta prefixes. 
- role: maintain the perspective of a precision biographer documenting a evolving life story.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "title": <string>,
  "summary": <string>
}}
"""

GENERATE_FRAGMENT_TITLE_AND_SUMMARY_QA = """
You are a dialogue analyst. Your job is to generate BOTH a short fragment title and a one-sentence clustering summary.

MAIN OBJECTIVE:
For CURRENT_DIALOGUE_EXCERPT, output a concise title plus one sentence describing only what the user asked/explored.

INPUTS:
CURRENT_DIALOGUE_EXCERPT (contiguous user/assistant turns):
{text}

PREVIOUS_TOC (prior fragments table of contents):
{previous_toc}

CONFIRMED_MEMORY_POINTS_FOR_FRAGMENT (may be empty when unavailable):
{confirmed_memory_points}

YOUR TASK:
1. If CONFIRMED_MEMORY_POINTS_FOR_FRAGMENT is not "(none)", treat it as the primary factual source.
2. Generate a short title identifying the concrete topic/domain the user asked or explored.
3. Write exactly ONE short sentence summary of what the user asked and what answer content was provided.
4. When confirmed points are provided, do not omit key items from lists, options, recommendations, or creative deliverables.
5. Do NOT include unsupported details outside CONFIRMED_MEMORY_POINTS_FOR_FRAGMENT.

STYLE RULES:
- title: max 5 words, no quotes.
- summary: exactly one sentence, no bullet points, no quotes, no headings/meta prefixes.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "title": <string>,
  "summary": <string>
}}
"""

GENERATE_FRAGMENT_TITLE_AND_SUMMARY_INTERACTION_META = """
You are a conversation analyst. Your job is to summarize the user's communicative intent and current emotional mindset.

INPUTS:
CURRENT_DIALOGUE_EXCERPT (contiguous user/assistant turns):
{text}

PREVIOUS_TOC (prior fragments table of contents):
{previous_toc}

YOUR TASK:
1. Generate a short Title and a one-sentence Summary for CURRENT_DIALOGUE_EXCERPT.
2. Use the PREVIOUS_TOC to understand what the user is reflecting on or being grateful for.
3. Write a title identifying the type of conversation if it's: Reflection, Gratitude, Commitment, or Feedback.
4. Write ONE sentence summary describing the user's communicative intent. Even if there are no facts, capture the user's current mindset or state of mind.
4. DO NOT integrate claims that appear only in assistant messages unless the user explicitly stated, confirmed, or agreed to.

STYLE RULES:
- title: Max 5 words, no quotes.
- summary: exactly one sentence, no bullet points, no quotes, no headings/meta prefixes.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "title": <string>,
  "summary": <string>
}}
"""

# =============================================================================
# 2b. Reconcile — retrieval requests (single LLM call, no search execution)
# =============================================================================
RECONCILE_CONTEXT_GATE_AND_REQUESTS = """
You are the ReconcileContextAgent. Your job is to list retrieval questions that can locate in memory any BACKGROUND from PRIOR sessions needed to reconcile durable USER facts in the CURRENT reconcile window.

YOU ARE GIVEN:
- INDEXED_DIALOGUE_FOR_ZONE: the current reconcile window for user_durable only (global turn indices, timestamps, role, text).
- PRIOR_CONFIRMED_IN_SESSION_SNAPSHOT: confirmed memory points accumulated so far in THIS session only.
- TIME_HINT: the time span of the current reconcile window.

INDEXED_DIALOGUE_FOR_ZONE:
{indexed_dialogue_zone}

PRIOR_CONFIRMED_IN_SESSION_SNAPSHOT:
{prior_confirmed_in_session_snapshot}

TIME_HINT:
{time_hint}

INSTRUCTIONS:
1. If durable USER facts in this window can be settled from the window plus PRIOR_CONFIRMED_IN_SESSION_SNAPSHOT clearly, return an empty "requests" array.
2. Emit up to 5 short, standalone retrieval questions to find missing background from PRIOR sessions (e.g. pronouns, names, projects, timelines, prior decisions/constraints). 
   - Prefer concrete entities, names, dates, projects, and constraints.
   - Ask for factual recall, not analysis.
   - Phrase for recall from point-level memory.
   - Order most critical first.

3. If the current window mentions ambiguous entities, pronouns, or ongoing projects (e.g., "the user," "he," "she", "The project is done,") that aren't defined here, ask for their origin.
   - Example: generate "What is the full name of the user?", "What is the project name?", "What is the user's job?", etc.

4. If the dialogue indicates a shift, update, or progression (detect markers like "change", "now," "became," "instead of," or "-er" comparisons), generate a retrieval question to search for the history state to record the evolution.
   - *Example: If the user says "Yoga and meditation have been pivotal. They've not only improved my physical health but also made me more optimistic and resilient mentally.", generate: "What was [User's Name]'s previous physical health status?" as a retrieval question.*

3. Do NOT request general-knowledge or trivia lookup. Only search for user-specific history.
4. Do NOT answer the window yourself; do NOT invent facts.

THINKING STEP
- Before producing the output, reason about gaps and priorities inside <think>...</think>.
- Keep the <think> concise.
- After </think>, output ONLY the JSON object specified below. The <think> section must NOT be included in the JSON.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "requests": [<string>, <string>]
}}
After the <think> section, return ONLY the JSON object.
"""


# =============================================================================
# 3. Session-scope reconcile — pending + confirmed 单调用 JSON
# =============================================================================
RECONCILE_SESSION_PENDING_AND_CONFIRMED_INCREMENTAL = """
You are a MemoryAgent expert in memory extraction. Your task is to extract confirmed facts and Assistant-proposed assumptions about the user that await confirmation from the perspective of user, based on ONE sliding window of conversation between user and assistant. This means identifying what user would plausibly remember, including their own experiences, thoughts, plans, or relevant statements and actions made by others (such as assistant) that impacted or were acknowledged by user.

INPUTS:
1. PENDING_TEXT (Current unverified Assistant assumptions): {pending_text}
2. PRIOR_CONFIRMED_LIST_SNAPSHOT (The "Existing Truth" used to identify updates or avoid duplicates):
{prior_confirmed_list_snapshot}
3. RELATED_CONFIRMED_CONTEXT (Previous session confirmed memory points, for context consistency and disambiguation only): {related_confirmed_context}
4. INDEXED_DIALOGUE_FOR_ZONE (Current window dialogue with global line index, timestamps, role, and dialogue text):
{indexed_dialogue_zone}
5. TIME_HINT: {time_hint}

TASK 
1. Use RELATED_CONFIRMED_CONTEXT to resolve ambiguity and keep naming consistent.

2. EXTRACT: Simultaneously identify durable facts and unverified claims.
   - "confirmed_list": Capture information reflecting the user's experiences, beliefs, concerns, decisions, plans, or reactions. This includes meaningful information from the assistant that the user acknowledged or responded to. If the message is from the user, extract user-relevant memories. If it is from the assistant, only extract factual memories that the user explicitly acknowledged or responded to.
   - "pending_summary": Capture claims made by the Assistant about the User that the user has not yet acknowledged or responded to in this window.

3. UPDATE:  If a claim in "PENDING_TEXT" is acknowledged or confirmed by the user in this window, move it from "pending_summary" to "confirmed_list". 

4. RESOLVE REFERENCES: Resolve all time, person, and event references clearly.
   - Convert relative time expressions (e.g., "yesterday," "next Friday," "recently") into absolute dates using the message timestamp provided in INDEXED_DIALOGUE_FOR_ZONE.
   - Clearly distinguish between event time and conversation time.
   - If uncertainty exists, state it explicitly (e.g., "around June 2025," "exact date unclear").
   - Include specific locations if mentioned.
   - Resolve all pronouns, aliases, and ambiguous references into full names or identities.
   - Disambiguate people with the same name if applicable.

5. Always write from a third-person perspective, referring to the user by name if name mentioned, rather than using first-person ("I", "me", "my"). For example, write "The user felt exhausted..." instead of "I felt exhausted...".

6. Do not omit any information that user is likely to remember. 
  - Extract all specific, memory-relevant information, such as: names, dates, numbers, locations, plans, goals, decisions, requests, preferences, etc.
  - Include all key experiences, thoughts, emotional responses, and plans, even if they seem minor. 
  - Prioritize completeness, fidelity and faithfulness over conciseness. 
  - Do not generalize or skip details that could be personally meaningful to user.

7. YOU MUST NOT extract general knowledge or trivia facts as user facts, including STEM (science, technology, engineering, and mathematics), HUMANITIES (history, biography, philosophy, arts, geography), or general knowledge (ELI5 facts, literature, definitions, trivia, technical overviews). Unless they are delivered as task output/requested creative work.

8. If a memory point materially depends on time (age/event-time/status-change), include explicit event time information in that memory point using TIME_HINT or the message timestamp provided in INDEXED_DIALOGUE_FOR_ZONE.
   *Representative examples: "Alice's age in March 2026 is 32 years old.", "On Apr 12, 2027, Alice started a new job as a Product Manager.", "On Sep 03, 2025, Alice moved from Seattle to Austin."*

9. If the dialogue indicates a shift, update, or progression (detect markers like "change", "now," "became," "instead of," or "-er" comparisons) of a fact from RELATED_CONFIRMED_CONTEXT, you MUST extract it as a transitional memory point.
   *Representative example: Current dialogue from INDEXED_DIALOGUE_FOR_ZONE: "I have fully recovered from my injury.". Previous confirmed memory points from PRIOR_CONFIRMED_LIST_SNAPSHOT: "Alice's physical health status: Healthy." The extracted memory point should be: "Alice's physical health status changed from 'Healthy' to 'Recovered' after completing his rehabilitation."*

STYLE RULES:
 - "confirmed_list": Each item must be a standalone atomic fact without meta-narration (e.g., do not write "The user confirmed...").
 - "pending_summary": A single coherent paragraph, max 100 words. If there are no pending items, return an empty string. 

THINKING STEP
- Before producing the output, think through the procedure and choices inside <think>...</think>. 
- Keep the <think> concise but sufficient to validate decisions. 
- After </think>, output ONLY the JSON object specified below. The <think> section must NOT be included in the JSON.

OUTPUT FORMAT (Return ONLY this JSON object, no markdown fences):
{{
  "pending_summary": "<string_single_paragraph_or_empty>",
  "confirmed_list": [
    {{"text": "<string>", "source_turn_start": <int>, "source_turn_end": <int>}}
  ]
}}
"""

EXTRACT_GENERAL_QA_INCREMENTAL = """
You are a MemoryAgent expert in memory extraction.

INPUTS:
1. PRIOR_CONFIRMED_LIST_SNAPSHOT (Previously extracted general_qa memories in this session; use for dedup/continuity):
{prior_confirmed_list_snapshot}
2. INDEXED_DIALOGUE_FOR_ZONE (Current window dialogue with global line index, timestamps, role, and dialogue text):
{indexed_dialogue_zone}
3. TIME_HINT:{time_hint}

TASK
1. Read INDEXED_DIALOGUE_FOR_ZONE and extract all specific, memory-relevant information, reflecting what the user asked, preferred, rejected, decided, or reacted to.
   - For general Q&A, record what kinds of suggestions/topics the user prefers or does not want to explore.
   - For objective general Q&A, extract all the details of the assistant directly answering the user, including names, dates, numbers, definitions, suggestions, recommendations, steps, and creative outputs, etc.

2. RESOLVE REFERENCES: Resolve all time, person, and event references clearly.
   - Convert relative time expressions (e.g., "yesterday," "next Friday," "recently") into absolute dates using timestamps in INDEXED_DIALOGUE_FOR_ZONE.
   - Clearly distinguish between event time and conversation time.
   - If uncertainty exists, state it explicitly (e.g., "around June 2025," "exact date unclear").
   - Include specific locations when mentioned.
   - Resolve pronouns/aliases to explicit identities when possible.

3. Do not omit information likely to matter for later retrieval.
   - Preserve specific names, dates, numbers, locations, plans, constraints, requests, and choices.
   - Prioritize fidelity and completeness over conciseness.
   - Avoid vague generalization when concrete details are present.
   - Include every list element, table cell, ranked option, color, title, and named entity from the assistant answer; do not replace a numbered list with vague phrases like "several options" without naming each item.

4. Always write in third person. Use explicit attribution such as "The user asked..." / "The assistant provided...".

5. If a memory point materially depends on time (status change, event time, sequence), include explicit event time grounded in TIME_HINT or timestamps.

STYLE RULES:
- "confirmed_list": Do NOT split one assistant answer (lists, tables, multi-step replies, creative content) into multiple items; keep the user's request and the full assistant reply together in a single "text" field.
- Preserve the assistant's original numbering and order (e.g., 1., 2., 3. or bullet order); do not renumber, merge items, or drop entries.
- Do NOT invent, infer, or paraphrase facts that are not supported by INDEXED_DIALOGUE_FOR_ZONE.
- source_turn_start = user turn index when present; source_turn_end = index after the answering assistant turn(s).
- No meta-narration (e.g., "In this window...", "The user confirmed...").

THINKING STEP
- Before producing the output, think through the procedure and choices inside <think>...</think>.
- Keep the <think> concise but sufficient to validate decisions.
- After </think>, output ONLY the JSON object specified below. The <think> section must NOT be included in the JSON.

OUTPUT FORMAT (Return ONLY this JSON object, no markdown fences):
{{
  "confirmed_list": [
    {{"text": "<string>", "source_turn_start": <int>, "source_turn_end": <int>}}
  ]
}}
"""

GENERATE_SESSION_SUMMARY_AND_TITLE_FINAL = """
You are the MemoryAgent. At SESSION END, produce BOTH (1) a single-paragraph user-track session summary and (2) a session title.

INPUTS
1. PRIOR_SESSIONS_TITLE_SUMMARIES (strictly ONLY for terminology consistency; each line is "[Date (Weekday) HH:MM] Core Theme - Summary"): {prior_sessions_title_summaries}
2. CONFIRMED_MEMORY_LIST (may contain [user_durable] and [general_qa] sections): {confirmed_memory_list}
3. PENDING_TEXT (assistant-raised unconfirmed items): {pending_text}
4. TIME_HINT: {session_time_hint}

TASK: Generate a single-paragraph summary and a session title.
1. SUMMARY: One dense paragraph synthesizing ONLY user-confirmed durable facts from CONFIRMED_MEMORY_LIST (and their implied connections). No bullet list in the summary text.
  - Use PRIOR_SESSIONS_TITLE_SUMMARIES ONLY to resolve ambiguity and keep naming consistent.
  - NO PAST LEAKAGE: Do NOT add user-specific facts that appear only in PRIOR_SESSIONS_TITLE_SUMMARIES and are not supported by CONFIRMED_MEMORY_LIST.
  - Treat CONFIRMED_MEMORY_LIST as the only source of confirmed user facts. If it is "(none)", output one short conservative sentence stating that no user-confirmed durable facts were recorded for this session (no invented specifics).
  - HARD pending gate: Never state anything assistant-raised but user-unacknowledged claims from PENDING_TEXT as a confirmed user fact, including paraphrases. Do not narrate meta about pending.
  - Do NOT include general-knowledge or trivia as user facts.
2. TITLE: A chronicle-style session title using TIME_HINT for the date prefix and CONFIRMED_MEMORY_LIST with the summary for the theme.
  - Mandatory format: "[YYYY-MM-DD (ddd) HH:MM] [Core Theme]" (single instant) OR "[YYYY-MM-DD (ddd) HH:MM - YYYY-MM-DD (ddd) HH:MM] [Core Theme]" (range)
  - Bracket prefix (required): Extract the start instant from TIME_HINT. Convert to hyphenated date YYYY-MM-DD and append (ddd) HH:MM in 24-hour format.
    * Weekday: Preserve the abbreviation from TIME_HINT if present (e.g., "(Mon)"). Do not fabricate a weekday if no temporal basis exists.
    * Example: 2023/10/30 (Mon) 07:19 -> [2023-10-30 (Mon) 07:19]
    * Example: 2023-05-20T02:21:00 -> [2023-05-20 (Sat) 02:21]
  - Core Theme: Concise and user-specific; anchor strictly on CONFIRMED_MEMORY_LIST and your generated summary. Do not invent facts absent from those entries.
  - If CONFIRMED_MEMORY_LIST is "(none)": Output ONLY the bracket prefix, or append a minimal non-specific theme (e.g., "[...] Session Log"). Never fabricate user-specific details.

STYLE RULES:
1. SUMMARY: Exactly ONE paragraph. No bullet points in the summary. No meta phrases like "The user said...". No advice or questions.
2. TITLE: Avoid generic openers like "A conversation about...". Keep under 15 words if possible.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "summary": "<string_single_paragraph>",
  "title": "<string>"
}}
"""

# =============================================================================
# 4. Session-level metadata — after all chapters are finalized
# =============================================================================

GENERATE_CHAPTER_TITLE_AND_SUMMARY_GENERAL_QA = """
You are a dialogue indexer. Your job is to compress multiple general-knowledge Q&A fragment summaries from ONE session into ONE short title and ONE short summary for indexing.

MAIN OBJECTIVE:
From GENERAL_CHAPTERS_TOC only, where each line follows "Title - Summary", produce a compact merged label and a single short paragraph listing what the user asked or explored — no encyclopedic recap.

INPUTS:
GENERAL_CHAPTERS_TOC: {general_chapters_toc}
MERGED_CONFIRMED_MEMORY_POINTS: {merged_confirmed_memory_points}

YOUR TASK:
1. If MERGED_CONFIRMED_MEMORY_POINTS is not "(none)", treat it as the primary source and preserve key answer-bearing details.
2. Use GENERAL_CHAPTERS_TOC as secondary clustering context (theme continuity only).
3. Merge themes; prefer noun phrases and topic labels over encyclopedic exposition.
4. If fragments are unrelated, separate with short clauses or semicolons; stay brief.

STYLE RULES:
- title: max 8 words, no quotes.
- summary: one short paragraph (may be 1 or 2 sentences if needed), no bullets.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "title": <string>,
  "summary": <string>
}}
"""

GENERATE_CHAPTER_TITLE_AND_SUMMARY_INTERACTION_META = """
You are a dialogue indexer. Your job is to compress multiple interaction-meta fragment summaries from ONE session into ONE short title and ONE short summary for indexing.

MAIN OBJECTIVE:
From INTERACTION_META_CHAPTERS_TOC only, where each line follows "Title - Summary", produce a compact merged label and a short paragraph describing interaction style and emotional-progress signals.

INPUTS:
INTERACTION_META_CHAPTERS_TOC: {interaction_meta_chapters_toc}

YOUR TASK:
1. Merge recurring interaction themes (gratitude, reflection, encouragement, process feedback) into one coherent label.
2. Keep the summary concise and behavioral; avoid inventing durable personal facts.

STYLE RULES:
- title: max 8 words, no quotes.
- summary: one short paragraph (may be 1 or 2 sentences if needed), no bullets.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "title": <string>,
  "summary": <string>
}}
"""

GENERATE_CHAPTER_TITLE_AND_SUMMARY_USER_DURABLE = """
You are a durable-memory indexer. Your job is to generate one merged title and summary from merged user_durable memory points.

MAIN OBJECTIVE:
From MERGED_CONFIRMED_MEMORY_POINTS only, produce a merged durable chapter title and summary that preserves key personal facts without over-generalizing.

INPUTS:
MERGED_CONFIRMED_MEMORY_POINTS: {merged_confirmed_memory_points}

YOUR TASK:
1. Synthesize overlapping personal facts into one coherent durable topic.
2. Preserve specific, user-centric facts and commitments from the provided memory points.
3. Keep output concise but faithful; do not invent facts.

STYLE RULES:
- title: max 8 words, no quotes.
- summary: one paragraph (1-2 sentences), no bullets.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "title": <string>,
  "summary": <string>
}}
"""

MERGE_FRAGMENTS_TO_CHAPTERS = """
You are a conservative memory-structure reviewer.

MAIN OBJECTIVE:
Given fragments from ONE session, identify which fragment indices can be safely merged into larger coherent chapters.

INPUTS:
FRAGMENTS:
{fragments}

TASK:
1. Output only merge groups whose members describe the same durable personal topic and can be consolidated without losing meaning.
2. Each group must contain at least 2 fragment indices.
3. Prefer conservative behavior: if uncertain, do not merge.
4. Do NOT merge fragments that are semantically unrelated, conflict with each other, or are too far apart in topic continuity.
5. Keep groups disjoint (no index can appear in more than one group).
6. Indices must come from FRAGMENTS exactly; do not invent indices.

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown fences, no commentary):
{{
  "merge_groups": [[<int>, <int>], [<int>, <int>, <int>]]
}}
"""
