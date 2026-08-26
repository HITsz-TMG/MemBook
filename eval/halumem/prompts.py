PROMPT_MEMZERO = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    {context}

    Question: {question}

    Answer:
    """

# MemBook HaluMem QA 专用：``eval_membook.py`` 使用；槽位与 MEMZERO 相同（{context} / {question}）。
# 实际 ``context`` 来自 ``ResearchAgent.research`` 的 integrated 摘要，经 ``TEMPLATE_MEMORY`` 包装，
# 通常是已整合的叙述片段，而非「逐条独立记忆 + 每条独立 timestamp」的列表；故措辞按「单段证据」写。
PROMPT_MEMBOOK = """
    Your task is to answer questions about the user based on the integrated memory evidence block provided below.

    # CONTEXT:
    You have access to a memory evidence block containing user-specific facts that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided facts in the memory evidence block.
    2. Pay special attention to the time information to determine the Answer.
      - For time-related Question (e.g. anchored to a calendar date such as "on Sep 15, 2033", or "after December 25, 2035"), use temporal markers to decouple distinct events or states. Extract the specific evidence matching the queried timeframe rather than merging different time points. 
      - For the Question about latest or current: base on the most recent timestamped evidence only; do not treat all retrieved evidence as describing the current state.
      - For other questions, use the provided facts to answer the Question.
    3. Answer solely based on the provided facts. If the information is empty or missing, state that it is unknown or cannot be determined (e.g.,"Unknown based on provided information.").
    4. Strictly distinguish between established facts and future intentions. Phrases like "considering," "planning," "envisaging," or "aiming to" describe intents, not finalized events or current states. If the Question asks for a definitive fact (like a specific date or a status change) and the evidence only shows an intention, you must state that it is a plan or unknown. 
    5. The Answer should be a short sentence, less than 10 words if possible.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the Question.
    2. Examine the time information of these facts carefully.
    3. Look for explicit mentions of dates, times, locations, or events that answer the Question.
    4. If the Answer requires calculation (e.g., converting relative time references), show your work.
    5. Formulate an answer that maintains high precision and fidelity to the evidence. Prefer fideliy, completeness and faithfulness over conciseness.
    6. Double-check that your Answer directly addresses the Question asked.
    7. Ensure your final Answer is specific and avoids vague time references.

    {context}

    Question: {question}

    Answer:
    """

# Deprecated alias (prefer PROMPT_MEMBOOK)
PROMPT_MYMEM = PROMPT_MEMBOOK

# Full-Context / rereading 基线：context 为 FIFO 截断后的对话原文（非记忆系统检索结果）
PROMPT_FULL_CONTEXT = """
    You are an intelligent assistant answering questions about a user from conversation history.

    # CONTEXT:
    You have access to a chronologically ordered dialogue transcript (possibly truncated to the
    most recent turns that fit the context window). Each line is formatted as
    [timestamp]role: content.

    # INSTRUCTIONS:
    1. Carefully analyze the provided dialogue for evidence relevant to the question
    2. Pay special attention to timestamps to determine the answer
    3. If the dialogue contains contradictory information, prioritize the most recent evidence
    4. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the dialogue timestamp
    5. Always convert relative time references to specific dates, months, or years
    6. Answer solely based on the provided dialogue. If the information is missing, state that
       it is unknown or cannot be determined
    7. The answer should be a short sentence, less than 10 words if possible

    # APPROACH (Think step by step):
    1. Examine dialogue turns that contain information related to the question
    2. Examine the timestamps and content carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. Formulate a precise answer based solely on the evidence in the dialogue
    5. Double-check that your answer directly addresses the question asked

    {context}

    Question: {question}

    Answer:
    """

PROMPT_ZEP = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Context:

    {context}

    Question: {question}
    Answer:
    """

PROMPT_MEMOS = """
    You are a knowledgeable and helpful AI assistant.

   # CONTEXT:
   You have access to memories from two speakers in a conversation. These memories contain
   timestamped information that may be relevant to answering the question.

   # INSTRUCTIONS:
   1. Carefully analyze all provided memories. Synthesize information across different entries if needed to form a complete answer.
   2. Pay close attention to the timestamps to determine the answer. If memories contain contradictory information, the **most recent memory** is the source of truth.
   3. If the question asks about a specific event or fact, look for direct evidence in the memories.
   4. Your answer must be grounded in the memories. However, you may use general world knowledge to interpret or complete information found within a memory (e.g., identifying a landmark mentioned by description).
   5. If the question involves time references (like "last year", "two months ago", etc.), you **must** calculate the actual date based on the memory's timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
   6. Always convert relative time references to specific dates, months, or years in your final answer.
   7. Do not confuse character names mentioned in memories with the actual users who created them.
   8. The answer must be brief (under 5-6 words) and direct, with no extra description.

   # APPROACH (Think step by step):
   1. First, examine all memories that contain information related to the question.
   2. Synthesize findings from multiple memories if a single entry is insufficient.
   3. Examine timestamps and content carefully, looking for explicit dates, times, locations, or events.
   4. If the answer requires calculation (e.g., converting relative time references), perform the calculation.
   5. Formulate a precise, concise answer based on the evidence from the memories (and allowed world knowledge).
   6. Double-check that your answer directly addresses the question asked and adheres to all instructions.
   7. Ensure your final answer is specific and avoids vague time references.

   {context}

   Question: {question}

   Answer:
   """

PROMPT_MEMOBASE = """
   You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

   # CONTEXT:
   You have access to memories from two speakers in a conversation. These memories contain 
   timestamped information that may be relevant to answering the question.

   # INSTRUCTIONS:
   1. Carefully analyze all provided memories from both speakers
   2. Pay special attention to the timestamps to determine the answer
   3. If the question asks about a specific event or fact, look for direct evidence in the memories
   4. If the memories contain contradictory information, prioritize the most recent memory
   5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
   6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory timestamp. Ignore the reference while answering the question.
   7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
   8. The answer should be less than 5-6 words.

   # APPROACH (Think step by step):
   1. First, examine all memories that contain information related to the question
   2. Examine the timestamps and content of these memories carefully
   3. Look for explicit mentions of dates, times, locations, or events that answer the question
   4. If the answer requires calculation (e.g., converting relative time references), show your work
   5. Formulate a precise, concise answer based solely on the evidence in the memories
   6. Double-check that your answer directly addresses the question asked
   7. Ensure your final answer is specific and avoids vague time references

   {context}

   Question: {question}

   Answer:
   """