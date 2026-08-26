import re
import os
import json
import logging
try:
    from dotenv import load_dotenv
except ImportError:  # 精简环境可省略 python-dotenv
    def load_dotenv(*_a, **_k):
        return False

from typing import Any, Dict, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log

from membook.openai_provider_extra import merge_dashscope_qwen3_nonstream_extra_body
from membook.usage.llm_usage import LLMUsageStats

load_dotenv()

logger = logging.getLogger(__name__)

# Define retry strategy parameters
RETRY_TIMES = int(os.getenv('RETRY_TIMES', '10'))
# 0 = 立即重试；也可用 WAIT_TIME_LOWER 兼容旧 .env
WAIT_TIME = float(os.getenv('WAIT_TIME', os.getenv('WAIT_TIME_LOWER', '0')))

# 兼容两套变量名：优先 OPENAI_BASE_URL；未设置时回退 MEMBOOK_LLM_BASE_URL。
# 都未设置时默认 api.openai.com，内网/代理环境需显式设置其一。
OPENAI_BASE_URL = (
    os.getenv('OPENAI_BASE_URL')
    or os.getenv('MEMBOOK_LLM_BASE_URL')
    or 'https://api.openai.com/v1'
)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')


common_params = {}

if os.getenv('OPENAI_MAX_TOKENS'):
    common_params["max_tokens"] = int(os.getenv('OPENAI_MAX_TOKENS'))

if os.getenv('OPENAI_TEMPERATURE'):
    common_params["temperature"] = float(os.getenv('OPENAI_TEMPERATURE'))

if os.getenv('OPENAI_TIMEOUT'):
    common_params["timeout"] = int(os.getenv('OPENAI_TIMEOUT'))


client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)


def _balanced_brace_substring(s: str, start: int) -> Optional[str]:
    """从 s[start]=='{' 起截取与之配平的 {...}（尊重字符串与转义）。"""
    n = len(s)
    if start < 0 or start >= n or s[start] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, n):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _repair_invalid_json_escapes(s: str) -> str:
    out: list[str] = []
    in_string = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == '"':
            bs = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                bs += 1
                j -= 1
            if bs % 2 == 0:
                in_string = not in_string
            out.append(ch)
            i += 1
            continue

        if in_string and ch == "\\":
            if i + 1 >= n:
                out.append("\\\\")
                i += 1
                continue
            nxt = s[i + 1]
            valid_basic = {'"', "\\", "/", "b", "f", "n", "r", "t"}
            if nxt in valid_basic:
                out.append("\\")
                i += 1
                continue
            if nxt == "u":
                hex_part = s[i + 2 : i + 6]
                if len(hex_part) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex_part):
                    out.append("\\")
                    i += 1
                    continue
            out.append("\\\\")
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _repair_json_string_control_chars(s: str) -> str:
    """将 JSON 字符串值内的未转义 ASCII 控制字符改为合法转义（常见于 reason 里粘贴多行对话）。"""
    out: list[str] = []
    i = 0
    n = len(s)
    in_string = False
    while i < n:
        ch = s[i]
        if ch == '"':
            bs = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                bs += 1
                j -= 1
            if bs % 2 == 0:
                in_string = not in_string
            out.append(ch)
            i += 1
            continue

        if in_string and ch == "\\" and i + 1 < n:
            out.append(ch)
            i += 1
            nxt = s[i]
            out.append(nxt)
            if nxt == "u" and i + 4 < n:
                out.extend(s[i + 1 : i + 5])
                i += 5
            else:
                i += 1
            continue

        if in_string and ord(ch) < 32:
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append("\\u%04x" % ord(ch))
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_smart_quotes(s: str) -> str:
    """将常见中文/弯引号归一化为 ASCII 双引号，减少 JSON 定界符错误。"""
    return s.replace("“", '"').replace("”", '"')


def _loads_json_with_repair(raw_json: str) -> Dict[str, Any]:
    s = _normalize_smart_quotes(raw_json)
    for _ in range(64):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            msg = str(e)
            if "Invalid \\escape" in msg:
                idx = max(0, e.pos - 1)
                if idx < len(s) and s[idx] == "\\":
                    s = s[:idx] + "\\\\" + s[idx + 1 :]
                else:
                    s = _repair_invalid_json_escapes(s)
            elif "Invalid control character" in msg:
                s = _repair_json_string_control_chars(s)
            elif "property name enclosed in double quotes" in msg or "Unterminated string" in msg:
                s = _normalize_smart_quotes(s)
            else:
                raise
    return json.loads(s)


def _extract_json_object_from_llm_text(content: str) -> str:
    s = (content or "").strip()
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.DOTALL | re.IGNORECASE):
        inner = (m.group(1) or "").strip()
        j = inner.find("{")
        if j < 0:
            continue
        sub = _balanced_brace_substring(inner, j)
        if sub:
            return sub
        # 兜底：即使字符串引号破损导致无法配平，也先切到首尾大括号交给 repair 处理
        k = inner.rfind("}")
        if k > j:
            return inner[j : k + 1]
    j = s.find("{")
    if j >= 0:
        sub = _balanced_brace_substring(s, j)
        if sub:
            return sub
        k = s.rfind("}")
        if k > j:
            return s[j : k + 1]
    raise ValueError(f"No JSON object found in model output: {s[:500]}")


@retry(
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def llm_request(prompt, tracker: Optional[LLMUsageStats] = None, category: str = "qa"):
    """
    Sends a request to the specified language model with a given prompt.

    Args:
        prompt (str): The input text or message to send to the model.
        tracker: Optional LLMUsageStats to record api_calls and tokens.
        category: Category label when tracker is set.

    Returns:
        str: The response generated by the model.
    """
    call_params = dict(common_params)
    merge_dashscope_qwen3_nonstream_extra_body(MODEL, OPENAI_BASE_URL, call_params)

    response_obj = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                'role': 'user', 
                'content': prompt
            }
        ],
        **call_params
    )

    content = response_obj.choices[0].message.content or ""
    if tracker is not None:
        tracker.record(
            response_obj.usage,
            category,
            prompt_text=prompt,
            completion_text=content,
        )
    return content

@retry(
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def llm_request_for_json(prompt):

    params = dict(common_params)
    # 默认开启 JSON object mode，减少评测阶段 JSON 解析失败；如需关闭可显式设为 false/0/no。
    json_object_mode = os.getenv("OPENAI_JSON_OBJECT_MODE", "true").strip().lower()
    if json_object_mode not in ("0", "false", "no", "off"):
        params = {**params, "response_format": {"type": "json_object"}}

    merge_dashscope_qwen3_nonstream_extra_body(MODEL, OPENAI_BASE_URL, params)

    response_obj = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        **params
    )

    content = response_obj.choices[0].message.content or ""
    json_str = _extract_json_object_from_llm_text(content)
    return _loads_json_with_repair(json_str)


if __name__ == '__main__':
    r = llm_request_for_json('hello')
    print(r)