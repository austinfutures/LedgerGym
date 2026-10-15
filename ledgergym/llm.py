"""Model backends.

One OpenAI-compatible chat backend covers Ollama (local CPU), Gemini (its OpenAI-compatible
endpoint), vLLM, and anything else that speaks /v1/chat/completions. On top of the raw API:

  * Disk cache:   every request is keyed by sha256(model, messages, tools, sampling params,
                  sample index). Re-runs and resumes replay from disk at zero quota cost.
  * Quota ledger: counts real (non-cached) requests per Pacific-time day in a small JSON file
                  and stops cleanly before the daily cap (Gemini quotas reset at midnight PT).
  * Rate limiter: spaces requests to stay under requests-per-minute.
  * Backoff:      retries transient errors and per-minute 429s; a daily-quota 429 raises
                  QuotaExhausted so the runner can save progress and exit.
  * Text fallback: small models sometimes write a tool call as text (Qwen's
                  <tool_call>{...}</tool_call>, a ```json block, or bare JSON) instead of a
                  structured call. Those are recovered, and parse_mode records which path was
                  used. The model really did intend the call, but the parse mode is kept so the
                  failure taxonomy can count format problems.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

OLLAMA_URL = "http://localhost:11434/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# rpm/rpd for Gemini are deliberately conservative defaults. Check your real numbers at
# https://aistudio.google.com/rate-limit and override with --rpm/--rpd or edit here.
PRESETS: dict[str, dict] = {
    "qwen1.5b": {"base_url": OLLAMA_URL, "model": "ledger-qwen1.5b", "api_key_env": None, "rpm": None, "rpd": None},
    "qwen3b":   {"base_url": OLLAMA_URL, "model": "ledger-qwen3b", "api_key_env": None, "rpm": None, "rpd": None},
    "qwen7b":   {"base_url": OLLAMA_URL, "model": "ledger-qwen7b", "api_key_env": None, "rpm": None, "rpd": None},
    "gemini-flash": {"base_url": GEMINI_URL, "model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY",
                     "rpm": 5, "rpd": 100},
}


class QuotaExhausted(Exception):
    """Daily budget reached (ledger or server). Progress is saved; resume after midnight PT."""


class BackendError(Exception):
    """The backend kept failing after retries (e.g. Ollama not running, bad API key)."""


class RateLimited(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class Transient(Exception):
    pass


_DAILY = re.compile(r"per.?day|perday|daily", re.I)


# ---------------------------------------------------------------- text tool-call fallback
_TC_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_text_tool_calls(content: str | None) -> list[dict]:
    if not content:
        return []
    cands = _TC_BLOCK.findall(content) or _FENCE.findall(content)
    if not cands:
        s = content.strip()
        if s.startswith("{") and s.endswith("}"):
            cands = [s]
    calls = []
    for c in cands:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", obj.get("parameters", {}))
            calls.append({"id": "", "name": obj["name"],
                          "arguments": args if isinstance(args, str) else json.dumps(args)})
    return calls


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[dict]           # [{"id", "name", "arguments": json-string}]
    parse_mode: str                  # "native" | "text_fallback" | "none"
    usage: dict = field(default_factory=dict)
    cached: bool = False
    latency_s: float = 0.0
    finish_reason: str | None = None


def turn_from_raw(raw: dict, cached: bool = False, latency_s: float = 0.0) -> AssistantTurn:
    """Parse an OpenAI-style chat.completion dict (live or cached) into an AssistantTurn."""
    choice = (raw.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        calls.append({"id": tc.get("id") or "", "name": fn.get("name") or "",
                      "arguments": args if isinstance(args, str) else json.dumps(args or {})})
    mode = "native" if calls else "none"
    if not calls:
        calls = extract_text_tool_calls(content)
        mode = "text_fallback" if calls else "none"
    return AssistantTurn(content, calls, mode, raw.get("usage") or {}, cached, latency_s,
                         choice.get("finish_reason"))


# ---------------------------------------------------------------- quota + rate limiting
def pacific_today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    except Exception as e:  # Windows without tzdata
        raise RuntimeError("Install tzdata for Pacific-time quota tracking: pip install tzdata") from e


class QuotaLedger:
    def __init__(self, path: str | Path, rpd: int | None, margin: float = 0.05):
        self.path, self.rpd, self.margin = Path(path), rpd, margin

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def used_today(self) -> int:
        return self._load().get(pacific_today(), 0)

    @property
    def cap(self) -> int | None:
        return None if not self.rpd else max(1, int(self.rpd * (1 - self.margin)))

    def check(self) -> None:
        if self.cap is not None and self.used_today() >= self.cap:
            raise QuotaExhausted(f"local ledger: {self.used_today()}/{self.rpd} requests used today "
                                 f"(stopping at {self.cap}). Resume after midnight Pacific.")

    def add(self, n: int = 1) -> None:
        data = self._load()
        day = pacific_today()
        data[day] = data.get(day, 0) + n
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, self.path)


class RateLimiter:
    def __init__(self, rpm: float | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.interval:
            delay = self._last + self.interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


# ---------------------------------------------------------------- backend
class ChatBackend:
    def __init__(self, name: str, base_url: str, model: str, api_key: str | None = None,
                 rpm: float | None = None, rpd: int | None = None, cache_dir: str | Path = "runs/cache",
                 temperature: float = 0.0, max_tokens: int = 1024, timeout: float = 900.0,
                 max_retries: int = 5):
        self.name, self.base_url, self.model = name, base_url, model
        self.api_key = api_key or "not-needed"
        self.temperature, self.max_tokens, self.timeout, self.max_retries = temperature, max_tokens, timeout, max_retries
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{name}__{model}")
        self.cache_dir = Path(cache_dir) / safe
        self.ledger = QuotaLedger(Path(cache_dir).parent / "quota" / f"{safe}.json", rpd)
        self.limiter = RateLimiter(rpm)
        self._client = None
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0}

    def _key(self, messages: list[dict], tools: list[dict], sample: int) -> str:
        blob = json.dumps({"model": self.model, "base_url": self.base_url, "messages": messages, "tools": tools,
                           "temperature": self.temperature, "max_tokens": self.max_tokens, "sample": sample},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _request(self, messages: list[dict], tools: list[dict]) -> dict:
        """One API call -> raw completion dict. Maps SDK errors to RateLimited / Transient."""
        import openai
        if self._client is None:
            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key,
                                         timeout=self.timeout, max_retries=0)
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto",
                temperature=self.temperature, max_tokens=self.max_tokens)
        except openai.RateLimitError as e:
            ra = None
            try:
                ra = float(e.response.headers.get("retry-after"))
            except Exception:
                pass
            raise RateLimited(str(e), ra) from e
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            raise Transient(str(e)) from e
        except openai.APIStatusError as e:
            raise BackendError(f"{self.name}: HTTP {e.status_code}: {e.message}") from e
        return resp.model_dump()

    def chat(self, messages: list[dict], tools: list[dict], sample: int = 0) -> AssistantTurn:
        key = self._key(messages, tools, sample)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            self.stats["cache_hits"] += 1
            rec = json.loads(path.read_text(encoding="utf-8"))
            return turn_from_raw(rec["raw"], cached=True, latency_s=rec.get("latency_s", 0.0))

        self.ledger.check()
        last_err = ""
        for attempt in range(self.max_retries):
            self.limiter.wait()
            t0 = time.monotonic()
            try:
                raw = self._request(messages, tools)
            except RateLimited as e:
                if _DAILY.search(str(e)):
                    raise QuotaExhausted(f"server says daily quota is exhausted: {str(e)[:300]}") from e
                last_err = str(e)
                wait = e.retry_after or min(120.0, 15.0 * 2 ** attempt)
            except Transient as e:
                last_err = str(e)
                wait = min(60.0, 5.0 * 2 ** attempt)
            else:
                latency = time.monotonic() - t0
                self.ledger.add(1)
                self.stats["requests"] += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"raw": raw, "latency_s": latency}, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, path)
                return turn_from_raw(raw, cached=False, latency_s=latency)
            self.stats["retries"] += 1
            print(f"    [{self.name}] retry {attempt + 1}/{self.max_retries} in {wait:.0f}s: {last_err[:120]}")
            time.sleep(wait)
        hint = " Is Ollama running (ollama serve) and the model created?" if "11434" in self.base_url else ""
        raise BackendError(f"{self.name}: failed after {self.max_retries} attempts: {last_err[:300]}.{hint}")


def make_backend(preset: str, *, model: str | None = None, base_url: str | None = None,
                 rpm: float | None = None, rpd: int | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, cache_dir: str | Path = "runs/cache") -> ChatBackend:
    cfg = dict(PRESETS[preset])
    key = None
    if cfg["api_key_env"]:
        key = os.environ.get(cfg["api_key_env"])
        if not key:
            raise BackendError(f"{preset} needs the {cfg['api_key_env']} environment variable.")
    return ChatBackend(preset, base_url or cfg["base_url"], model or cfg["model"], key,
                       rpm if rpm is not None else cfg["rpm"], rpd if rpd is not None else cfg["rpd"],
                       cache_dir, temperature, max_tokens)
