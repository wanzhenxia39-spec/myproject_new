"""Transport copied from /home/wanzhen/gpt_run.py; no embedded credentials."""
import os
import ssl
import json
import sys
import urllib.request
import urllib.error
from typing import List

import requests


def api_key_env_names(api_format: str, base_url: str = "") -> tuple[str, ...]:
    """Return key aliases in provider-aware priority order."""
    if api_format != "openai":
        return ("LLM_API_KEY", "GEMINI_API_KEY", "ZZZ_API_KEY")

    host = (base_url or "").lower()
    if "zhizengzeng.com" in host:
        # Do not let a stale Kami/OpenAI key shadow the key for this endpoint.
        return ("ZZZ_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
    if "kamiapi.top" in host:
        return (
            "KAMI_API_KEY", "NEW_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "ZZZ_API_KEY"
        )
    return (
        "LLM_API_KEY", "OPENAI_API_KEY", "ZZZ_API_KEY", "KAMI_API_KEY", "NEW_API_KEY"
    )


def resolve_api_key(api_format: str, base_url: str = "") -> str:
    """Resolve a key without putting credentials in command-line arguments."""
    return next(
        (os.getenv(name) for name in api_key_env_names(api_format, base_url) if os.getenv(name)),
        "",
    )

def _ssl_context(no_verify: bool):
    if no_verify:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None

def _openai_chat_completions_url(base_url: str) -> str:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return f"{root}/chat/completions"

def _gemini_generate_content_url(base_url: str, model: str, api_key: str) -> str:
    from urllib.parse import quote, urlencode

    root = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    if root.endswith(":generateContent"):
        url = root
    elif root.endswith("/v1") or root.endswith("/v1beta"):
        url = f"{root}/models/{quote(model, safe='')}:generateContent"
    else:
        url = f"{root}/v1beta/models/{quote(model, safe='')}:generateContent"
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({'key': api_key})}"


def _openai_content(response_json: dict) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compatible API returned no choices: {response_json}")
    return str((choices[0].get("message") or {}).get("content", "")).strip()


def _call_openai(
    *, messages: List[dict], api_key: str, base_url: str, model: str,
    temperature: float, timeout: float, ssl_no_verify: bool, stream: bool,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": bool(stream),
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": os.getenv("LLM_USER_AGENT", "curl/8.0.0"),
        "Authorization": f"Bearer {api_key}",
    }
    try:
        with requests.post(
            _openai_chat_completions_url(base_url),
            headers=headers,
            json=payload,
            stream=stream,
            timeout=(30.0, float(timeout)),
            verify=not ssl_no_verify,
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(
                    f"openai API HTTP {response.status_code}: {response.text[:4000]}"
                )
            if not stream:
                return _openai_content(response.json())

            content_parts = []
            try:
                for line in response.iter_lines(decode_unicode=True):
                    line = (line or "").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError(
                            f"OpenAI-compatible stream returned an error: {event['error']}"
                        )
                    for choice in event.get("choices") or []:
                        content = (choice.get("delta") or {}).get("content")
                        if isinstance(content, str):
                            content_parts.append(content)
                        message_content = (choice.get("message") or {}).get("content")
                        if isinstance(message_content, str):
                            content_parts.append(message_content)
            except requests.exceptions.ChunkedEncodingError:
                if not content_parts:
                    raise
                print(
                    "[WARN] OpenAI-compatible stream ended before [DONE]; "
                    "validating the content already received.",
                    file=sys.stderr,
                )
            streamed_text = "".join(content_parts).strip()
            if not streamed_text:
                raise RuntimeError("OpenAI-compatible API returned an empty stream")
            return streamed_text
    except requests.RequestException as exc:
        raise RuntimeError(f"openai API request failed: {exc}") from exc

def call_llm(
    messages: List[dict],
    api_format: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout: float,
    ssl_no_verify: bool,
    stream: bool = False,
) -> str:
    if not api_key:
        raise ValueError("Missing API key. Set LLM_API_KEY or a provider-specific key environment variable.")
    if not model:
        raise ValueError("Missing model. Fill --model or MODEL_NAME.")

    if api_format == "openai":
        return _call_openai(
            messages=messages,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
            ssl_no_verify=ssl_no_verify,
            stream=stream,
        )
    if api_format == "gemini":
        system_prompt = ""
        user_parts = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "system":
                system_prompt = content if not system_prompt else system_prompt + "\n" + content
            else:
                user_parts.append(content)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "\n\n".join(user_parts)}]}],
            "generationConfig": {"temperature": float(temperature)},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        url = _gemini_generate_content_url(base_url, model, api_key)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
        }
    else:
        raise ValueError(f"Unsupported --api-format: {api_format}")

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=_ssl_context(ssl_no_verify),
        ) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{api_format} API HTTP {exc.code}: {error_body}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{api_format} API request timed out after {timeout} seconds") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{api_format} API request failed: {exc}") from exc

    candidates = response_json.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {response_json}")
    parts = candidates[0].get("content", {}).get("parts", []) or []
    return "".join(str(part.get("text", "")) for part in parts).strip()
