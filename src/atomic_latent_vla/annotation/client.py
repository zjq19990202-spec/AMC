from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Literal, Protocol


CompletionCreate = Callable[..., Any]
ProviderName = Literal["qwen", "codex"]

DEFAULT_QWEN_MODEL = "qwen3-vl-plus"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_QWEN_TEMPERATURE = 0.25


class AnnotationClient(Protocol):
    provider: ProviderName
    model: str

    def complete_json(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
    ) -> tuple[dict, dict]: ...


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("model response did not contain a JSON object") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response must be one JSON object")
    return value


class QwenVLPlusClient:
    """Qwen3-VL client using the native DashScope multimodal endpoint.

    The production path intentionally does not use the OpenAI-compatible
    endpoint: native DashScope accepts the multimodal video-list parameters
    (including the larger Qwen3-VL frame budget) directly.
    """

    provider: ProviderName = "qwen"

    def __init__(
        self,
        *,
        model: str = DEFAULT_QWEN_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        temperature: float = DEFAULT_QWEN_TEMPERATURE,
        completion_create: CompletionCreate | None = None,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        if completion_create is not None:
            self._create = completion_create
            self._native = False
            return
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("set DASHSCOPE_API_KEY before calling Qwen3-VL-Plus")
        self._api_key = key
        self._endpoint = base_url or os.getenv(
            "DASHSCOPE_NATIVE_ENDPOINT",
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        )
        self._timeout_s = timeout_s
        self._native = True

    def complete_json(self, messages: list[dict], *, max_tokens: int = 4096) -> tuple[dict, dict]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self._native:
                    if self._has_local_video_file(messages):
                        return self._complete_native_sdk(messages, max_tokens=max_tokens)
                    return self._complete_native(messages, max_tokens=max_tokens)
                completion = self._create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    max_completion_tokens=max_tokens,
                )
                content = completion.choices[0].message.content
                if not isinstance(content, str):
                    raise ValueError("Qwen response content is not text")
                usage = getattr(completion, "usage", None)
                usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
                return parse_json_object(content), usage_dict
            except Exception as error:  # API SDKs expose several provider-specific errors.
                last_error = error
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
        detail = str(last_error) if last_error is not None else "unknown error"
        raise RuntimeError(
            f"Qwen3-VL-Plus failed after {self.max_retries} attempts: {detail}"
        ) from last_error

    @staticmethod
    def _has_local_video_file(messages: list[dict]) -> bool:
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "video":
                    continue
                video = part.get("video")
                if isinstance(video, str) and Path(video).is_file():
                    return True
        return False

    @staticmethod
    def _native_messages(messages: list[dict]) -> list[dict]:
        """Convert provider-neutral content parts to DashScope native parts."""
        converted: list[dict] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                native_content: object = content
            else:
                native_parts: list[dict] = []
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                        native_parts.append({"text": part.get("text", "")})
                    elif part_type == "video":
                        video = {"video": part.get("video", [])}
                        for key in ("fps", "max_pixels", "min_pixels", "max_frames"):
                            if key in part:
                                video[key] = part[key]
                        native_parts.append(video)
                    elif part_type == "image_url":
                        native_parts.append({"image": part["image_url"]["url"]})
                    else:
                        raise ValueError(f"unsupported native content type: {part_type!r}")
                native_content = native_parts
            converted.append({"role": message["role"], "content": native_content})
        return converted

    def _complete_native_sdk(
        self, messages: list[dict], *, max_tokens: int
    ) -> tuple[dict, dict]:
        try:
            from dashscope import MultiModalConversation
        except ImportError as error:
            raise RuntimeError("install dashscope before using local video files") from error
        response = MultiModalConversation.call(
            api_key=self._api_key,
            model=self.model,
            messages=self._native_messages(messages),
            response_format={"type": "json_object"},
            result_format="message",
            temperature=self.temperature,
            max_tokens=max_tokens,
            # DashScope's SDK otherwise uses its internal 300 s default.  A
            # complete sampled recording can legitimately take longer than that
            # to upload and decode, so honour the timeout chosen by the caller.
            request_timeout=self._timeout_s,
        )
        status_code = getattr(response, "status_code", None)
        if status_code is not None and int(status_code) >= 400:
            code = getattr(response, "code", "unknown")
            message = getattr(response, "message", response)
            raise RuntimeError(f"DashScope SDK HTTP {status_code}: {code}: {message}")
        try:
            content = response.output.choices[0].message.content[0]["text"]
        except (AttributeError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"DashScope SDK response has no text content: {response}") from error
        usage = getattr(response, "usage", {})
        return parse_json_object(content), dict(usage) if isinstance(usage, dict) else {}

    def _complete_native(
        self, messages: list[dict], *, max_tokens: int
    ) -> tuple[dict, dict]:
        payload = {
            "model": self.model,
            "input": {"messages": self._native_messages(messages)},
            "parameters": {
                "result_format": "message",
                "response_format": {"type": "json_object"},
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            },
        }
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope HTTP {error.code}: {detail[:1200]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"DashScope request failed: {error.reason}") from error

        if body.get("code"):
            raise RuntimeError(
                f"DashScope {body.get('code')}: {body.get('message', 'unknown error')}"
            )
        try:
            content = body["output"]["choices"][0]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"DashScope response has no text content: {body}") from error
        return parse_json_object(content), body.get("usage", {})


def _openai_content(content: Any) -> Any:
    """Convert Qwen's image-list video block into OpenAI multi-image content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("message content must be text or a content-part list")

    converted: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("message content parts must be objects")
        part_type = part.get("type")
        if part_type == "video":
            frames = part.get("video")
            if not isinstance(frames, list) or not frames:
                raise ValueError("video content requires a non-empty frame list")
            converted.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": frame, "detail": "auto"},
                }
                for frame in frames
            )
        elif part_type == "text":
            converted.append({"type": "text", "text": part.get("text", "")})
        elif part_type == "image_url":
            converted.append(part)
        else:
            raise ValueError(f"unsupported message content type: {part_type!r}")
    return converted


def convert_messages_for_openai(messages: list[dict]) -> list[dict]:
    """Map the provider-neutral/Qwen-compatible prompt into Chat Completions input."""
    return [
        {"role": message["role"], "content": _openai_content(message.get("content", ""))}
        for message in messages
    ]


class OpenAICodexClient:
    """Use a Codex model as the visual annotation backend.

    OpenAI models do not accept a video content part here, so the already sampled
    timestamped montage frames are sent as an ordered multi-image message.
    """

    provider: ProviderName = "codex"

    def __init__(
        self,
        *,
        model: str = DEFAULT_CODEX_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        completion_create: CompletionCreate | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        if completion_create is not None:
            self._create = completion_create
            return
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("set OPENAI_API_KEY before calling the Codex provider")
        endpoint = base_url or os.getenv("OPENAI_BASE_URL")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("install the project first: pip install -e .") from error
        client_kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout_s}
        if endpoint:
            client_kwargs["base_url"] = endpoint
        client = OpenAI(**client_kwargs)
        self._create = client.chat.completions.create

    def complete_json(self, messages: list[dict], *, max_tokens: int = 4096) -> tuple[dict, dict]:
        openai_messages = convert_messages_for_openai(messages)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = self._create(
                    model=self.model,
                    messages=openai_messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                )
                content = completion.choices[0].message.content
                if not isinstance(content, str):
                    raise ValueError("Codex response content is not text")
                usage = getattr(completion, "usage", None)
                usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
                return parse_json_object(content), usage_dict
            except Exception as error:  # API SDKs expose provider-specific errors.
                last_error = error
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"Codex provider failed after {self.max_retries} attempts"
        ) from last_error


def create_annotation_client(
    provider: ProviderName,
    *,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
) -> AnnotationClient:
    if provider == "qwen":
        kwargs: dict[str, Any] = {
            "model": model or DEFAULT_QWEN_MODEL,
            "base_url": base_url,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return QwenVLPlusClient(**kwargs)
    if provider == "codex":
        if temperature is not None:
            raise ValueError("--temperature is currently supported only by the Qwen provider")
        return OpenAICodexClient(model=model or DEFAULT_CODEX_MODEL, base_url=base_url)
    raise ValueError(f"unsupported provider: {provider!r}")
