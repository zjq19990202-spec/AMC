import json
from types import SimpleNamespace

from atomic_latent_vla.annotation.client import (
    OpenAICodexClient,
    QwenVLPlusClient,
    convert_messages_for_openai,
    parse_json_object,
)


def test_parse_fenced_json() -> None:
    assert parse_json_object('```json\n{"segments": []}\n```') == {"segments": []}


def test_client_requests_json_mode() -> None:
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(
            content=json.dumps({"global_description": "test demonstration", "segments": []})
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = QwenVLPlusClient(completion_create=fake_create, max_retries=1)
    payload, usage = client.complete_json([{"role": "user", "content": "x"}])
    assert payload["global_description"] == "test demonstration"
    assert usage == {}
    assert calls[0]["model"] == "qwen3-vl-plus"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["max_completion_tokens"] == 4096
    assert calls[0]["temperature"] == 0.25


def test_qwen_client_accepts_custom_temperature() -> None:
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content='{"global_description":"demo","segments":[]}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = QwenVLPlusClient(
        completion_create=fake_create,
        max_retries=1,
        temperature=0.6,
    )
    client.complete_json([{"role": "user", "content": "x"}])
    assert calls[0]["temperature"] == 0.6


def test_codex_converts_sampled_video_to_ordered_images() -> None:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Return JSON."}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": ["data:image/jpeg;base64,AAA", "frame-b"]},
                {"type": "text", "text": "Annotate."},
            ],
        },
    ]
    converted = convert_messages_for_openai(messages)
    user_content = converted[1]["content"]
    assert [part["type"] for part in user_content] == ["image_url", "image_url", "text"]
    assert user_content[0]["image_url"]["url"] == "data:image/jpeg;base64,AAA"
    assert user_content[1]["image_url"]["url"] == "frame-b"


def test_codex_client_uses_json_mode_without_temperature() -> None:
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content='{"global_description":"test demo","segments":[]}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = OpenAICodexClient(completion_create=fake_create, max_retries=1)
    payload, _ = client.complete_json([{"role": "user", "content": "Return JSON."}])
    assert payload["global_description"] == "test demo"
    assert calls[0]["model"] == "gpt-5.3-codex"
    assert calls[0]["max_completion_tokens"] == 4096
    assert "temperature" not in calls[0]
