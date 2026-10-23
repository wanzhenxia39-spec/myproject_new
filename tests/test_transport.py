import json
from zshar.llm import transport


def test_zhizengzeng_endpoint_prefers_zzz_key(monkeypatch):
    for name in transport.api_key_env_names("openai", ""):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "stale-openai-key")
    monkeypatch.setenv("KAMI_API_KEY", "stale-kami-key")
    monkeypatch.setenv("ZZZ_API_KEY", "zhizengzeng-key")

    assert transport.resolve_api_key(
        "openai", "https://api.zhizengzeng.com/v1"
    ) == "zhizengzeng-key"


class _Response:
    status_code = 200
    text = ""

    def __init__(self, *, payload=None, lines=None):
        self.payload = payload
        self.lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def json(self):
        return self.payload

    def iter_lines(self, decode_unicode=False):
        assert decode_unicode is True
        return iter(self.lines)


def _call(*, stream):
    return transport.call_llm(
        messages=[{"role": "user", "content": "test"}],
        api_format="openai",
        api_key="placeholder",
        base_url="https://www.kamiapi.top/v1",
        model="gpt-5.5",
        temperature=0,
        timeout=300,
        ssl_no_verify=False,
        stream=stream,
    )


def test_openai_transport_uses_requests_for_non_streaming(monkeypatch):
    response = _Response(payload={"choices": [{"message": {"content": "ok"}}]})
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr(transport.requests, "post", fake_post)
    assert _call(stream=False) == "ok"
    assert calls[0][1]["stream"] is False
    assert calls[0][1]["json"]["stream"] is False


def test_openai_transport_collects_streamed_content(monkeypatch):
    response = _Response(lines=[
        'data: ' + json.dumps({"choices": [{"delta": {"content": '{"ok":'}}]}),
        'data: ' + json.dumps({"choices": [{"delta": {"content": "true}"}}]}),
        "data: [DONE]",
    ])
    monkeypatch.setattr(transport.requests, "post", lambda *args, **kwargs: response)
    assert json.loads(_call(stream=True)) == {"ok": True}


def test_openai_transport_keeps_complete_content_when_stream_closes_early(monkeypatch):
    class PrematureResponse(_Response):
        def iter_lines(self, decode_unicode=False):
            yield 'data: ' + json.dumps({
                "choices": [{"delta": {"content": '{"ok": true}'}}]
            })
            raise transport.requests.exceptions.ChunkedEncodingError(
                "Response ended prematurely"
            )

    monkeypatch.setattr(
        transport.requests,
        "post",
        lambda *args, **kwargs: PrematureResponse(),
    )
    assert json.loads(_call(stream=True)) == {"ok": True}
