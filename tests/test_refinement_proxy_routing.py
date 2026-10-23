import importlib.util
from pathlib import Path

import requests


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/refinement_xxapi_common.py"
    spec = importlib.util.spec_from_file_location("refinement_proxy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_route_bypasses_all_proxy_aliases_only_for_api_host(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:19100")
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    module = launcher()
    module.configure_proxy_routing("https://xxapi.info/v1", "direct")
    settings = requests.Session().merge_environment_settings("https://xxapi.info/v1/chat/completions", {}, False, None, None)
    assert settings["proxies"] == {}
    assert settings["verify"] is True
    assert not requests.utils.should_bypass_proxies("https://example.com", no_proxy=None)
    for url in ("http://localhost", "http://127.0.0.1"):
        assert requests.utils.should_bypass_proxies(url, no_proxy=None)
    assert module.build_parser("wisdm").parse_args([]).proxy_mode == "direct"


def test_environment_route_preserves_existing_proxy_settings(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:19100")
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    before = requests.utils.get_environ_proxies("https://xxapi.info/v1")
    launcher().configure_proxy_routing("https://xxapi.info/v1", "environment")
    assert requests.utils.get_environ_proxies("https://xxapi.info/v1") == before
