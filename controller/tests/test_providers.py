"""Provider routing: prefixes, credentials, and accepted audio containers."""
import pytest

import config
import providers
from providers import Provider


class TestRouting:
    @pytest.mark.parametrize("model, expected", [
        ("whisper-large", Provider.LOCAL),
        ("aya-expanse-8b", Provider.LOCAL),
        ("openai:gpt-4o-mini", Provider.OPENAI),
        ("gemini:gemini-2.5-pro", Provider.GEMINI),
        (None, Provider.LOCAL),
        ("", Provider.LOCAL),
    ])
    def test_prefix_selects_provider(self, model, expected):
        assert providers.provider_of(model) is expected

    @pytest.mark.parametrize("model, expected", [
        ("openai:gpt-4o-mini", "gpt-4o-mini"),
        ("gemini:gemini-2.5-pro", "gemini-2.5-pro"),
        ("whisper-large", "whisper-large"),
        (None, ""),
    ])
    def test_prefix_is_stripped(self, model, expected):
        assert providers.bare_model(model) == expected

    def test_only_cloud_models_are_cloud(self):
        assert providers.is_cloud("openai:gpt-4o-mini")
        assert providers.is_cloud("gemini:gemini-2.5-pro")
        assert not providers.is_cloud("whisper-large")


class TestCredentials:
    def test_request_values_win_over_env(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_API_KEY", "from-env")
        key, url = providers.credentials(Provider.OPENAI, "from-request", "https://proxy/v1")
        assert (key, url) == ("from-request", "https://proxy/v1")

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_API_KEY", "from-env")
        monkeypatch.setattr(config, "OPENAI_BASE_URL", "https://api.openai.com/v1")
        assert providers.credentials(Provider.OPENAI) == ("from-env", "https://api.openai.com/v1")

    def test_gemini_uses_its_own_key(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_API_KEY", "openai-key")
        monkeypatch.setattr(config, "GEMINI_API_KEY", "gemini-key")
        key, _ = providers.credentials(Provider.GEMINI)
        assert key == "gemini-key"

    def test_trailing_slash_is_trimmed(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_API_KEY", "k")
        _, url = providers.credentials(Provider.OPENAI, base_url="https://proxy/v1/")
        assert url == "https://proxy/v1"

    def test_missing_key_names_the_env_var(self, monkeypatch):
        monkeypatch.setattr(config, "GEMINI_API_KEY", "")
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            providers.credentials(Provider.GEMINI)


class TestAudioFormat:
    @pytest.mark.parametrize("filename, expected", [
        ("report.wav", "wav"),
        ("report.MP3", "mp3"),
        ("a.b.report.flac", "flac"),
    ])
    def test_extension_is_extracted(self, filename, expected):
        assert providers.audio_format(filename, "gemini:x") == expected

    def test_openai_rejects_what_gemini_accepts(self):
        assert providers.audio_format("report.flac", "gemini:x") == "flac"
        with pytest.raises(ValueError, match="flac"):
            providers.audio_format("report.flac", "openai:x")

    @pytest.mark.parametrize("filename", ["report", "", "report.txt"])
    def test_unusable_names_are_rejected(self, filename):
        with pytest.raises(ValueError):
            providers.audio_format(filename, "gemini:x")
