import json
import time
from unittest.mock import patch

import pytest

import pricing


@pytest.fixture
def litellm_sample():
    return {
        "sample_spec": {"input_cost_per_token": 1, "output_cost_per_token": 1},
        "gpt-5": {
            "input_cost_per_token": 1.25e-6,
            "output_cost_per_token": 1.0e-5,
            "cache_read_input_token_cost": 1.25e-7,
            "cache_creation_input_token_cost": 1.25e-6,
        },
        "deepseek/deepseek-chat": {
            "input_cost_per_token": 2.7e-7,
            "output_cost_per_token": 1.1e-6,
        },
        "no-cost-model": {"litellm_provider": "x"},
    }


@pytest.fixture(autouse=True)
def no_zai_network():
    with patch.object(pricing, "_fetch_zai_pricing", return_value=None):
        yield


class TestNormalizeLitellm:
    def test_per_mtok_conversion(self, litellm_sample):
        out = pricing._normalize_litellm(litellm_sample)
        assert out["gpt-5"] == {
            "input": pytest.approx(1.25),
            "output": pytest.approx(10.0),
            "cached": pytest.approx(0.125),
            "cache_creation": pytest.approx(1.25),
        }

    def test_skips_entries_without_costs(self, litellm_sample):
        out = pricing._normalize_litellm(litellm_sample)
        assert "no-cost-model" not in out
        assert "sample_spec" in out  # sample_spec has costs and is included

    def test_default_cached_to_input_rate(self, litellm_sample):
        out = pricing._normalize_litellm(litellm_sample)
        # deepseek/deepseek-chat has no cache_read_input_token_cost
        assert out["deepseek/deepseek-chat"]["cached"] == pytest.approx(0.27)


class TestLookup:
    def test_exact_match(self):
        rates = {"gpt-5": {"input": 1.25}}
        assert pricing.lookup(rates, "gpt-5") == {"input": 1.25}

    def test_provider_prefix_fallback(self):
        rates = {"deepseek/deepseek-chat": {"input": 0.27}}
        assert pricing.lookup(rates, "deepseek-chat") == {"input": 0.27}

    def test_normalized_model_fallback(self):
        rates = {"glm-5": {"input": 0.6}}
        assert pricing.lookup(rates, "GLM5") == {"input": 0.6}

    def test_future_glm_version_fallback(self):
        rates = {
            "baseten/zai-org/GLM-5.2": {"input": 0.9},
            "zai/glm-5.2": {"input": 1.0},
        }
        assert pricing.lookup(rates, "GLM5.2") == {"input": 1.0}

    def test_glm_family_prefers_official_provider(self):
        rates = {
            "baseten/zai-org/GLM-5": {"input": 0.95},
            "zai/glm-5": {"input": 1.0},
        }
        assert pricing.lookup(rates, "GLM-5") == {"input": 1.0}

    def test_glm_family_uses_zai_official_fallback(self):
        rates = {"fireworks_ai/glm-5p1": {"input": 1.4}}
        assert pricing.lookup(rates, "GLM-5.1") == {
            "input": 1.4,
            "cached": 0.26,
            "output": 4.4,
            "cache_creation": 0,
        }

    def test_dotted_provider_tail_fallback(self):
        rates = {"bedrock/us-east-1/amazon.nova-lite-v1:0": {"input": 1.0}}
        assert pricing.lookup(rates, "nova-lite-v1:0") == {"input": 1.0}

    def test_normalized_provider_suffix_fallback(self):
        rates = {"zai/glm-4.5": {"input": 0.6}}
        assert pricing.lookup(rates, "GLM4.5") == {"input": 0.6}

    def test_kimi_final_path_segment_fallback(self):
        rates = {
            "groq/moonshotai/kimi-k2.5": {"input": 0.9},
            "moonshot/kimi-k2.5": {"input": 0.6},
        }
        assert pricing.lookup(rates, "Kimi K2.5") == {"input": 0.6}

    def test_kimi_instruct_uses_official_preview_pricing(self):
        rates = {
            "groq/moonshotai/kimi-k2-instruct-0905": {"input": 1.0},
            "moonshot/kimi-k2-0905-preview": {"input": 0.6},
        }
        assert pricing.lookup(rates, "Kimi K2 Instruct 0905") == {"input": 0.6}

    def test_kimi_family_does_not_fall_back_to_third_party(self):
        rates = {"groq/moonshotai/kimi-k2-instruct-0905": {"input": 1.0}}
        assert pricing.lookup(rates, "Kimi K2 Instruct 0905") == {}

    def test_minimax_provider_suffix_fallback(self):
        rates = {
            "baseten/MiniMaxAI/MiniMax-M2.5": {"input": 0.4},
            "minimax/MiniMax-M2.5": {"input": 0.3},
        }
        assert pricing.lookup(rates, "MiniMax-M2.5") == {"input": 0.3}
        assert pricing.lookup(rates, "minimax-m25") == {"input": 0.3}

    def test_providerless_official_family_rate_wins(self):
        rates = {
            "zai/glm-5": {"input": 9.0},
            "glm-5": {"input": 1.0},
        }
        assert pricing.lookup(rates, "GLM-5") == {"input": 1.0}

    def test_providerless_official_family_rate_with_dot_wins(self):
        rates = {
            "zai/glm-5.1": {"input": 9.0},
            "glm-5.1": {"input": 1.4},
        }
        assert pricing.lookup(rates, "GLM-5.1") == {"input": 1.4}

    def test_unknown_model_returns_empty(self):
        assert pricing.lookup({"gpt-5": {"input": 1}}, "mystery-model") == {}

    def test_empty_model_name(self):
        assert pricing.lookup({"gpt-5": {"input": 1}}, "") == {}


class TestCostFor:
    def test_known_model(self):
        rates = {"gpt-5": {"input": 1.25, "cached": 0.125, "output": 10.0, "cache_creation": 1.25}}
        # 1M input + 100K output -> 1.25 + 1.0 = 2.25
        cost = pricing.cost_for(rates, "gpt-5", 1_000_000, 0, 100_000)
        assert cost == pytest.approx(2.25)

    def test_unknown_model_is_zero(self):
        assert pricing.cost_for({}, "mystery", 1_000_000, 0, 100_000) == 0.0

    def test_includes_cached_and_cache_creation(self):
        rates = {"m": {"input": 1.0, "cached": 0.1, "output": 2.0, "cache_creation": 0.5}}
        # 1M input * 1 + 1M cached * 0.1 + 1M output * 2 + 1M cc * 0.5 = 3.6
        cost = pricing.cost_for(rates, "m", 1_000_000, 1_000_000, 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.6)


class TestLoadPricing:
    def test_parse_zai_pricing_text_models(self):
        html = """
        <h3>Text Models</h3>
        <p>Prices per 1M tokens.</p>
        Model Input Cached Input Cached Input Storage Output
        GLM-5.1$1.4$0.26 Limited-time Free$4.4
        GLM-4-32B-0414-128K$0.1--$0.1
        GLM-4.7-Flash Free Free Free Free
        <h3>Vision Models</h3>
        GLM-5V-Turbo$1.2$0.24 Limited-time Free$4
        """

        rates = pricing._parse_zai_pricing(html)

        assert rates["zai/glm-5.1"] == {
            "input": 1.4,
            "cached": 0.26,
            "output": 4.4,
            "cache_creation": 0.0,
        }
        assert rates["glm-4-32b-0414-128k"]["cached"] == 0
        assert rates["glm-4-32b-0414-128k"]["output"] == 0.1
        assert rates["glm-4.7-flash"]["input"] == 0
        assert "zai/glm-5v-turbo" not in rates

    def test_load_pricing_adds_dynamic_zai_rates(self, tmp_path, litellm_sample):
        cache = tmp_path / "litellm.json"
        cache.write_text(json.dumps(litellm_sample))
        zai_rates = {"zai/glm-5.2": {"input": 1.5, "cached": 0.3, "output": 4.8, "cache_creation": 0}}

        with patch.object(pricing, "_fetch_zai_pricing", return_value=zai_rates):
            rates = pricing.load_pricing("", str(cache), ttl=3600)

        assert pricing.lookup(rates, "GLM-5.2") == zai_rates["zai/glm-5.2"]

    def test_fresh_zai_cache_skips_fetch(self, tmp_path):
        cache = tmp_path / "litellm.json"
        cache.write_text("{}")
        zai_cache = tmp_path / "litellm-zai.json"
        cached_rates = {
            "zai/glm-5.2": {
                "input": 1.5,
                "cached": 0.3,
                "output": 4.8,
                "cache_creation": 0,
            }
        }
        zai_cache.write_text(json.dumps(cached_rates))

        with patch.object(pricing, "_fetch_zai_pricing", return_value={}) as m:
            rates = pricing.load_pricing("", str(cache), ttl=3600)

        m.assert_not_called()
        assert pricing.lookup(rates, "GLM-5.2") == cached_rates["zai/glm-5.2"]

    def test_stale_zai_fetch_failure_falls_back_to_cache(self, tmp_path):
        cache = tmp_path / "litellm.json"
        cache.write_text("{}")
        zai_cache = tmp_path / "litellm-zai.json"
        cached_rates = {
            "zai/glm-5.2": {
                "input": 1.5,
                "cached": 0.3,
                "output": 4.8,
                "cache_creation": 0,
            }
        }
        zai_cache.write_text(json.dumps(cached_rates))
        old = time.time() - 7200
        import os
        os.utime(zai_cache, (old, old))

        with patch.object(pricing, "_fetch_zai_pricing", return_value=None) as m:
            rates = pricing.load_pricing("", str(cache), ttl=3600)

        m.assert_called_once()
        assert pricing.lookup(rates, "GLM-5.2") == cached_rates["zai/glm-5.2"]

    def test_dynamic_zai_rates_override_litellm_catalog(self, tmp_path):
        cache = tmp_path / "litellm.json"
        cache.write_text(json.dumps({
            "zai/glm-5.2": {
                "input_cost_per_token": 9e-6,
                "output_cost_per_token": 9e-6,
            }
        }))
        zai_rates = {
            "zai/glm-5.2": {
                "input": 1.5,
                "cached": 0.3,
                "output": 4.8,
                "cache_creation": 0,
            }
        }

        with patch.object(pricing, "_fetch_zai_pricing", return_value=zai_rates):
            rates = pricing.load_pricing("", str(cache), ttl=3600)

        assert pricing.lookup(rates, "GLM-5.2") == zai_rates["zai/glm-5.2"]

    def test_override_wins_over_dynamic_zai_rates(self, tmp_path, litellm_sample):
        cache = tmp_path / "litellm.json"
        cache.write_text(json.dumps(litellm_sample))
        override = tmp_path / "pricing.json"
        override.write_text(json.dumps({
            "models": {
                "glm-5.2": {
                    "input": 7,
                    "cached": 0.7,
                    "output": 8,
                    "cache_creation": 0,
                }
            }
        }))
        zai_rates = {
            "zai/glm-5.2": {
                "input": 1.5,
                "cached": 0.3,
                "output": 4.8,
                "cache_creation": 0,
            }
        }

        with patch.object(pricing, "_fetch_zai_pricing", return_value=zai_rates):
            rates = pricing.load_pricing(str(override), str(cache), ttl=3600)

        assert pricing.lookup(rates, "GLM-5.2") == {
            "input": 7,
            "cached": 0.7,
            "output": 8,
            "cache_creation": 0,
        }

    def test_dynamic_zai_rate_wins_over_bundled_fallback(self):
        rates = {
            "zai/glm-5.1": {
                "input": 7,
                "cached": 0.7,
                "output": 8,
                "cache_creation": 0,
            },
            "fireworks_ai/glm-5p1": {"input": 99},
        }

        assert pricing.lookup(rates, "GLM-5.1") == {
            "input": 7,
            "cached": 0.7,
            "output": 8,
            "cache_creation": 0,
        }

    def test_override_wins_over_litellm(self, tmp_path, litellm_sample):
        cache = tmp_path / "litellm.json"
        cache.write_text(json.dumps(litellm_sample))
        override = tmp_path / "pricing.json"
        override.write_text(json.dumps({"models": {"gpt-5": {"input": 99, "output": 99, "cached": 99, "cache_creation": 99}}}))

        # Fresh cache so no fetch attempt
        with patch.object(pricing, "_fetch_litellm", return_value=None):
            rates = pricing.load_pricing(str(override), str(cache), ttl=3600)
        assert rates["gpt-5"]["input"] == 99
        assert rates["deepseek/deepseek-chat"]["input"] == pytest.approx(0.27)

    def test_missing_override_is_ok(self, tmp_path, litellm_sample):
        cache = tmp_path / "litellm.json"
        cache.write_text(json.dumps(litellm_sample))
        with patch.object(pricing, "_fetch_litellm", return_value=None):
            rates = pricing.load_pricing("", str(cache), ttl=3600)
        assert "gpt-5" in rates

    def test_stale_cache_triggers_fetch(self, tmp_path, litellm_sample):
        cache = tmp_path / "litellm.json"
        cache.write_text("{}")
        old = time.time() - 7200
        import os
        os.utime(cache, (old, old))
        with patch.object(pricing, "_fetch_litellm", return_value=litellm_sample) as m:
            rates = pricing.load_pricing("", str(cache), ttl=3600)
        m.assert_called_once()
        assert "gpt-5" in rates

    def test_fetch_failure_falls_back_to_cache(self, tmp_path, litellm_sample):
        cache = tmp_path / "litellm.json"
        cache.write_text(json.dumps(litellm_sample))
        old = time.time() - 7200
        import os
        os.utime(cache, (old, old))
        with patch.object(pricing, "_fetch_litellm", return_value=None):
            rates = pricing.load_pricing("", str(cache), ttl=3600)
        assert "gpt-5" in rates

    def test_missing_cache_and_no_fetch(self, tmp_path):
        cache = tmp_path / "litellm.json"
        with patch.object(pricing, "_fetch_litellm", return_value=None):
            rates = pricing.load_pricing("", str(cache), ttl=3600)
        assert rates == {}
