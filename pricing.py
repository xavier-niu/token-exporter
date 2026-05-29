import json
import logging
import os
import re
import urllib.request
from html import unescape
from pathlib import Path
from time import time

logger = logging.getLogger("token-stats")

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
ZAI_PRICING_URL = "https://docs.z.ai/guides/overview/pricing"
DEFAULT_TTL_SECS = 24 * 3600
_RATE_FIELDS = ("input", "output", "cached", "cache_creation")
_DECIMAL_P_RE = re.compile(r"(?<=\d)p(?=\d)")
_ZAI_TEXT_MODEL_RE = re.compile(r"\b(GLM-[A-Za-z0-9][A-Za-z0-9.-]*)(?=\s*(?:\$|Free|-))")
_PRICE_TOKEN_RE = re.compile(r"Limited-time Free|Free|\$[0-9]+(?:\.[0-9]+)?|-+")
_OFFICIAL_PROVIDER_PREFIXES = (
    ("glm", ("zai/",)),
    ("autoglm", ("zai/",)),
    ("minimax", ("minimax/",)),
    ("kimi", ("moonshot/",)),
)
_OFFICIAL_MODEL_RATES = {
    # Z.ai official pricing, USD per 1M tokens.
    # https://docs.z.ai/guides/overview/pricing
    "glm-5.1": {"input": 1.4, "cached": 0.26, "output": 4.4, "cache_creation": 0},
    "glm-5": {"input": 1.0, "cached": 0.2, "output": 3.2, "cache_creation": 0},
    "glm-5-turbo": {"input": 1.2, "cached": 0.24, "output": 4.0, "cache_creation": 0},
    "glm-4.7": {"input": 0.6, "cached": 0.11, "output": 2.2, "cache_creation": 0},
    "glm-4.7-flashx": {"input": 0.07, "cached": 0.01, "output": 0.4, "cache_creation": 0},
    "glm-4.7-flash": {"input": 0, "cached": 0, "output": 0, "cache_creation": 0},
    "glm-4.6": {"input": 0.6, "cached": 0.11, "output": 2.2, "cache_creation": 0},
    "glm-4.5": {"input": 0.6, "cached": 0.11, "output": 2.2, "cache_creation": 0},
    "glm-4.5-x": {"input": 2.2, "cached": 0.45, "output": 8.9, "cache_creation": 0},
    "glm-4.5-air": {"input": 0.2, "cached": 0.03, "output": 1.1, "cache_creation": 0},
    "glm-4.5-airx": {"input": 1.1, "cached": 0.22, "output": 4.5, "cache_creation": 0},
    "glm-4.5-flash": {"input": 0, "cached": 0, "output": 0, "cache_creation": 0},
    "glm-4-32b-0414-128k": {"input": 0.1, "cached": 0, "output": 0.1, "cache_creation": 0},
}


def _load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return {}


def _fetch_litellm(timeout: float = 30.0) -> dict | None:
    try:
        with urllib.request.urlopen(LITELLM_URL, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("Could not fetch LiteLLM pricing: %s", e)
        return None


def _price_value(token: str) -> float:
    token = token.strip()
    if not token or token in ("-", "--") or token.lower().endswith("free"):
        return 0.0
    match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", token)
    if not match:
        return 0.0
    return float(match.group(1))


def _html_text(content: str) -> str:
    content = re.sub(r"(?i)<br\s*/?>", "\n", content)
    content = re.sub(r"(?i)</(?:tr|p|h[1-6]|div|li|table|thead|tbody)>", "\n", content)
    content = re.sub(r"(?i)</(?:td|th)>", " ", content)
    content = re.sub(r"<[^>]+>", "", content)
    return unescape(content)


def _zai_text_model_segment(text: str) -> str:
    start = text.find("Prices per 1M tokens")
    if start == -1:
        return text
    ends = [
        pos
        for marker in ("Vision Models", "Built-in Tools", "Image Generation Models")
        if (pos := text.find(marker, start + 1)) != -1
    ]
    end = min(ends) if ends else len(text)
    return text[start:end]


def _parse_zai_pricing(content: str) -> dict:
    """Extract Z.ai text model prices, in USD per 1M tokens, from the docs page."""
    segment = _zai_text_model_segment(_html_text(content))
    matches = list(_ZAI_TEXT_MODEL_RE.finditer(segment))
    rates: dict[str, dict[str, float]] = {}

    for idx, match in enumerate(matches):
        model = match.group(1)
        row_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(segment)
        row = segment[match.end():row_end]
        tokens = _PRICE_TOKEN_RE.findall(row)
        if len(tokens) < 3:
            continue

        input_rate = _price_value(tokens[0])
        cached_rate = _price_value(tokens[1])
        output_rate = _price_value(tokens[3] if len(tokens) >= 4 else tokens[2])
        spec = {
            "input": input_rate,
            "cached": cached_rate,
            "output": output_rate,
            "cache_creation": 0.0,
        }
        model_key = model.lower()
        rates[model_key] = spec
        rates[f"zai/{model_key}"] = spec

    return rates


def _fetch_zai_pricing(timeout: float = 30.0) -> dict | None:
    try:
        with urllib.request.urlopen(ZAI_PRICING_URL, timeout=timeout) as resp:
            rates = _parse_zai_pricing(resp.read().decode("utf-8", "replace"))
            if not rates:
                logger.warning("Could not parse Z.ai pricing page")
                return None
            return rates
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as e:
        logger.warning("Could not fetch Z.ai pricing: %s", e)
        return None


def _refresh_cache(cache_path: str, ttl: int) -> dict:
    p = Path(cache_path)
    if p.exists() and (time() - p.stat().st_mtime) < ttl:
        return _load_json(cache_path)
    data = _fetch_litellm()
    if data is None:
        return _load_json(cache_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)
        logger.info("Refreshed LiteLLM pricing cache (%d models)", len(data))
    except OSError as e:
        logger.warning("Could not write pricing cache %s: %s", cache_path, e)
    return data


def _derived_cache_path(cache_path: str, label: str) -> str:
    p = Path(cache_path)
    suffix = p.suffix or ".json"
    return str(p.with_name(f"{p.stem}-{label}{suffix}"))


def _refresh_zai_cache(cache_path: str, ttl: int) -> dict:
    p = Path(cache_path)
    if p.exists() and (time() - p.stat().st_mtime) < ttl:
        return _load_json(cache_path)
    data = _fetch_zai_pricing()
    if data is None:
        return _load_json(cache_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)
        logger.info("Refreshed Z.ai pricing cache (%d models)", len(data))
    except OSError as e:
        logger.warning("Could not write pricing cache %s: %s", cache_path, e)
    return data


def _normalize_litellm(litellm_data: dict) -> dict:
    """Convert LiteLLM per-token rates into per-million-token rates."""
    out: dict[str, dict[str, float]] = {}
    for name, spec in litellm_data.items():
        if not isinstance(spec, dict):
            continue
        ipt = spec.get("input_cost_per_token")
        opt = spec.get("output_cost_per_token")
        if ipt is None and opt is None:
            continue
        ipt = ipt or 0
        opt = opt or 0
        cached = spec.get("cache_read_input_token_cost", ipt)
        cache_creation = spec.get("cache_creation_input_token_cost", ipt)
        out[name] = {
            "input": ipt * 1_000_000,
            "output": opt * 1_000_000,
            "cached": (cached or 0) * 1_000_000,
            "cache_creation": (cache_creation or 0) * 1_000_000,
        }
    return out


def load_pricing(override_path: str = "", cache_path: str = "", ttl: int = DEFAULT_TTL_SECS) -> dict:
    """Load pricing rates per model. Local overrides take final precedence."""
    rates: dict[str, dict[str, float]] = {}
    if cache_path:
        rates.update(_normalize_litellm(_refresh_cache(cache_path, ttl)))
        rates.update(_refresh_zai_cache(_derived_cache_path(cache_path, "zai"), ttl))
    if override_path:
        for name, spec in _load_json(override_path).get("models", {}).items():
            if isinstance(spec, dict):
                rates[name] = {f: float(spec.get(f, 0) or 0) for f in _RATE_FIELDS}
    return rates


def _normalized_model_key(model: str) -> str:
    model = _DECIMAL_P_RE.sub("", model.lower())
    return "".join(ch.lower() for ch in model if ch.isalnum())


_OFFICIAL_MODEL_RATES_NORMALIZED = {
    _normalized_model_key(model): rates for model, rates in _OFFICIAL_MODEL_RATES.items()
}


def _normalized_model_keys(model: str) -> set[str]:
    normalized = _normalized_model_key(model)
    keys = {normalized}
    if normalized.startswith("kimi") or "kimi" in normalized:
        keys.add(normalized.replace("instruct", "").replace("preview", ""))
    return {key for key in keys if key}


def _model_key_variants(model: str) -> list[str]:
    variants = [model]
    if "/" in model:
        variants.append(model.split("/", 1)[1])
        variants.append(model.rsplit("/", 1)[1])
    for variant in tuple(variants):
        if "." not in variant:
            continue
        prefix, suffix = variant.split(".", 1)
        if not any(ch.isdigit() for ch in prefix):
            variants.append(suffix)

    out = []
    for variant in variants:
        if variant and variant not in out:
            out.append(variant)
    return out


def _official_prefixes_for(model_variants: list[str]) -> tuple[str, ...]:
    normalized_variants = [_normalized_model_key(variant) for variant in model_variants]
    for family, prefixes in _OFFICIAL_PROVIDER_PREFIXES:
        if any(variant.startswith(family) for variant in normalized_variants):
            return prefixes
    return ()


def lookup(rates: dict, model: str) -> dict:
    """Find rates for model.

    Tries exact match, provider suffixes, final path segments, dotted provider
    tails, then case/punctuation insensitive variants such as GLM5 -> glm-5.
    """
    if not model:
        return {}

    model_variants = _model_key_variants(model)
    model_normalized = {
        normalized
        for variant in model_variants
        for normalized in _normalized_model_keys(variant)
    }
    official_prefixes = _official_prefixes_for(model_variants)

    for key, spec in rates.items():
        if key in model_variants:
            return spec

    matches = []
    for key, spec in rates.items():
        key_variants = _model_key_variants(key)
        if any(variant in model_variants for variant in key_variants):
            matches.append((key, spec))
            continue
        if any(
            normalized in model_normalized
            for variant in key_variants
            for normalized in _normalized_model_keys(variant)
        ):
            matches.append((key, spec))

    if official_prefixes:
        for key, spec in matches:
            if "/" not in key:
                return spec
        for key, spec in matches:
            if key.lower().startswith(official_prefixes):
                return spec
        for normalized in model_normalized:
            spec = _OFFICIAL_MODEL_RATES_NORMALIZED.get(normalized)
            if spec:
                return spec
        return {}

    if matches:
        return matches[0][1]
    return {}


def cost_for(rates: dict, model: str, input_tokens: int, cached: int, output: int, cache_creation: int = 0) -> float:
    spec = lookup(rates, model)
    if not spec:
        return 0.0
    return (
        input_tokens * spec.get("input", 0)
        + cached * spec.get("cached", 0)
        + output * spec.get("output", 0)
        + cache_creation * spec.get("cache_creation", 0)
    ) / 1_000_000
