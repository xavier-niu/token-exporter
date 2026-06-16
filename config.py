import os


def load_config():
    default_claude_paths = os.path.expanduser("~/.claude,~/.codefuse/engine/cc")
    claude_dirs = os.environ.get("CLAUDE_CONFIG_DIR", default_claude_paths)
    claude_paths = [os.path.expanduser(p.strip()) for p in claude_dirs.split(",") if p.strip()]

    default_codex_paths = os.path.expanduser("~/.codex,~/.codefuse/engine/codex")
    codex_dirs = os.environ.get("CODEX_CONFIG_DIR", default_codex_paths)
    codex_paths = [os.path.expanduser(p.strip()) for p in codex_dirs.split(",") if p.strip()]

    watch_interval = int(os.environ.get("WATCH_INTERVAL", "60"))
    # Persisting more often than we scan is pointless (state only changes on a
    # scan) and a non-positive value would fsync every cycle; floor at the scan
    # interval.
    persist_interval = max(int(os.environ.get("PERSIST_INTERVAL", "600")), watch_interval)

    return {
        "listen_port": int(os.environ.get("LISTEN_PORT", "14531")),
        "watch_interval": watch_interval,
        "persist_interval": persist_interval,
        "claude_dirs": claude_paths,
        "codex_dirs": codex_paths,
        "days_back": int(os.environ.get("DAYS_BACK", "7")),
        "source": os.environ.get("SOURCE", ""),
        "state_file": os.path.expanduser(os.environ.get("STATE_FILE", "~/.token-exporter/state.json")),
        "timezone": os.environ.get("TZ", "UTC"),
        "pricing_override": os.path.expanduser(os.environ.get("PRICING_FILE", "/opt/token-exporter/pricing.json")),
        "pricing_cache": os.path.expanduser(os.environ.get("PRICING_CACHE", "~/.token-exporter/pricing-cache.json")),
        "pricing_ttl_secs": int(os.environ.get("PRICING_TTL_SECS", "3600")),
    }