# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Launch an SGLang LLM server from a YAML config profile.

Usage::

    source sandbox.sh
    python scripts/serve_llm.py
    python scripts/serve_llm.py --config sglang/qwen3.6-35b-a3b-fp8:h100
    python scripts/serve_llm.py --config sglang/qwen3.6-35b-a3b-fp8:h100 port=8080

``--config`` selects a YAML profile under ``scripts/configs/`` (default:
``sglang/qwen3.6-35b-a3b-fp8:h100``). Extra ``key=value`` args override
fields from the config.

Outputs::

    OpenAI-compatible API at http://<host>:<port>/v1
    With background=true, logs at ~/cache/sira/sglang-<port>.log
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from sira.schema.sglang import SGLangServerConfig

TMUX_SESSION_PREFIX = "sira-llm"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _tmux_session(port: int) -> str:
    return f"{TMUX_SESSION_PREFIX}-{port}"


def _load_config(argv: list[str]) -> SGLangServerConfig:
    """Load config from YAML profile + CLI overrides."""
    import yaml

    config_name = "sglang/qwen3.6-35b-a3b-fp8:h100"
    overrides = []

    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_name = argv[i + 1]
            i += 2
        else:
            overrides.append(argv[i])
            i += 1

    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    cfg = SGLangServerConfig()

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        for key, val in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

    for arg in overrides:
        if "=" not in arg:
            print(f"ERROR: expected key=value, got: {arg}", file=sys.stderr)
            sys.exit(1)
        key, val = arg.split("=", 1)
        if not hasattr(cfg, key):
            print(f"ERROR: unknown config key: {key}", file=sys.stderr)
            sys.exit(1)
        field_type = type(getattr(cfg, key))
        if field_type is bool:
            setattr(cfg, key, val.lower() in ("true", "1", "yes"))
        else:
            setattr(
                cfg,
                key,
                field_type(
                    ast.literal_eval(val) if field_type in (int, float) else val
                ),
            )

    return cfg


def main():
    cfg = _load_config(sys.argv[1:])

    health_url = f"http://127.0.0.1:{cfg.port}/v1/models"
    try:
        with urllib.request.urlopen(health_url, timeout=2) as resp:
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            print(f"Server already running on port {cfg.port}")
            print(f"  Models: {models}")
            print(f"  API:    {health_url}")
            print(f"\nTo restart: tmux kill-session -t {_tmux_session(cfg.port)}")
            return
    except (urllib.error.URLError, OSError):
        pass

    import torch

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("ERROR: no CUDA devices visible.", file=sys.stderr)
        sys.exit(1)

    if cfg.dp == 0:
        cfg.dp = max(1, num_gpus // cfg.tp)

    print(f"Model:    {cfg.model}")
    print(f"Port:     {cfg.port}")
    print(f"TP:       {cfg.tp}")
    print(f"DP:       {cfg.dp}")
    print(f"GPUs:     {num_gpus} ({cfg.tp * cfg.dp} used)")
    print(f"Mem:      {cfg.mem}")
    print(f"Context:  {cfg.context_length}")
    print(f"KV dtype: {cfg.kv_cache_dtype}")

    cmd = [sys.executable, "-m", "sglang.launch_server"] + cfg.to_cli_args()

    env = os.environ.copy()

    if cfg.background:
        if not shutil.which("tmux"):
            print("ERROR: tmux not found", file=sys.stderr)
            sys.exit(1)
        session = _tmux_session(cfg.port)
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

        log_dir = Path.home() / "cache" / "sira"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"sglang-{cfg.port}.log")
        open(log_path, "w").close()

        repo_root = Path(__file__).resolve().parent.parent
        sandbox = f"source {repo_root / 'sandbox.sh'} 2>/dev/null"
        shell_cmd = f"{sandbox} && {' '.join(cmd)} 2>&1 | tee -a {log_path}"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, shell_cmd],
            env=env,
            check=True,
        )
        print(f"\nStarting in tmux session '{session}'...")
        print(f"  Logs: tail -f {log_path}")
        print(f"  Live: tmux attach -t {session}")
        print(f"  Stop: tmux kill-session -t {session}")

        for i in range(600):
            time.sleep(1)
            ret = subprocess.run(
                ["tmux", "has-session", "-t", session], capture_output=True
            )
            if ret.returncode != 0:
                print("\nERROR: server exited. Check logs.", file=sys.stderr)
                sys.exit(1)
            try:
                with urllib.request.urlopen(health_url, timeout=2):
                    pass
                print(f"\nServer ready! API: {health_url}")
                return
            except (urllib.error.URLError, OSError):
                if i % 30 == 29:
                    print(f"  Waiting... ({i + 1}s)")
        print("\nWARNING: server did not become ready within 10 minutes")
        print(f"  Check: tmux attach -t {session}")
    else:
        print(f"\nLaunching: {' '.join(cmd)}")
        subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
