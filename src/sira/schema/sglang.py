# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class SGLangServerConfig:
    """Configuration for launching an SGLang inference server.

    Each field maps 1:1 to an ``sglang.launch_server`` CLI flag.
    Hardware-specific YAML profiles live under ``scripts/configs/sglang/``.
    """

    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    port: int = 30000
    host: str = "0.0.0.0"

    # Parallelism — tp × dp must equal the number of GPUs to use.
    tp: int = 2
    dp: int = 0  # 0 = auto (num_gpus // tp)

    # Memory & KV cache
    mem: float = 0.88
    context_length: int = 65536
    kv_cache_dtype: str = "fp8_e4m3"

    # Batching & scheduling
    max_running_requests: int = 512
    chunked_prefill_size: int = 8192
    schedule_policy: str = "lpm"

    # CUDA graph
    cuda_graph_max_bs: int = 1024
    disable_cuda_graph: bool = False

    # Radix cache — disable for unique-prompt workloads (3× in-flight cap).
    disable_radix_cache: bool = True

    # Multi-node DP requires dp attention.
    enable_dp_attention: bool = False

    # Startup
    skip_server_warmup: bool = True
    trust_remote_code: bool = True
    log_level: str = "info"

    # Background mode (tmux)
    background: bool = False

    def to_cli_args(self) -> list[str]:
        """Convert to ``sglang.launch_server`` CLI argument list."""
        args = [
            "--model-path", self.model,
            "--port", str(self.port),
            "--host", self.host,
            "--tp", str(self.tp),
            "--dp", str(self.dp),
            "--mem-fraction-static", str(self.mem),
            "--context-length", str(self.context_length),
            "--chunked-prefill-size", str(self.chunked_prefill_size),
            "--schedule-policy", self.schedule_policy,
            "--log-level", self.log_level,
        ]
        if self.kv_cache_dtype:
            args += ["--kv-cache-dtype", self.kv_cache_dtype]
        if self.max_running_requests > 0:
            args += ["--max-running-requests", str(self.max_running_requests)]
        if self.cuda_graph_max_bs > 0:
            args += ["--cuda-graph-max-bs", str(self.cuda_graph_max_bs)]
        if self.disable_cuda_graph:
            args.append("--disable-cuda-graph")
        if self.disable_radix_cache:
            args.append("--disable-radix-cache")
        if self.enable_dp_attention:
            args.append("--enable-dp-attention")
        if self.skip_server_warmup:
            args.append("--skip-server-warmup")
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        return args
