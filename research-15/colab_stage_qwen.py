#!/usr/bin/env python3
"""Stage Qwen2.5-0.5B on a persistent Colab runtime without UI secrets."""
from __future__ import annotations

import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import snapshot_download

MODEL_ROOT = Path("/content/qwen2.5/transformers/0.5b/1")
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
config_path = MODEL_ROOT / "config.json"
if config_path.is_file():
    print("QWEN_MODEL_CACHE present", flush=True)
else:
    print("QWEN_MODEL_DOWNLOAD Qwen/Qwen2.5-0.5B", flush=True)
    snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B",
        local_dir=str(MODEL_ROOT),
        allow_patterns=[
            "config.json", "generation_config.json", "*.safetensors",
            "*.safetensors.index.json", "tokenizer*", "special_tokens_map.json",
            "vocab.json", "merges.txt",
        ],
    )
if not config_path.is_file():
    raise RuntimeError(f"missing {config_path}")
print("QWEN_MODEL_STAGE_PASS", str(MODEL_ROOT), flush=True)
