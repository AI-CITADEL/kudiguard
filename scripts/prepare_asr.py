"""Convert and verify the official NCAIR1 speech models before serving (never on a user's request).

    python scripts/prepare_asr.py                    # all enabled languages, CTranslate2 int8
    python scripts/prepare_asr.py --quantization float32

Needs HF_TOKEN (from the environment or kudiguard/.env) for an account that accepted the NCAIR1 model terms.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from kudiguard.envfile import load_env_file
from kudiguard.languages import LanguageRegistry
from kudiguard.speech.natlas_asr import NatlasASR, ct2_loader

ROOT = Path(__file__).resolve().parents[1]


async def main(cache: Path, quantization: str) -> int:
    load_env_file(ROOT / ".env")
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set (add HF_TOKEN=hf_... to kudiguard/.env or run huggingface-cli login)")
        return 1
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    loader = ct2_loader(
        cache_dir=cache, whisper_languages=registry.asr_whisper_languages(), quantization=quantization
    )
    asr = NatlasASR(registry.asr_models(), loader=loader, failure_cooldown_s=0)
    failures = 0
    for language, repo in registry.asr_models().items():
        started = time.perf_counter()
        errors = await asr.preload([language])
        status = f"FAILE***REMOVED*** {errors[language]}" if errors else "ok"
        failures += bool(errors)
        print(f"{language}: {repo} -> {status} ({time.perf_counter() - started:.0f} s)")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path, default=Path(os.environ.get("KUDIGUARD_MODEL_CACHE", str(ROOT / "models")))
    )
    parser.add_argument("--quantization", choices=["int8", "float32"], default="int8")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.cache, args.quantization)))
