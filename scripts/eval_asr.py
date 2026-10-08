"""Speech-recognition accuracy and latency on recorded clips (data/eval/asr/manifest.yaml).

    python scripts/eval_asr.py                  # official NCAIR1 models, CTranslate2 int8 (needs HF_TOKEN)
    python scripts/eval_asr.py --compare        # int8 vs float32 (same weights, same decoding) per language
    python scripts/eval_asr.py --proxy          # public openai/whisper-small for every language (pipeline check)

Manifest rows: {file: <path relative to the manifest>, lang: ha|yo|ig|en, reference: "<exact transcript>"}.
Record clips as WhatsApp voice notes (Opus) from real speakers; references typed by a native speaker.
Reports WER (tones folded and tone-sensitive), CER and end-to-end latency (decode + ASR). Record results in SPIKES.md.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import yaml

from kudiguard.languages import LanguageRegistry
from kudiguard.speech.audio import find_ffmpeg
from kudiguard.speech.base import ASRError, AudioError
from kudiguard.speech.metrics import character_error_rate, word_error_rate
from kudiguard.speech.natlas_asr import NatlasASR, ct2_loader
from kudiguard.speech.service import transcribe_upload

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "eval" / "asr" / "manifest.yaml"
PROXY = "openai/whisper-small"


async def evaluate(
    asr: NatlasASR, rows: list[dict[str, str]], ffmpeg: str
) -> dict[str, list[tuple[float, ...]]]:
    by_language: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    for row in rows:
        try:
            result = await transcribe_upload(
                asr, (MANIFEST.parent / row["file"]).read_bytes(), row["lang"], ffmpeg=ffmpeg
            )
        except (AudioError, ASRError, OSError) as exc:  # one bad clip or model failure must not abort the run
            print(f"[{row['lang']}] SKIPPED {row['file']}: {exc}")
            continue
        text = result.transcript.text
        scores = (
            word_error_rate(row["reference"], text),
            word_error_rate(row["reference"], text, tone_sensitive=True),
            character_error_rate(row["reference"], text),
            float(result.total_ms),
        )
        by_language[row["lang"]].append(scores)
        print(
            f"[{row['lang']}] WER {scores[0]:.0%} (tones {scores[1]:.0%}) CER {scores[2]:.0%} {result.total_ms} ms  {text!r}"
        )
    return by_language


def summarise(label: str, by_language: dict[str, list[tuple[float, ...]]]) -> None:
    print(f"\n{label}")
    for language, rows in sorted(by_language.items()):
        wer, tone_wer, cer, ms = (list(column) for column in zip(*rows, strict=True))
        print(
            f"  {language}: n={len(rows)}  WER {mean(wer):.1%}  WER(tones) {mean(tone_wer):.1%}  "
            f"CER {mean(cer):.1%}  latency p50 {median(ms):.0f} ms"
        )


async def main(proxy: bool, compare: bool, cache: Path) -> None:
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    rows = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or []
    models = {code: PROXY for code in registry.asr_models()} if proxy else registry.asr_models()
    tokens = {PROXY: None} if proxy else registry.asr_whisper_languages()
    ffmpeg = find_ffmpeg()
    for quantization in ("int8", "float32") if compare else ("int8",):
        loader = ct2_loader(cache_dir=cache, whisper_languages=tokens, quantization=quantization)
        languages = sorted({row["lang"] for row in rows})
        asr = NatlasASR(models, loader=loader, max_loaded=len(languages))
        await asr.preload(languages)  # latency below is per message, not first-use model loading
        summarise(f"CTranslate2 {quantization} (proxy={proxy})", await evaluate(asr, rows, ffmpeg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--compare", action="store_true", help="int8 vs float32 on the same weights")
    parser.add_argument(
        "--cache", type=Path, default=Path(os.environ.get("KUDIGUARD_MODEL_CACHE", r"***REMOVED***\ncair\.cache"))
    )
    args = parser.parse_args()
    asyncio.run(main(args.proxy, args.compare, args.cache))
