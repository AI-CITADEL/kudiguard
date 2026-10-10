"""Speech-recognition accuracy and latency on recorded clips (data/eval/asr/manifest.yaml).

    python scripts/eval_asr.py                  # official NCAIR1 models, CTranslate2 int8 (needs HF_TOKEN)
    python scripts/eval_asr.py --compare        # int8 vs float32 (same weights, same decoding) per language
    python scripts/eval_asr.py --proxy          # public openai/whisper-small for every language (pipeline check)

Manifest rows: {file: <path relative to the manifest>, lang: ha|yo|ig|en, reference: "<exact transcript>"}.
Record clips as WhatsApp voice notes (Opus) from real speakers; references typed by a native speaker.
Reports pooled WER (diacritics folded and kept), CER; rejected clips count as errors; per-clip JSONL logs; and end-to-end latency (decode + ASR). Record results in SPIKES.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import yaml

from kudiguard.envfile import load_env_file
from kudiguard.languages import LanguageRegistry
from kudiguard.speech.audio import find_ffmpeg
from kudiguard.speech.base import ASRError, AudioError
from kudiguard.speech.metrics import POOLED_METRICS
from kudiguard.speech.natlas_asr import NatlasASR, ct2_loader
from kudiguard.speech.service import transcribe_upload

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "eval" / "asr" / "manifest.yaml"
PROXY = "openai/whisper-small"


@dataclass
class Tally:
    clips: int = 0
    rejected: int = 0
    counts: dict[str, list[int]] = field(default_factory=lambda: {k: [0, 0] for k in METRICS})
    latencies: list[float] = field(default_factory=list)

    def add(self, reference: str, hypothesis: str) -> dict[str, float]:
        rates = {}
        for key, (fn, tone_sensitive) in METRICS.items():
            edits, total = fn(reference, hypothesis, tone_sensitive=tone_sensitive)
            self.counts[key][0] += edits
            self.counts[key][1] += total
            rates[key] = edits / max(total, 1)
        return rates

    def rate(self, key: str) -> float:
        edits, total = self.counts[key]
        return edits / max(total, 1)


METRICS = POOLED_METRICS


async def evaluate(
    asr: NatlasASR, rows: list[dict[str, str]], ffmpeg: str, log_path: Path
) -> dict[str, Tally]:
    """Per-clip results go to ``log_path`` (JSON lines). A rejected clip counts as all words deleted."""
    tallies: dict[str, Tally] = defaultdict(Tally)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for row in rows:
            tally = tallies[row["lang"]]
            tally.clips += 1
            record: dict[str, object] = {
                "file": row["file"],
                "lang": row["lang"],
                "reference": row["reference"],
            }
            try:
                result = await transcribe_upload(
                    asr, (MANIFEST.parent / row["file"]).read_bytes(), row["lang"], ffmpeg=ffmpeg
                )
            except (AudioError, ASRError, OSError) as exc:  # one bad clip must not abort the run
                tally.rejected += 1
                record |= {"rejected": str(exc), **tally.add(row["reference"], "")}
                print(f"[{row['lang']}] REJECTED {row['file']}: {exc} (scored as 100% errors)")
            else:
                text = result.transcript.text
                tally.latencies.append(float(result.total_ms))
                rates = tally.add(row["reference"], text)
                record |= {"hypothesis": text, "total_ms": result.total_ms, **rates}
                print(
                    f"[{row['lang']}] WER {rates['wer']:.0%} CER {rates['cer']:.0%} {result.total_ms} ms  {text!r}"
                )
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
    return tallies


def summarise(label: str, tallies: dict[str, Tally]) -> None:
    print(f"\n{label}  (pooled over clips; WER folds diacritics, WER+d keeps them)")
    for language, t in sorted(tallies.items()):
        p50 = median(t.latencies) if t.latencies else float("nan")
        print(
            f"  {language}: clips={t.clips} rejected={t.rejected}  WER {t.rate('wer'):.1%}  "
            f"WER+d {t.rate('wer_diacritics'):.1%}  CER {t.rate('cer'):.1%}  latency p50 {p50:.0f} ms"
        )


async def main(
    proxy: bool, compare: bool, auto_language: bool, cache: Path, sources: set[str] | None
) -> None:
    load_env_file(ROOT / ".env")  # HF_TOKEN for one-time conversions of gated models
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    rows = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or []
    if sources:
        rows = [r for r in rows if r.get("source", "tts") in sources]
    models = {code: PROXY for code in registry.asr_models()} if proxy else registry.asr_models()
    tokens = {PROXY: None} if proxy else registry.asr_whisper_languages()
    if auto_language:  # let each fine-tuned model detect the language (NCAIR's own generation config)
        tokens = dict.fromkeys(tokens)
    ffmpeg = find_ffmpeg()
    for quantization in ("int8", "float32") if compare else ("int8",):
        loader = ct2_loader(cache_dir=cache, whisper_languages=tokens, quantization=quantization)
        languages = sorted({row["lang"] for row in rows})
        asr = NatlasASR(models, loader=loader, max_loaded=len(languages))
        await asr.preload(languages)  # latency below is per message, not first-use model loading
        label = f"{quantization}{'-proxy' if proxy else ''}{'-auto' if auto_language else ''}"
        log_path = MANIFEST.parent / "results" / f"{label}.jsonl"
        summarise(f"CTranslate2 {label}", await evaluate(asr, rows, ffmpeg, log_path))
        print(f"  per-clip log: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--compare", action="store_true", help="int8 vs float32 on the same weights")
    parser.add_argument("--auto-language", action="store_true", help="no forced Whisper language token")
    parser.add_argument("--source", nargs="*", help="only these manifest sources (e.g. fleurs whatsapp)")
    parser.add_argument(
        "--cache", type=Path, default=Path(os.environ.get("KUDIGUARD_MODEL_CACHE", str(ROOT / "models")))
    )
    args = parser.parse_args()
    asyncio.run(main(args.proxy, args.compare, args.auto_language, args.cache, set(args.source or ())))
