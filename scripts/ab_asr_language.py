"""A/B of Whisper language tokens for one language's NCAIR1 speech model (docs/SPIKES.md H: Igbo).

    python scripts/ab_asr_language.py --lang ig --tokens auto yo ha en --rounds 3 [--cache ***REMOVED***/ncair/.cache]

Runs every clip of the language in data/eval/asr/manifest.yaml through the production audio path (decode, trim,
ASR; CTranslate2 int8) once per token ("auto" = no forced token: the model detects the language), interleaved and
rotated per clip so CPU-load drift hits every arm alike, ``--rounds`` times for latency. Accuracy (pooled WER/CER, as
scripts/eval_asr.py) is scored on the first round (decoding is greedy, so later rounds repeat it). Also prints which
language the model itself detects on each clip and how many transcripts each token shares with "auto".
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from statistics import median

import yaml

from kudiguard.languages import LanguageRegistry
from kudiguard.speech.audio import decode, find_ffmpeg, trim_silence
from kudiguard.speech.base import ASRError, AudioError
from kudiguard.speech.metrics import POOLED_METRICS
from kudiguard.speech.natlas_asr import NatlasASR, ct2_loader
from kudiguard.speech.service import transcribe_upload

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "eval" / "asr" / "manifest.yaml"


async def detected_languages(cache: Path, repo: str, rows: list[dict[str, str]], ffmpeg: str) -> list[str]:
    from faster_whisper import WhisperModel  # optional dependency

    model = WhisperModel(
        str(cache / "ct2" / f"{repo.replace('/', '--')}-int8"), device="cpu", compute_type="int8"
    )
    found = []
    for row in rows:
        try:
            audio = trim_silence(await decode((MANIFEST.parent / row["file"]).read_bytes(), ffmpeg=ffmpeg))
        except AudioError as exc:
            found.append(f"rejected ({exc})")
            continue
        _segments, info = model.transcribe(
            audio.samples, task="transcribe", beam_size=1, without_timestamps=True
        )
        found.append(f"{info.language} (p={info.language_probability:.2f})")
    return found


async def main(args: argparse.Namespace) -> None:
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    repo = registry.asr_models()[args.lang]
    rows = [r for r in yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) if r["lang"] == args.lang]
    ffmpeg = find_ffmpeg()
    for row, language in zip(rows, await detected_languages(args.cache, repo, rows, ffmpeg), strict=True):
        print(f"detected {language:22} {row['file']}")
    tokens: list[str] = args.tokens
    asrs = {}
    for token in tokens:
        loader = ct2_loader(
            cache_dir=args.cache, whisper_languages={repo: None if token == "auto" else token}
        )
        asrs[token] = NatlasASR({args.lang: repo}, loader=loader, max_loaded=1)
        await asrs[token].preload([args.lang])
    counts = {t: {k: [0, 0] for k in POOLED_METRICS} for t in tokens}
    latencies: dict[str, list[int]] = {t: [] for t in tokens}
    texts: dict[str, list[str]] = {t: [] for t in tokens}
    for round_ in range(args.rounds):
        for i, row in enumerate(rows):
            data = (MANIFEST.parent / row["file"]).read_bytes()
            for token in tokens[i % len(tokens) :] + tokens[: i % len(tokens)]:
                try:
                    result = await transcribe_upload(asrs[token], data, args.lang, ffmpeg=ffmpeg)
                except (AudioError, ASRError) as exc:  # scored as 100% errors, like eval_asr.py
                    text = ""
                    if round_ == 0:
                        print(f"[{token}] rejected {row['file']}: {exc}")
                else:
                    text = result.transcript.text
                    latencies[token].append(result.total_ms)
                if round_ == 0:
                    texts[token].append(text)
                    for key, (count, tone_sensitive) in POOLED_METRICS.items():
                        edits, total = count(row["reference"], text, tone_sensitive=tone_sensitive)
                        counts[token][key][0] += edits
                        counts[token][key][1] += total
    print(f"\n{args.lang}: {len(rows)} clips, {args.rounds} round(s); latency = decode + ASR per clip")
    for token in tokens:
        rate = {k: e / max(n, 1) for k, (e, n) in counts[token].items()}
        same = sum(a == b for a, b in zip(texts[token], texts.get("auto", texts[token]), strict=True))
        lat = latencies[token]
        print(
            f"  {token:5} WER {rate['wer']:.1%}  WER+d {rate['wer_diacritics']:.1%}  CER {rate['cer']:.1%}  "
            f"p50 {median(lat):.0f} ms  mean {sum(lat) / len(lat):.0f} ms  same as auto {same}/{len(rows)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ig")
    parser.add_argument("--tokens", nargs="+", default=["auto", "yo", "ha", "en"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--cache", type=Path, default=Path(os.environ.get("KUDIGUARD_MODEL_CACHE", str(ROOT / "models")))
    )
    asyncio.run(main(parser.parse_args()))
