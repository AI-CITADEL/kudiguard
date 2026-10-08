"""Fetch public evaluation clips (Google FLEURS, CC-BY-4.0) and convert them to WhatsApp-style voice notes.

    python scripts/fetch_eval_audio.py --per-language ha=20 yo=10 ig=10

Downloads the FLEURS dev split per language (once, into the cache), streams only the needed clips out of the archive,
re-encodes them as Opus 16 kHz / 24 kbps (like WhatsApp voice notes) into data/eval/asr/fleurs/<lang>/, and appends
manifest rows (source: fleurs) with the official human transcriptions. Attribution: data/eval/asr/SOURCES.md.
Caveats recorded with results: read speech (not spontaneous), and FLEURS may overlap NCAIR1 training data.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from kudiguard.speech.audio import find_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval" / "asr"
FLEURS = {"ha": "ha_ng", "yo": "yo_ng", "ig": "ig_ng"}
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


def read_transcripts(tsv: Path) -> dict[str, str]:
    """FLEURS TSV columns: id, file name, raw transcription, normalized transcription, ..."""
    rows: dict[str, str] = {}
    with tsv.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 3 and row[1].endswith(".wav"):
                rows.setdefault(row[1], row[2].strip())
    return rows


def fetch(code: str, count: int, cache: Path, ffmpeg: str) -> list[dict[str, str]]:
    config = FLEURS[code]
    tsv = Path(
        hf_hub_download("google/fleurs", f"data/{config}/dev.tsv", repo_type="dataset", cache_dir=cache)
    )
    archive = Path(
        hf_hub_download(
            "google/fleurs", f"data/{config}/audio/dev.tar.gz", repo_type="dataset", cache_dir=cache
        )
    )
    transcripts = read_transcripts(tsv)
    out_dir = EVAL / "fleurs" / code
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with tarfile.open(archive, "r:gz") as tar, tempfile.TemporaryDirectory(dir=os.environ.get("TMP")) as tmp:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name not in transcripts:
                continue
            wav = Path(tmp) / name
            source = tar.extractfile(member)
            if source is None:
                continue
            wav.write_bytes(source.read())
            target = out_dir / f"{Path(name).stem}.ogg"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                 "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", str(target)],
                check=True,
            )  # fmt: skip
            rows.append(
                {"file": target.relative_to(EVAL).as_posix(), "lang": code, "source": "fleurs",
                 "reference": transcripts[name]}
            )  # fmt: skip
            if len(rows) >= count:
                break
    return rows


def main(counts: dict[str, int], cache: Path) -> None:
    ffmpeg = find_ffmpeg()
    manifest = EVAL / "manifest.yaml"
    existing = yaml.safe_load(manifest.read_text(encoding="utf-8")) or []
    known = {row["file"] for row in existing}
    added = 0
    for code, count in counts.items():
        rows = [r for r in fetch(code, count, cache, ffmpeg) if r["file"] not in known]
        existing.extend(rows)
        added += len(rows)
        print(f"{code}: {len(rows)} clips")
    header = (
        "# ASR evaluation clips. `source`: tts (synthetic), fleurs (Google FLEURS dev, CC-BY-4.0, read speech),\n"
        "# whatsapp (real recordings from testers). Add real voice notes with native-typed references.\n"
    )
    body = yaml.safe_dump(existing, allow_unicode=True, sort_keys=False, width=1000)
    manifest.write_text(header + body, encoding="utf-8")
    print(f"manifest: {len(existing)} rows ({added} added)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-language", nargs="+", default=["ha=20", "yo=10", "ig=10"])
    parser.add_argument("--cache", type=Path, default=Path(r"***REMOVED***\ncair\.cache\huggingface\hub"))
    args = parser.parse_args()
    main({k: int(v) for k, v in (item.split("=") for item in args.per_language)}, args.cache)
