"""Translation fidelity over eval results (docs/SPIKES.md H): how often English-first translations fail the checks.

    python scripts/check_translations.py data/eval/results/answers-h_final.jsonl [more.jsonl ...] [-v]

For every distinct (language, English draft, translated body) in the files' answered records, reruns
``reply_translation.translation_problems`` (the current checks) and reports, per language, how many accepted
translations would now be rejected and why; plus the rejections the engine itself recorded at run time
(``translation_rejections``: translations rejected before the reply was sent, retried or replaced by fixed text).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from kudiguard.prompts import clean_reply
from kudiguard.reply_translation import problem_kind, strip_reply_tags, translation_problems


def main(paths: list[Path], verbose: bool) -> None:
    accepted: dict[tuple[str, str, str], str] = {}
    recorded: dict[str, Counter[str]] = defaultdict(Counter)
    records: Counter[str] = Counter()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if "meta" in item or not item.get("reply_english"):
                continue
            records[item["lang"]] += 1
            recorded[item["lang"]].update(item.get("translation_rejections", []))
            if item["outcome"] == "answered":
                accepted[(item["lang"], item["reply_english"], item["body"])] = item["id"]
    flagged: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for (lang, english, body), case in accepted.items():
        totals[lang] += 1
        problems = translation_problems(english, clean_reply(strip_reply_tags(body)), lang)
        if problems:
            flagged[lang].update({problem_kind(p) for p in problems} | {"any"})
            if verbose:
                print(f"== {case} [{lang}] {problems}\nEN: {english}\nTR: {body}\n")
    for lang in sorted(records):
        rate = flagged[lang]["any"] / max(totals[lang], 1)
        kinds = {k: v for k, v in flagged[lang].items() if k != "any"}
        print(
            f"{lang}: {totals[lang]} distinct accepted translations, {flagged[lang]['any']} ({rate:.0%}) fail the "
            f"current checks {kinds}; rejected at run time in {records[lang]} records: {dict(recorded[lang])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args.paths, args.verbose)
