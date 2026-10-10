"""Operator tools for donated voice notes: transcription export, WER, stats, access and erasure requests, purge."""

import asyncio
import csv
import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from kudiguard import cli, voice_cli
from kudiguard.donation import DONATION_VERSION, Donation, Sealer, VoiceVault
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage.base import UserState, VoiceSample
from kudiguard.storage.sqlite import SQLiteStore

SECRET = "s" * MIN_SECRET_BYTES
PHONE = "+2348031234567"
OGG = b"OggS" + b"\x00" * 64


@dataclass
class Env:
    root: Path
    db: Path
    voice: Path
    ids: dict[str, list[str]]


def sample(sample_id: str = "a" * 32, language: str = "ha", **kw: object) -> VoiceSample:
    values: dict[str, object] = {"id": sample_id, "user_hash": "u", "week": date(2026, 10, 12), "language": language,
                                 "channel": "whatsapp", "consent_version": DONATION_VERSION, "hypothesis": "x"}  # fmt: skip
    values.update(kw)
    return VoiceSample(**values)  # type: ignore[arg-type]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    db, voice = tmp_path / "var" / "kg.db", tmp_path / "var" / "voice"  # var/: the database's folder
    monkeypatch.setenv("KUDIGUARD_HASH_SECRET", SECRET)
    monkeypatch.setenv("KUDIGUARD_DB_PATH", str(db))
    monkeypatch.setenv("KUDIGUARD_VOICE_DIR", str(voice))
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)
    monkeypatch.setattr(voice_cli, "ROOT", tmp_path / "repo")  # a fake checkout with var/ and exports/
    hasher = UserHasher(SECRET.encode())
    ids: dict[str, list[str]] = {"donor": [], "other": []}

    async def seed() -> None:
        store = await SQLiteStore.open(db, hasher=hasher)
        vault = VoiceVault(
            voice, store, Sealer(SECRET.encode()), clock=lambda: datetime(2026, 9, 2, tzinfo=UTC)
        )
        donor = hasher.user_id(PHONE)
        await store.save_user(
            UserState(
                donor, donation_version=DONATION_VERSION, donation_at=datetime(2026, 10, 10, tzinfo=UTC)
            )
        )
        for n, (user, language, text) in enumerate(
            [(donor, "ha", "=HYPERLINK(evil)"), (donor, "yo", "ẹ ṣé"), ("other", "ha", "sannu"),
             ("other", "ha", "na gode"), ("other", "ig", "daalụ")]
        ):  # fmt: skip
            sample_id = await vault.donate(
                Donation(
                    user, "whatsapp" if n else "web", language, OGG + bytes([n]), text, DONATION_VERSION, 2.0
                )
            )
            assert sample_id
            ids["donor" if user == donor else "other"].append(sample_id)
        await store.close()

    asyncio.run(seed())
    yield Env(tmp_path, db, voice, ids)


def manifest_rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM: opens cleanly in Excel
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# --- the transcription loop ------------------------------------------------------------------------------------


def test_export_then_wer(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    out = env.db.parent / "export"
    assert (
        cli.main(["voice", "export", "--out", str(out), "--languages", "ha,yo", "--n", "2", "--seed", "7"])
        == 0
    )
    printed = capsys.readouterr().out
    assert "exported 3 voice samples" in printed and "PRIVACY" in printed and "delete every copy" in printed
    rows = manifest_rows(out / voice_cli.MANIFEST)
    assert list(rows[0]) == list(voice_cli.COLUMNS)
    assert [r["language"] for r in rows] == ["ha", "ha", "yo"]  # 2 of the 3 ha samples, the only yo one
    assert all(r["reference"] == "" and r["week"] == "2026-08-31" and r["duration_s"] == "2.0" for r in rows)
    evil = [r for r in rows if "HYPERLINK" in r["asr_hypothesis"]]
    assert all(r["asr_hypothesis"].startswith("'=") for r in evil)  # formula-injection safe
    for row in rows:
        assert (out / "audio" / f"{row['id']}.ogg").read_bytes().startswith(OGG)  # decrypted original

    for row in rows:  # the transcriber types references (here: the hypothesis itself, or a variant)
        row["reference"] = voice_cli.plain_cell(row["asr_hypothesis"]) if row["language"] == "ha" else "ẹ ṣe"
    with (out / voice_cli.MANIFEST).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=voice_cli.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert cli.main(["voice", "wer", str(out / voice_cli.MANIFEST)]) == 0
    report = capsys.readouterr().out
    assert "ha" in report and "yo" in report and "all" in report and "PRIVACY" in report
    assert "  0.0%" in report  # ha: perfect; yo: a tone-mark difference only shows in WER+d


def test_export_is_reproducible_and_refuses_unsafe_folders(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    def ids(folder: str) -> list[str]:
        assert (
            cli.main(["voice", "export", "--out", str(env.db.parent / folder), "--n", "1", "--seed", "3"])
            == 0
        )
        return [r["id"] for r in manifest_rows(env.db.parent / folder / voice_cli.MANIFEST)]

    assert ids("a") == ids("b")
    assert (
        cli.main(["voice", "export", "--out", str(env.db.parent / "a")]) == 2
    )  # never overwrites a manifest
    assert "already exists" in capsys.readouterr().err
    repo = env.root / "repo"
    assert cli.main(["voice", "export", "--out", str(repo / "data" / "x")]) == 2  # tracked folder
    assert "git-ignored" in capsys.readouterr().err
    assert cli.main(["voice", "export", "--out", str(env.root / "Desktop" / "x")]) == 2  # anywhere else
    assert "git-ignored" in capsys.readouterr().err
    assert cli.main(["voice", "export", "--out", str(repo / "var" / "voice-export")]) == 0
    assert cli.main(["voice", "export", "--out", str(repo / "exports" / "v")]) == 0


def test_export_skips_samples_deleted_or_tampered_meanwhile(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    (env.voice / f"{env.ids['other'][0]}.kgv").unlink()
    tampered = env.voice / f"{env.ids['other'][1]}.kgv"
    tampered.write_bytes(tampered.read_bytes()[:-1] + b"\x00")
    assert cli.main(["voice", "export", "--out", str(env.db.parent / "x")]) == 0
    printed = capsys.readouterr().out
    assert "exported 3 voice samples" in printed and "2 skipped" in printed


def test_stats(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["voice", "stats"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "ha: 2 donors, 3 samples, 0.1 min" and lines[-1] == "all: 2 donors, 5 samples, 0.2 min"


# --- data-subject requests and retention ----------------------------------------------------------------------


def test_access_request_lists_and_copies_only_the_donors_samples(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["voice", "access", "--user", "0803 123 4567"]) == 0  # any phone format
    printed = capsys.readouterr().out
    assert f"version {DONATION_VERSION}, given 2026-10-10" in printed
    assert all(i in printed for i in env.ids["donor"]) and not any(i in printed for i in env.ids["other"])
    assert "2 donated voice samples" in printed
    out = env.db.parent / "access"
    assert cli.main(["voice", "access", "--user", PHONE, "--out", str(out)]) == 0
    assert sorted(p.stem for p in (out / "audio").iterdir()) == sorted(env.ids["donor"])
    assert cli.main(["voice", "access", "--user", "web:nobody-at-all-123"]) == 0
    assert "none on record" in capsys.readouterr().out


def test_erase_request(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["voice", "erase", "--user", PHONE]) == 0
    assert "deleted 2 donated voice samples" in capsys.readouterr().out
    assert sorted(p.stem for p in env.voice.glob("*.kgv")) == sorted(env.ids["other"])
    assert cli.main(["voice", "access", "--user", PHONE]) == 0
    assert "none on record" in capsys.readouterr().out
    assert cli.main(["voice", "erase", "--user", "web:never-donated-session"]) == 0  # no user row: still fine


def test_remove_flagged_samples(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    flagged = env.ids["other"][0]
    assert cli.main(["voice", "remove", flagged, "f" * 32]) == 0
    assert "deleted 1 of 2 voice samples" in capsys.readouterr().out
    assert not (env.voice / f"{flagged}.kgv").exists() and len(list(env.voice.glob("*.kgv"))) == 4
    assert cli.main(["voice", "remove", "../../etc/passwd"]) == 2  # ids are validated
    assert "invalid sample id" in capsys.readouterr().err


def test_purge_removes_expired_voice_samples(
    env: Env, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(["purge"]) == 0
    assert capsys.readouterr().out.strip().endswith("0 donated voice samples older than 365 days")
    monkeypatch.setenv("KUDIGUARD_VOICE_RETENTION_DAYS", "1")
    assert cli.main(["purge"]) == 0
    assert "5 donated voice samples older than 1 days" in capsys.readouterr().out
    assert list(env.voice.glob("*.kgv")) == []


def test_voice_commands_need_the_database(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KUDIGUARD_DB_PATH", str(env.root / "missing.db"))
    assert cli.main(["voice", "stats"]) == 2
    assert "no evidence database" in capsys.readouterr().err


# --- pure helpers -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head", "suffix"),
    [(b"OggS", ".ogg"), (b"\x1aE\xdf\xa3", ".webm"), (b"\x00\x00\x00\x20ftypM4A ", ".m4a"), (b"RIFF", ".wav"),
     (b"ID3", ".mp3"), (b"\xff\xfb", ".mp3"), (b"??", ".bin"), (b"", ".bin")],
)  # fmt: skip
def test_audio_suffix(head: bytes, suffix: str) -> None:
    assert voice_cli.audio_suffix(head + b"\x00" * 8) == suffix


@pytest.mark.parametrize(
    "value",
    ["=1+1", "+234", "-x", "@SUM", "|cmd", "\tx", "\rx", "\nx", " =x", "\u00a0=x", "\u3000+x", "\uff1dSUM(1)",
     "\uff0b1", "\uff0d1", "\uff20x", "'=x", "''=x", "plain", "'quoted", "", " plain"],
)  # fmt: skip
def test_cells_round_trip(value: str) -> None:
    cell = voice_cli.spreadsheet_cell(value)
    assert not cell.lstrip().startswith(voice_cli.FORMULA_PREFIXES)
    assert voice_cli.plain_cell(cell) == value  # exact, also for a real leading apostrophe


def test_cells_only_escape_what_spreadsheets_evaluate() -> None:
    assert voice_cli.spreadsheet_cell("sannu") == "sannu" and voice_cli.spreadsheet_cell("'hi") == "'hi"
    assert voice_cli.spreadsheet_cell(" =1") == "' =1" and voice_cli.plain_cell("'hi") == "'hi"


def test_wer_masks_numbers_in_typed_references() -> None:
    """Transcribers may type the digits they hear; the hypothesis is stored with [number], so both are masked."""
    rows = [{"language": "en", "asr_hypothesis": "call [number] now", "reference": "call 08031234567 now"}]
    assert voice_cli.wer_by_language(rows)["en"]["wer"] == 0


def test_preconsent_records_form_consent_for_hashed_numbers(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    signups = env.root / "signups.csv"
    signups.write_text(
        "number,age_confirmed,own_voice_confirmed,consented_at\n"
        "0809 999 9999,yes,yes,2026-10-09T10:00:00+01:00\n"
        f"{PHONE},TRUE,1,2026-10-09T09:00:00Z\n",
        encoding="utf-8-sig",
    )
    assert cli.main(["voice", "preconsent", "--csv", str(signups), "--version", DONATION_VERSION]) == 0
    printed = capsys.readouterr().out
    assert "for 2 testers" in printed and "Delete the sign-up export" in printed

    async def load(number: str) -> UserState | None:
        store = await SQLiteStore.open(env.db, hasher=UserHasher(SECRET.encode()))
        try:
            return await store.get_user(UserHasher(SECRET.encode()).user_id(number))
        finally:
            await store.close()

    new = asyncio.run(load("+2348099999999"))
    assert new is not None and new.donation_source == "form" and new.donation_version == DONATION_VERSION
    assert (
        new.donation_at == datetime(2026, 10, 9, 9, 0, tzinfo=UTC) and new.consented_at is None
    )  # chat consent: later
    assert new.donation_offered_version == DONATION_VERSION  # not asked again in the chat
    existing = asyncio.run(load(PHONE))
    assert existing is not None and existing.donation_source == "form"
    assert "2348099999999" not in env.db.read_bytes().decode("latin-1")  # only hashes stored


@pytest.mark.parametrize(
    ("row", "version"),
    [("0809 999 9999,yes,no,2026-10-09T10:00:00Z", DONATION_VERSION),  # own voice not ticked
     ("0809 999 9999,,yes,2026-10-09T10:00:00Z", DONATION_VERSION),  # 18+ not ticked
     (",yes,yes,2026-10-09T10:00:00Z", DONATION_VERSION),  # no number
     ("0809 999 9999,yes,yes,2026-10-09T10:00:00", DONATION_VERSION),  # no timezone
     ("0809 999 9999,yes,yes,not a date", DONATION_VERSION),
     ("0809 999 9999,yes,yes,2099-01-01T00:00:00Z", DONATION_VERSION),  # in the future
     ("0809 999 9999,yes,yes,2026-10-09T10:00:00Z", "2020-01-01")],  # an old form text
)  # fmt: skip
def test_preconsent_rejects_incomplete_rows_and_imports_nothing(
    env: Env, capsys: pytest.CaptureFixture[str], row: str, version: str
) -> None:
    signups = env.root / "signups.csv"
    signups.write_text(
        f"number,age_confirmed,own_voice_confirmed,consented_at\n{PHONE},yes,yes,2026-10-09T09:00:00Z\n{row}\n",
        encoding="utf-8",
    )
    assert cli.main(["voice", "preconsent", "--csv", str(signups), "--version", version]) == 2
    assert "nothing was imported" in capsys.readouterr().err or version != DONATION_VERSION
    assert cli.main(["voice", "access", "--user", "0809 999 9999"]) == 0
    assert "none on record" in capsys.readouterr().out


def test_preconsent_needs_the_expected_columns(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    signups = env.root / "signups.csv"
    signups.write_text("number,consented_at\n0809,2026-10-09T09:00:00Z\n", encoding="utf-8")
    assert cli.main(["voice", "preconsent", "--csv", str(signups), "--version", DONATION_VERSION]) == 2
    assert "missing columns" in capsys.readouterr().err


def test_wipe_needs_all_and_deletes_everything(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["voice", "wipe"]) == 2
    assert "refusing" in capsys.readouterr().err and len(list(env.voice.glob("*.kgv"))) == 5
    assert cli.main(["voice", "wipe", "--all"]) == 0
    assert "deleted all 5" in capsys.readouterr().out and list(env.voice.glob("*.kgv")) == []


def test_exports_never_go_to_cloud_synced_folders(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("OneDrive", "Dropbox", "Google Drive"):
        assert cli.main(["voice", "export", "--out", str(env.db.parent / name / "x")]) == 2
        assert "cloud-synced" in capsys.readouterr().err


def test_purge_needs_an_existing_database(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KUDIGUARD_DB_PATH", str(env.root / "elsewhere" / "kg.db"))
    assert cli.main(["purge"]) == 2
    assert "no evidence database" in capsys.readouterr().err
    assert not (env.root / "elsewhere").exists() and len(list(env.voice.glob("*.kgv"))) == 5


def test_choose_is_stratified_seeded_and_filtered() -> None:
    pool = [sample(f"{n:032x}", "ha" if n % 3 else "yo") for n in range(12)]
    first = voice_cli.choose(pool, per_language=2, seed=1)
    assert [s.language for s in first] == ["ha", "ha", "yo", "yo"]
    assert first == voice_cli.choose(list(reversed(pool)), per_language=2, seed=1)  # input order irrelevant
    assert len(voice_cli.choose(pool)) == 12
    assert {s.language for s in voice_cli.choose(pool, languages={"yo"}, per_language=99)} == {"yo"}


def test_wer_report_pools_counts_and_shows_pending_rows() -> None:
    rows = [
        {"language": "yo", "asr_hypothesis": "ẹ ṣe o", "reference": "ẹ ṣé o"},  # tone mark only
        {"language": "yo", "asr_hypothesis": "", "reference": ""},  # not transcribed yet
        {"language": "ha", "asr_hypothesis": "na gode sosai", "reference": "na gode"},
        {"language": "", "asr_hypothesis": "x", "reference": "x"},
    ]
    report = voice_cli.wer_by_language(rows)
    assert list(report) == ["?", "ha", "yo", "all"]
    assert report["yo"]["wer"] == 0 and report["yo"]["wer_diacritics"] == pytest.approx(1 / 3)
    assert report["ha"]["wer"] == pytest.approx(1 / 2) and report["yo"]["without_reference"] == 1
    assert report["all"]["clips"] == 3 and report["all"]["wer"] == pytest.approx(1 / 6)
    only_pending = voice_cli.wer_by_language([{"language": "ig", "asr_hypothesis": "x", "reference": " "}])
    assert only_pending["ig"]["clips"] == 0 and math.isnan(only_pending["ig"]["wer"])
    assert "nan" in voice_cli.format_wer(only_pending)


def test_manifest_must_have_the_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("id,text\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a voice manifest"):
        voice_cli.read_manifest(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="not a voice manifest"):
        voice_cli.read_manifest(path)


def test_stats_helper_handles_unknown_durations() -> None:
    result = voice_cli.stats([sample(duration_s=None), sample("b" * 32, user_hash="v", duration_s=90.0)])
    assert result == {
        "ha": {"donors": 2, "samples": 2, "minutes": 1.5},
        "all": {"donors": 2, "samples": 2, "minutes": 1.5},
    }
