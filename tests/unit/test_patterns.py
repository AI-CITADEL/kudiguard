"""De-identified scam pattern library: de-identification, storage contract, recording rules, retention, export."""

import asyncio
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest
from test_pipeline import OTP_SCAM, Harness
from test_pipeline import h as _harness_fixture

from kudiguard.domain import Flag, Verdict
from kudiguard.i18n import Msg
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.base import Purpose
from kudiguard.patterns import MAX_PATTERN_CHARS, ScamPattern, deidentify
from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher
from kudiguard.storage.memory import MemoryStore
from kudiguard.storage.sqlite import MIGRATIONS, SQLiteStore

h = _harness_fixture  # the shared conversation-harness fixture (pytest finds fixtures by module attribute)
HASHER = UserHasher(b"p" * MIN_SECRET_BYTES)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("click https://bit.ly/abc?u=8031234567 now", "click [link: bit.ly] now"),
        ("go to www.Fake-Bank.ng/login?id=55", "go to [link: fake-bank.ng]"),
        ("call 0803 123 4567 or 803 123 4567", "call [number] or [number]"),
        ("pay ₦5000 to 0123456789", "pay ₦5000 to [number]"),
        ("mail x@y.com", "mail [email]"),
        ("  lots   of\n spaces ", "lots of spaces"),
    ],
)
def test_deidentify(raw: str, expected: str) -> None:
    assert deidentify(raw) == expected


def test_deidentify_caps_length() -> None:
    assert len(deidentify("word " * 2000)) == MAX_PATTERN_CHARS


def pattern(day: date, text: str = "send me the OTP") -> ScamPattern:
    return ScamPattern.from_check(
        language="en", channel="whatsapp", verdict=Verdict.LIKELY_SCAM, flags=(Flag.ASKS_CREDENTIALS,),
        text=text, english=f"{text} 08031234567", day=day,
    )  # fmt: skip


@pytest.fixture(params=["memory", "sqlite"])
async def store(request: pytest.FixtureRequest, tmp_path: Path):  # type: ignore[no-untyped-def]
    impl = (
        MemoryStore(HASHER)
        if request.param == "memory"
        else await SQLiteStore.open(tmp_path / "kg.db", hasher=HASHER)
    )
    yield impl
    await impl.close()


async def test_store_contract(store) -> None:  # type: ignore[no-untyped-def]
    today = date(2026, 10, 8)
    later, earlier = pattern(today), pattern(today - timedelta(days=800), "old scam")
    await store.add_pattern(later)
    await store.add_pattern(earlier)
    rows = await store.export_patterns()
    assert [r["text"] for r in rows] == ["old scam", "send me the OTP"]  # by week, not insertion order
    assert rows[1]["english"] == "send me the OTP [number]" and rows[1]["flags"] == ["asks_credentials"]
    assert rows[1]["week"] == "2026-10-05"  # Monday of the ISO week, not the day
    assert set(rows[1]) == {
        "id",
        "week",
        "language",
        "channel",
        "verdict",
        "flags",
        "text",
        "english",
    }  # no user link
    assert await store.purge_patterns_before(today - timedelta(days=730)) == 1
    assert [r["text"] for r in await store.export_patterns()] == ["send me the OTP"]


async def test_sqlite_upgrade_from_schema_v1(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    db = sqlite3.connect(path, isolation_level=None)
    db.executescript(f"BEGIN; {MIGRATIONS[0]}; PRAGMA user_version=1; COMMIT;")
    db.close()
    store = await SQLiteStore.open(path, hasher=HASHER)
    await store.add_pattern(pattern(date(2026, 10, 8)))
    assert len(await store.export_patterns()) == 1
    await store.close()
    raw = sqlite3.connect(path)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert (
        "WITHOUT ROWID"
        in raw.execute("SELECT sql FROM sqlite_master WHERE name='scam_patterns'").fetchone()[0]
    )
    raw.close()


async def test_scam_checks_feed_the_library_without_any_user_link(h: Harness) -> None:
    await h.onboard("en")
    await h.say(f"{OTP_SCAM} He said to call him on 08031234567", message_id="wamid.UNIQUE-ID")
    [row] = await h.store.export_patterns()
    assert row["verdict"] == "likely_scam" and row["channel"] == "whatsapp" and "[number]" in row["text"]
    dumped = json.dumps(row)
    assert h.user_hash not in dumped and "wamid" not in dumped and "08031234567" not in dumped
    interactions = await h.store.export_interactions()
    assert str(interactions[-1]["id"]) not in {str(v) for v in row.values()}


async def test_questions_and_greetings_do_not(h: Harness) -> None:
    await h.onboard("en")
    h.llm.script[Purpose.ANALYSE] = [__import__("fakes").analysis("x", "question")]
    await h.say("How do I open a savings account?")
    await h.say("thanks!")
    assert await h.store.export_patterns() == []


async def test_delete_also_removes_patterns_from_the_users_stored_messages(h: Harness) -> None:
    await h.onboard("en")
    await h.say(OTP_SCAM)
    other = "+2348099999999"
    await h.onboard("en", user=other)
    await h.say(f"{OTP_SCAM} He is from the bank", user=other)
    assert len(await h.store.export_patterns()) == 2
    assert await h.say("DELETE") == [h.t(Msg.DELETED)]
    [kept] = await h.store.export_patterns()  # only the other user's example remains
    assert "He is from the bank" in kept["text"]


async def test_library_failure_never_breaks_the_answer(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(pattern: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(h.store, "add_pattern", broken)
    await h.onboard("en")
    assert (await h.say(OTP_SCAM))[0].startswith(h.t(Msg.VERDICT_LIKELY_SCAM))


async def test_welcome_links_the_privacy_notice(h: Harness) -> None:
    from kudiguard.pipeline import Conversation

    h.conversation = Conversation(
        engine=h.engine, store=h.store, messenger=h.messenger, hasher=h.hasher,
        privacy_url="https://kudiguard.example/privacy",
    )  # fmt: skip
    texts = await h.say("hello")
    assert texts[0].endswith(h.t(Msg.PRIVACY_LINK, url="https://kudiguard.example/privacy"))


async def test_purge_loop_also_expires_patterns() -> None:
    from kudiguard.api.services import purge_loop

    store = MemoryStore(HASHER)
    await store.add_pattern(pattern(date(2020, 1, 1)))
    task = asyncio.create_task(purge_loop(store, 90, interval_s=0.01, pattern_retention_days=730))
    await asyncio.sleep(0.03)
    task.cancel()
    assert await store.export_patterns() == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("see bit.ly/abc123 now", "see [link: bit.ly] now"),
        ("join t.me/scamgroup", "join [link: t.me]"),
        ("http://user:pass@evil.com/a", "[link]"),  # credentials: nothing of the link is kept
        ("HTTPS://Bit.Ly/X?u=1", "[link: bit.ly]"),
        ("e.g. this, i.e. that, version 2.5", "e.g. this, i.e. that, version 2.5"),
    ],
)
def test_links_without_scheme_and_credentials(raw: str, expected: str) -> None:
    assert deidentify(raw) == expected


def test_week_of() -> None:
    from kudiguard.patterns import week_of

    assert week_of(date(2026, 10, 11)) == date(2026, 10, 5) and week_of(date(2026, 10, 5)) == date(
        2026, 10, 5
    )


def test_web_consent_version_matches_the_pipeline() -> None:
    from kudiguard.pipeline import CONSENT_VERSION

    page = (Path(__file__).resolve().parents[2] / "src/kudiguard/api/static/voice.html").read_text(
        encoding="utf-8"
    )
    assert f'const CONSENT_VERSION = "{CONSENT_VERSION}";' in page


def test_web_donation_version_matches_the_server() -> None:
    from kudiguard.donation import DONATION_VERSION

    page = (Path(__file__).resolve().parents[2] / "src/kudiguard/api/static/voice.html").read_text(
        encoding="utf-8"
    )
    assert f'const DONATION_VERSION = "{DONATION_VERSION}";' in page
    assert '<input type="checkbox" id="donate">' in page  # unticked by default
    assert '"X-KudiGuard-Donation": DONATION_VERSION' in page and "/web/donation/withdraw" in page
    assert page.count("donate: ") == 4 and page.count("donationOff: ") == 4  # en, ha, yo, ig
    assert page.count("riskNigeria: ") == 4 and page.count("copy: ") == 4
    assert page.count("{risk}") == 8 + 1  # four consent and four donation texts + the replacement in t()
    assert (
        page.count("copied: ") == 4 and "Meta" not in page.split("<script>")[1]
    )  # the web is Cloudflare only
    assert '<code id="code"></code>' in page and "navigator.clipboard.writeText(session)" in page
    assert "18 or older" in page and "only my own voice" in page and "12 months" in page


async def test_delete_user_removes_matching_patterns_in_both_stores(store) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from kudiguard.domain import Interaction, MessageKind

    when = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
    text = "send me the OTP they sent you"
    await store.add_interaction(
        Interaction(channel="whatsapp", message_id="m1", user_hash="u1", language="en", kind=MessageKind.TEXT,
                    transcript=text, verdict=Verdict.LIKELY_SCAM, created_at=when)
    )  # fmt: skip
    await store.add_interaction(  # a non-scam turn of the same user: no pattern to look for
        Interaction(channel="whatsapp", message_id="m2", user_hash="u1", language="en", kind=MessageKind.TEXT,
                    transcript="hello", created_at=when)
    )  # fmt: skip
    await store.add_pattern(pattern(when.date(), text))
    await store.add_pattern(pattern(when.date(), "someone else's scam"))
    removed = await store.delete_user("u1")
    assert removed == 3  # one pattern + two interactions
    assert [r["text"] for r in await store.export_patterns()] == ["someone else's scam"]


def test_examples_round_trip_exactly_through_the_stored_transcript() -> None:
    """DELETE matches an example by deidentify(stored transcript); the example is built from redact(raw). Both must
    be equal for any input, or erasure silently misses it."""
    import random

    from kudiguard.text import redact

    pieces = ["send", "OTP", "0803 123 4567", "+234 803 123 4567", "08031234567.Me", "acct 0123456789.Click",
              "http://john@evil.com/x?u=1", "https://a:b@bank.ng", "bit.ly/abc?i=9", "www.Fake.ng/a", "N 5000000",  # noqa: RUF001 - unusual characters are the point of this test
              "₦ 2500000", "x@y.com", "5399 8312 3456 7890", "ＡＢＣ", "2 2 3 1 5 6 7 8 9 0 1", "10:30 12 45 2026",  # noqa: RUF001 - unusual characters are the point of this test
              "e.g.", "Musa", "\n", "  ", "0803 123 4567pay.ng", "😀"]  # fmt: skip
    rng = random.Random(8)
    for _ in range(500):
        raw = " ".join(rng.choice(pieces) for _ in range(rng.randint(1, 12)))
        stored = redact(raw)  # what the evidence log keeps (storage.export.protect)
        example = ScamPattern.from_check(language="en", channel="whatsapp", verdict=Verdict.CAUTION, flags=(),
                                         text=redact(raw), english=None, day=date(2026, 10, 8))  # fmt: skip
        assert example.text == deidentify(stored), raw
        assert "8031234567" not in example.text and "evil.com/x" not in example.text, raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("acct0123456789.ng", "[link: acct#.ng]"),
        ("8031234567pay.ng", "[link: #pay.ng]"),
        ("bit.ly/x1", "[link: bit.ly]"),
    ],
)
def test_digit_runs_inside_hosts_are_masked(raw: str, expected: str) -> None:
    assert deidentify(raw) == expected


async def test_weeks_are_utc_even_for_local_clocks() -> None:
    from datetime import datetime, timezone

    from kudiguard.domain import Outcome
    from kudiguard.engine import Answer, AnswerKind
    from kudiguard.pipeline import record_pattern

    store = MemoryStore(HASHER)
    lagos = timezone(timedelta(hours=1))
    answer = Answer("en", AnswerKind.CHECK, Outcome.ANSWERED, verdict=Verdict.LIKELY_SCAM)
    # Monday 00:30 in Lagos is still Sunday in UTC: the week must be the UTC one (the store keeps UTC).
    await record_pattern(store, "whatsapp", answer, OTP_SCAM, datetime(2026, 10, 12, 0, 30, tzinfo=lagos))
    [row] = await store.export_patterns()
    assert row["week"] == "2026-10-05"


# --- found by mutation testing (docs/SPIKES.md, G) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pay123456.ng", "[link: pay#.ng]"),  # six digits in a host: masked
        ("shop12345.ng", "[link: shop12345.ng]"),  # five: kept
        ("https://www.Scam.ng/a?u=1", "[link: scam.ng]"),  # www. dropped after a scheme too
        (
            "visit scam-site.com/x?id=1",
            "visit [link: scam-site.com]",
        ),  # bare hosts: first and last listed TLDs
        ("free.ga/win", "[link: free.ga]"),
    ],
)
def test_more_links(raw: str, expected: str) -> None:
    assert deidentify(raw) == expected


def test_length_caps() -> None:
    assert deidentify("w" * 2000) == "w" * 2000 and deidentify("w" * 2001) == "w" * 2000
    assert deidentify("a" + " " * 3000 + "b") == "a b"  # long whitespace runs do not eat the text
    assert deidentify("a" + " " * 4500 + "b") == "a"  # input is cut at twice the cap before any regex runs


def test_retention_and_channels_match_what_users_are_told(registry: LanguageRegistry) -> None:
    from kudiguard.patterns import PATTERN_CHANNELS, PATTERN_RETENTION_DAYS

    assert PATTERN_RETENTION_DAYS == 730 and "up to 2 years" in registry.get("en").strings[Msg.WELCOME]
    assert {"whatsapp", "web"} == PATTERN_CHANNELS  # end users who consented, never API partners' users
