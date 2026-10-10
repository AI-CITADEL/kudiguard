from pathlib import Path

import pytest

from kudiguard.domain import Flag
from kudiguard.knowledge.file_pack import (
    MAX_FACT_CHARS,
    MAX_FACTS,
    FilePackSource,
    KnowledgeConfigError,
    load_manifest,
    split_answer,
)
from kudiguard.knowledge.models import Contact, PackKind, Query

PACKS = Path(__file__).resolve().parents[3] / "data" / "packs"

MANIFEST = "id: acme\nname: Acme\nowner: Acme Ltd\n"


def make_pack(tmp_path: Path, **files: str) -> Path:
    pack = tmp_path / "acme"
    pack.mkdir()
    files.setdefault("pack.yaml", MANIFEST)
    for name, body in files.items():
        (pack / name.replace("_", ".", 1) if name != "pack.yaml" else pack / name).write_text(
            body, encoding="utf-8"
        )
    return pack


# --- shipped packs ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def core() -> FilePackSource:
    return FilePackSource.load(PACKS / "kudiguard-core")


def test_core_pack_covers_every_red_flag(core: FilePackSource) -> None:
    assert core.manifest.kind is PackKind.CORE
    covered = {flag for entry in core.entries for flag in entry.flags}
    assert covered == set(Flag)


def test_core_pack_has_a_verified_general_contact(core: FilePackSource) -> None:
    assert any(c.verified and not c.flags for c in core.contacts)


async def test_core_flag_lookup(core: FilePackSource) -> None:
    result = await core.search(Query(text="", flags=(Flag.ACCOUNT_THREAT, Flag.ASKS_CREDENTIALS)))
    assert [e.id for e in result.flag_entries] == ["flag-asks-credentials", "flag-account-threat"]


@pytest.mark.parametrize(
    ("text", "best"),
    [
        ("How do I start saving money?", "saving-basics"),
        ("What is a BVN?", "bvn-basics"),
        ("Is it wise to borrow from a loan app?", "loan-apps"),
        ("My transfer failed but I was debited", "failed-transfer"),
        ("What is SIM swap fraud?", "faq-sim-swap"),
    ],
)
async def test_core_text_search(core: FilePackSource, text: str, best: str) -> None:
    assert (await core.search(Query(text=text))).hits[0].entry.id == best


def test_money_skills_pack_ships_enabled_below_core(core: FilePackSource) -> None:
    money = FilePackSource.load(PACKS / "money-skills")
    assert money.manifest.kind is PackKind.PARTNER  # 'core' is reserved for the safety pack
    assert money.manifest.enabled and not money.manifest.sample
    assert money.manifest.priority < core.manifest.priority  # supplements core, never outranks it on ties
    assert not money.contacts  # report channels live in the verified core pack only
    assert not {e.id for e in money.entries} & {e.id for e in core.entries}
    assert not any(e.flags for e in money.entries)  # never takes a red-flag slot from core safety guidance
    assert money.manifest.cite is False  # our own content, not cited as a third party


def test_sample_bank_pack_loads_as_partner_sample() -> None:
    bank = FilePackSource.load(PACKS / "demo-bank")
    assert bank.manifest.kind is PackKind.PARTNER
    assert bank.manifest.sample is True
    assert bank.manifest.priority > FilePackSource.load(PACKS / "kudiguard-core").manifest.priority
    assert {"transfer-limit", "demo-never-ask"} <= {e.id for e in bank.entries}


async def test_refresh_is_a_noop(core: FilePackSource) -> None:
    await core.refresh()
    assert len(core) == len(core.entries)


# --- loading rules ------------------------------------------------------------------------------------


def test_minimal_pack_with_only_faq(tmp_path: Path) -> None:
    pack = make_pack(tmp_path, faq_csv="question,answer\nWhat is X?,X is Y.\n")
    source = FilePackSource.load(pack)
    assert [(e.id, e.title, e.facts) for e in source.entries] == [("faq-1", "What is X?", ("X is Y.",))]


def test_faq_keywords_split_and_bom_tolerated(tmp_path: Path) -> None:
    pack = make_pack(tmp_path, faq_csv="﻿id,question,answer,keywords\nq1,Q?,A.,one; two\n")
    assert FilePackSource.load(pack).entries[0].keywords == ("one", "two")


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"pack.yaml": "name: x\n"}, "pack.yaml"),
        ({"pack.yaml": "id: [unclosed"}, "pack.yaml"),
        ({"pack.yaml": MANIFEST + "cite: false\n"}, "reserved"),  # partners can never hide attribution
        ({"topics_yaml": "- id: a\n  title: T\n  facts: [f]\n  flags: [nope]\n"}, "topics.yaml"),
        ({"topics_yaml": "id: not-a-list\n"}, "topics.yaml"),
        ({"contacts_yaml": "- label: L\n  kind: pigeon\n  value: v\n"}, "contacts.yaml"),
        ({"faq_csv": "question\nQ?\n"}, "faq.csv"),
        ({"faq_csv": "question,answer\nQ?,\n"}, "faq.csv:2"),
        (
            {
                "topics_yaml": "- id: dup\n  title: T\n  facts: [f]\n",
                "faq_csv": "id,question,answer\ndup,Q?,A.\n",
            },
            "duplicate entry id 'dup'",
        ),
        ({"pack.yaml": MANIFEST + "source: {type: http, base_url: 'https://x.example'}\n"}, "files"),
    ],
)
def test_invalid_packs_fail_loudly(tmp_path: Path, files: dict[str, str], message: str) -> None:
    with pytest.raises(KnowledgeConfigError, match=message):
        FilePackSource.load(make_pack(tmp_path, **files))


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeConfigError, match=r"pack.yaml"):
        load_manifest(tmp_path)


@pytest.mark.live
async def test_real_embedder_finds_paraphrases() -> None:
    from embedding_cache import local_fast_embedder

    embedder = local_fast_embedder()  # local cache only: never downloads
    core = FilePackSource.load(PACKS / "kudiguard-core", embedder=embedder)
    text = "my money no reach the person but they debited me"
    hits = (await core.search(Query(text=text, vector=embedder.embed_query(text)))).hits
    assert hits[0].entry.id == "failed-transfer"
    joke = "tell me a joke"
    assert (await core.search(Query(text=joke, vector=embedder.embed_query(joke)))).hits == ()


def test_long_faq_answers_are_split_not_rejected(tmp_path: Path) -> None:
    long_answer = " ".join(f"Sentence number {i} explains one more detail." for i in range(60))
    pack = make_pack(tmp_path, faq_csv=f'question,answer\nLong?,"{long_answer}"\n')
    facts = FilePackSource.load(pack).entries[0].facts
    assert len(facts) > 1
    assert all(len(f) <= MAX_FACT_CHARS for f in facts)
    assert facts[0].startswith("Sentence number 0")


@pytest.mark.parametrize(
    "answer",
    ["Short.", "x" * 2500, "word " * 600, "A. " * 2000, "First sentence. " + "y" * 1500 + ". Last one."],
    ids=["short", "no-spaces", "no-sentences", "many-tiny", "giant-middle-sentence"],
)
def test_split_answer_properties(answer: str) -> None:
    facts = split_answer(answer)
    assert 1 <= len(facts) <= MAX_FACTS
    assert all(0 < len(f) <= MAX_FACT_CHARS for f in facts)
    joined = " ".join(facts).replace(" ", "")
    expected = " ".join(answer.split()).replace(" ", "")
    assert expected.startswith(joined)  # order preserved, nothing invented; only an over-cap tail may be cut
    if len(answer) <= MAX_FACT_CHARS * MAX_FACTS // 2:
        assert joined == expected  # nothing lost when well under the cap


def test_split_answer_empty() -> None:
    assert split_answer("   ") == ()


def test_oversized_pack_file_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kudiguard.knowledge.file_pack.MAX_FILE_BYTES", 10)
    with pytest.raises(KnowledgeConfigError, match="larger than"):
        FilePackSource.load(make_pack(tmp_path))


@pytest.mark.parametrize(
    ("kind", "value", "ok"),
    [
        ("phone", "+234 803 123 4567", True),
        ("phone", "call us", False),
        ("phone", "0700-CALL-CBN", True),
        ("phone", "0803 123 4567 ext 2", True),
        ("phone", "javascript:alert(1)", False),
        ("ussd", "*565*0#", True),
        ("ussd", "*565*0# now", False),
        ("web", "https://sec.gov.ng", True),
        ("web", "http://sec.gov.ng", False),
        ("email", "cpd@cbn.gov.ng", True),
        ("email", "cpd at cbn", False),
        ("note", "Any instruction text", True),
    ],
)
def test_contact_values_validated_by_kind(kind: str, value: str, ok: bool) -> None:
    data = {"label": "L", "kind": kind, "value": value}
    if ok:
        Contact.model_validate(data)
    else:
        with pytest.raises(ValueError, match="invalid value"):
            Contact.model_validate(data)


def test_split_answer_caps_fact_count_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    answer = " ".join(f"{i:04d}" + "z" * 990 + "." for i in range(MAX_FACTS + 3))
    with caplog.at_level("WARNING", logger="kudiguard.knowledge.file_pack"):
        facts = split_answer(answer)
    assert len(facts) == MAX_FACTS
    assert [f[:4] for f in facts] == [f"{i:04d}" for i in range(MAX_FACTS)]  # the head is kept, the tail cut
    assert "truncated" in caplog.text


def test_undecodable_pack_file_is_a_config_error(tmp_path: Path) -> None:
    pack = make_pack(tmp_path)
    (pack / "pack.yaml").write_bytes(b"id: acme\nname: \xff\xfe Acme\n")
    with pytest.raises(KnowledgeConfigError, match=r"pack\.yaml.*codec can't decode"):
        load_manifest(pack)
