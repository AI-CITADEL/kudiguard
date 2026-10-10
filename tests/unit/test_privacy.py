import pytest

from kudiguard.privacy import MIN_SECRET_BYTES, UserHasher, normalise_phone

SECRET = b"k" * MIN_SECRET_BYTES


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+2348031234567", "2348031234567"),
        ("2348031234567", "2348031234567"),
        ("+234 803 123 4567", "2348031234567"),
        ("08031234567", "2348031234567"),  # local Nigerian format -> E.164 digits
        ("+1 (202) 555-0147", "19175550147"),
        ("whatsapp:+2348031234567", "2348031234567"),
    ],
)
def test_normalise_phone(raw: str, expected: str) -> None:
    assert normalise_phone(raw) == expected


def test_same_person_same_hash_across_formats() -> None:
    hasher = UserHasher(SECRET)
    assert (
        hasher.user_id("+2348031234567")
        == hasher.user_id("08031234567")
        == hasher.user_id("234 803 123 4567")
    )


def test_hash_is_keyed_and_not_the_number() -> None:
    a = UserHasher(SECRET).user_id("+2348031234567")
    b = UserHasher(b"x" * MIN_SECRET_BYTES).user_id("+2348031234567")
    assert a != b  # without the secret, a phone number cannot be matched to its id
    assert "8031234567" not in a and len(a) == 32


def test_non_phone_ids_are_hashed_too() -> None:
    hasher = UserHasher(SECRET)
    assert hasher.user_id("web:session-abc") != hasher.user_id("web:session-abd")


@pytest.mark.parametrize("secret", [b"", b"short"])
def test_weak_secret_rejected(secret: bytes) -> None:
    with pytest.raises(ValueError, match="secret"):
        UserHasher(secret)


def test_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUDIGUARD_HASH_SECRET", "s" * MIN_SECRET_BYTES)
    assert UserHasher.from_env().user_id("08031234567")
    monkeypatch.delenv("KUDIGUARD_HASH_SECRET")
    with pytest.raises(ValueError, match="KUDIGUARD_HASH_SECRET"):
        UserHasher.from_env()


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("sess-a-12345678", "sess-b-12345678"),
        ("web:12345678", "12345678"),
        ("3f2a9c10-0000-0000-0000-000000000001", "9b7e1d20-0000-0000-0000-000000000001"),
    ],
)
def test_non_phone_ids_never_merge(a: str, b: str) -> None:
    hasher = UserHasher(SECRET)
    assert hasher.user_id(a) != hasher.user_id(b)


def test_foreign_local_numbers_are_not_nigerianised() -> None:
    assert normalise_phone("07911123456") == "07911123456"  # UK mobile stays as typed
    assert normalise_phone("09012345678") == "2349012345678"  # Nigerian 090x


def test_message_key_hides_the_number_and_is_stable() -> None:
    hasher = UserHasher(SECRET)
    wamid = "wamid.HBgNMjM0ODAzMTIzNDU2NxUCABIYFjNFQjBGNEY="  # base64 protobuf embedding 2348031234567
    key = hasher.message_key("whatsapp", wamid)
    assert key == hasher.message_key("whatsapp", wamid) and len(key) == 32
    assert "HBgN" not in key and key != hasher.message_key("web", wamid)


def test_short_numeric_ids_are_not_treated_as_phones() -> None:
    hasher = UserHasher(SECRET)
    assert hasher.user_id("12345678") != hasher.user_id("1234-5678")


# --- found by mutation testing (docs/SPIKES.md, G) ----------------------------------------------------------


def test_ids_are_stable_across_releases() -> None:
    """Golden values: changing a label or the format silently orphans every stored record (like a new secret)."""
    hasher = UserHasher(b"k" * 32)
    assert hasher.user_id("+2348031234567") == "785e6ef83a00ca7eb539e89b39c83ca1"
    assert hasher.user_id("web:session-abc") == "5074925464c6e5f8e7b9ebffc8c3726b"
    assert hasher.deletion_marker("abc") == "eeeadfee57cf90a171801d4ff40e2e5e"
    assert hasher.message_key("whatsapp", "wamid.X") == "3431d237bd303086b2d26b65feb837d8"
    assert hasher.review_id(42) == "R-0cfd0c7d6e"


def test_secret_must_be_at_least_32_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        UserHasher(b"k" * 31)
    UserHasher(b"k" * 32)
    UserHasher(b"k" * 64)
    for ok in ("s" * 32, "s" * 64):
        monkeypatch.setenv("KUDIGUARD_HASH_SECRET", ok)
        UserHasher.from_env()
    monkeypatch.setenv("KUDIGUARD_HASH_SECRET", "s" * 31)
    with pytest.raises(ValueError):
        UserHasher.from_env()


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("whatsapp:+2348031234567", "+2348031234567"),  # channel prefix ignored
        ("tel:08031234567", "2348031234567"),
        ("803-123-4567", "8031234567"),  # ten digits, ten characters: a phone number
        ("1-202-555-0147-12345", "+191755501471234 5"),  # twenty characters: still a phone number
    ],
)
def test_phone_formats_merge(a: str, b: str) -> None:
    hasher = UserHasher(SECRET)
    assert hasher.user_id(a) == hasher.user_id(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("803-123-456", "803 123 456"),  # nine digits: an id, hashed as typed
        ("----------", "((((()))))"),  # no digits at all
        ("08031234567abc", "08031234567"),  # trailing letters: an id
        ("234tel:8031234567", "2348031234567"),  # "tel:" only counts as a prefix
        ("1-202-555-0147-123456", "+1917555014712345 6"),  # over twenty characters
    ],
)
def test_non_phones_never_merge_with_phones(a: str, b: str) -> None:
    hasher = UserHasher(SECRET)
    assert hasher.user_id(a) != hasher.user_id(b)


def test_overlong_local_number_is_not_nigerianised() -> None:
    assert normalise_phone("080312345678") == "080312345678"  # twelve digits: not a Nigerian mobile
