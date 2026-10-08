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
