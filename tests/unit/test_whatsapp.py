import copy
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from kudiguard.channels.whatsapp import (
    BUTTONS_ONLY_BODY,
    MAX_BUTTON_BODY,
    MAX_LIST_BODY,
    MAX_TEXT,
    WhatsAppConfig,
    WhatsAppError,
    WhatsAppMessenger,
    build_payloads,
    parse_webhook,
    split_text,
    verify_signature,
    verify_subscription,
)
from kudiguard.domain import InboundMessage, MessageKind
from kudiguard.pipeline import Button, MenuOption, Outbound
from kudiguard.speech.base import AudioError, AudioProblem

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "whatsapp"
NOW = 1760000100.0
CONFIG = WhatsAppConfig(
    phone_number_id="PNID123", access_token="TOKEN-SECRET", app_secret="app-secret", verify_token="verify-me"
)
GRAPH = f"https://graph.facebook.com/{CONFIG.api_version}"
SEND_URL = f"{GRAPH}/PNID123/messages"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def sign(body: bytes, secret: str = "app-secret") -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# --- verification ----------------------------------------------------------------------------------------


def test_signature() -> None:
    body = b'{"object":"whatsapp_business_account"}'
    assert verify_signature(body, sign(body), "app-secret")
    assert verify_signature(body, sign(body).upper().replace("SHA256=", "sha256="), "app-secret")
    assert not verify_signature(body + b" ", sign(body), "app-secret")  # tampered body
    assert not verify_signature(body, sign(body, "other"), "app-secret")
    assert not verify_signature(body, None, "app-secret")
    assert not verify_signature(body, sign(body).removeprefix("sha256="), "app-secret")


def test_subscription_handshake() -> None:
    params = {"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "1158201444"}
    assert verify_subscription(params, "verify-me") == "1158201444"
    assert verify_subscription({**params, "hub.verify_token": "wrong"}, "verify-me") is None
    assert verify_subscription({**params, "hub.mode": "unsubscribe"}, "verify-me") is None
    assert verify_subscription({}, "verify-me") is None


def test_config_from_env_and_secret_hygiene() -> None:
    config = WhatsAppConfig.from_env(
        {"WHATSAPP_PHONE_NUMBER_ID": "1", "WHATSAPP_ACCESS_TOKEN": "tok", "WHATSAPP_APP_SECRET": "s",
         "WHATSAPP_VERIFY_TOKEN": "v", "WHATSAPP_API_VERSION": "v26.0"}
    )  # fmt: skip
    assert config.base == "https://graph.facebook.com/v26.0"
    assert "tok" not in repr(config) and "s'" not in repr(config)
    with pytest.raises(ValueError, match="missing"):
        WhatsAppConfig.from_env({})


# --- parsing ---------------------------------------------------------------------------------------------


def test_parse_mixed_payload_in_time_order() -> None:
    messages = parse_webhook(fixture("inbound_mixed.json"), phone_number_id="PNID123", now=NOW)
    assert [(m.message_id, m.kind) for m in messages] == [
        ("wamid.AUDIO1", MessageKind.AUDIO),  # sent before the text, delivered after it in the payload
        ("wamid.TEXT1", MessageKind.TEXT),
        ("wamid.BUTTON1", MessageKind.BUTTON),
        ("wamid.LIST1", MessageKind.BUTTON),
        ("wamid.TEMPLATE1", MessageKind.BUTTON),
        ("wamid.IMAGE1", MessageKind.UNSUPPORTED),
        ("wamid.UNSUPPORTED1", MessageKind.UNSUPPORTED),
    ]  # the reaction is ignored entirely
    audio, text, button, listed, template = messages[:5]
    assert (
        audio.media_id == "MEDIA123" and audio.audio_mime == "audio/ogg; codecs=opus" and audio.audio is None
    )
    assert text.text == "Someone called and asked for my OTP" and text.user_id == "2348000000001"
    assert (button.button_id, listed.button_id, template.button_id) == ("steps:12", "lang:ha", "helpful:12")
    assert all(m.channel == "whatsapp" for m in messages)


def test_statuses_produce_no_messages() -> None:
    assert parse_webhook(fixture("statuses_only.json"), now=NOW) == []


def test_other_phone_numbers_and_stale_messages_are_skipped() -> None:
    payload = fixture("inbound_mixed.json")
    assert parse_webhook(payload, phone_number_id="OTHER", now=NOW) == []
    assert parse_webhook(payload, phone_number_id="PNID123", now=NOW + 3 * 86400) == []


@pytest.mark.parametrize(
    "payload",
    [None, [], "x", {"object": "page"}, {"object": "whatsapp_business_account", "entry": "nope"},
     {"object": "whatsapp_business_account", "entry": [{"changes": [{"field": "messages", "value": None}]}]},
     {"object": "whatsapp_business_account", "entry": [None, {"changes": [None, {"field": "account_update"}]}]}],
)  # fmt: skip
def test_malformed_payloads_never_raise(payload: Any) -> None:
    assert parse_webhook(payload, now=NOW) == []


def test_malformed_messages_are_skipped_individually() -> None:
    payload = fixture("inbound_mixed.json")
    value = payload["entry"][0]["changes"][0]["value"]
    good = copy.deepcopy(value["messages"][0])
    value["messages"] = [
        None,
        {"id": "x"},  # no sender/type
        {**good, "id": "wamid.BADTS", "timestamp": "not-a-number"},  # bad timestamp: treated as now
        {**good, "id": "wamid.NOBODY", "type": "text", "text": "not a dict"},  # text without body
        {**good, "id": "wamid.AUDIONOID", "type": "audio", "audio": {}},
        {**good, "id": "wamid.IBAD", "type": "interactive", "interactive": {"type": "nfm_reply"}},
        {**good, "id": "wamid.NEWTYPE", "type": "hologram"},
        good,
    ]
    messages = {m.message_id: m.kind for m in parse_webhook(payload, now=NOW)}
    assert messages == {
        "wamid.BADTS": MessageKind.TEXT,
        "wamid.NOBODY": MessageKind.UNSUPPORTED,
        "wamid.AUDIONOID": MessageKind.UNSUPPORTED,
        "wamid.IBAD": MessageKind.UNSUPPORTED,
        "wamid.NEWTYPE": MessageKind.UNSUPPORTED,
        "wamid.TEXT1": MessageKind.TEXT,
    }


# --- outbound formatting ---------------------------------------------------------------------------------


def test_split_text_keeps_paragraphs() -> None:
    text = ("a " * 1250) + "\n\n" + ("b " * 1500)  # spaces after the paragraph break must not win
    parts = split_text(text)
    assert parts[0] == ("a " * 1250).strip() and all(len(p) <= MAX_TEXT for p in parts)
    assert all(len(p) <= MAX_TEXT for p in split_text("word " * 3000))
    assert split_text("x" * 5000) == ["x" * MAX_TEXT, "x" * 904]  # no boundary: hard cut
    assert split_text("   ") == []


def test_short_reply_with_buttons_is_one_interactive_message() -> None:
    message = Outbound(
        "Hello", (Button("steps:1", "What should I do?"), Button("report:1", "Where to report"))
    )
    [payload] = build_payloads("2348000000001", message)
    assert payload["to"] == "2348000000001" and payload["type"] == "interactive"
    interactive = payload["interactive"]
    assert interactive["body"]["text"] == "Hello"
    assert [b["reply"]["id"] for b in interactive["action"]["buttons"]] == ["steps:1", "report:1"]


def test_long_reply_with_buttons_is_split() -> None:
    long_text = "x" * (MAX_BUTTON_BODY + 10)
    buttons = tuple(Button(f"b:{i}", f"Button title number {i} too long") for i in range(5))
    payloads = build_payloads("1", Outbound(long_text, buttons))
    assert [p["type"] for p in payloads] == ["text", "interactive"]
    assert payloads[0]["text"]["body"] == long_text and payloads[0]["text"]["preview_url"] is False
    interactive = payloads[1]["interactive"]
    assert interactive["body"]["text"] == BUTTONS_ONLY_BODY
    titles = [b["reply"]["title"] for b in interactive["action"]["buttons"]]
    assert len(titles) == 3 and all(len(t) <= 20 for t in titles)


def test_list_menu_limits() -> None:
    options = tuple(MenuOption(f"lang:{i}", f"Language name number {i} long", "d" * 100) for i in range(12))
    [payload] = build_payloads(
        "1", Outbound("Which language?", menu_button="Choose language now please", options=options)
    )
    action = payload["interactive"]["action"]
    rows = action["sections"][0]["rows"]
    assert payload["interactive"]["type"] == "list" and len(action["button"]) <= 20
    assert len(rows) == 10 and all(len(r["title"]) <= 24 and len(r["description"]) <= 72 for r in rows)
    [plain] = build_payloads(
        "1", Outbound("x", menu_button=None, options=(MenuOption("lang:en", "English"),))
    )
    assert "description" not in plain["interactive"]["action"]["sections"][0]["rows"][0]


def test_plain_text_message() -> None:
    [payload] = build_payloads("1", Outbound("  hi  "))
    assert payload["text"] == {"body": "hi", "preview_url": False}


# --- messenger over the Graph API ------------------------------------------------------------------------


@pytest.fixture
async def messenger():  # type: ignore[no-untyped-def]
    async with httpx.AsyncClient() as client:
        yield WhatsAppMessenger(CONFIG, client=client, backoff_s=0.0)


def inbound(**kwargs: Any) -> InboundMessage:
    return InboundMessage("whatsapp", "wamid.IN1", "2348000000001", MessageKind.AUDIO, **kwargs)


@respx.mock
async def test_send_and_acknowledge(messenger: WhatsAppMessenger) -> None:
    route = respx.post(SEND_URL).mock(
        return_value=httpx.Response(200, json={"messages": [{"id": "wamid.OUT"}]})
    )
    await messenger.send("2348000000001", Outbound("Hello"))
    await messenger.acknowledge(inbound())
    sent, ack = (json.loads(call.request.content) for call in route.calls)
    assert sent["text"]["body"] == "Hello"
    assert ack == {"messaging_product": "whatsapp", "status": "read", "message_id": "wamid.IN1",
                   "typing_indicator": {"type": "text"}}  # fmt: skip
    assert route.calls[0].request.headers["authorization"] == "Bearer TOKEN-SECRET"


@respx.mock
async def test_transient_errors_are_retried(messenger: WhatsAppMessenger) -> None:
    route = respx.post(SEND_URL).mock(
        side_effect=[httpx.Response(503), httpx.ConnectError("boom"), httpx.Response(200, json={})]
    )
    await messenger.send("1", Outbound("x"))
    assert route.call_count == 3


@respx.mock
async def test_permanent_errors_raise_with_meta_code(messenger: WhatsAppMessenger) -> None:
    error = {"error": {"message": "Re-engagement message", "code": 131047}}
    route = respx.post(SEND_URL).mock(return_value=httpx.Response(400, json=error))
    with pytest.raises(WhatsAppError) as caught:
        await messenger.send("1", Outbound("x"))
    assert caught.value.code == 131047 and caught.value.status == 400 and route.call_count == 1
    assert "TOKEN-SECRET" not in str(caught.value)


@respx.mock
async def test_retries_are_bounded(messenger: WhatsAppMessenger) -> None:
    respx.post(SEND_URL).mock(return_value=httpx.Response(429, text="slow down"))
    with pytest.raises(WhatsAppError, match="429"):
        await messenger.send("1", Outbound("x"))
    respx.post(SEND_URL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(WhatsAppError, match="network"):
        await messenger.send("1", Outbound("x"))


@respx.mock
async def test_non_json_success_body_is_fine(messenger: WhatsAppMessenger) -> None:
    respx.post(SEND_URL).mock(return_value=httpx.Response(200, text="ok"))
    await messenger.send("1", Outbound("x"))


@respx.mock
async def test_fetch_media(messenger: WhatsAppMessenger) -> None:
    respx.get(f"{GRAPH}/MEDIA123").mock(
        return_value=httpx.Response(200, json={"url": "https://lookaside.fbsbx.com/m/1", "file_size": 5})
    )
    download = respx.get("https://lookaside.fbsbx.com/m/1").mock(
        side_effect=[httpx.Response(500), httpx.Response(200, content=b"OggS!")]
    )
    assert await messenger.fetch_media(inbound(media_id="MEDIA123")) == b"OggS!"
    assert download.calls[-1].request.headers["authorization"] == "Bearer TOKEN-SECRET"


@respx.mock
async def test_fetch_media_size_limits() -> None:
    async with httpx.AsyncClient() as client:
        await _check_size_limits(WhatsAppMessenger(CONFIG, client=client, max_media_bytes=4, backoff_s=0.0))


async def _check_size_limits(small: WhatsAppMessenger) -> None:
    respx.get(f"{GRAPH}/BIG").mock(
        return_value=httpx.Response(200, json={"url": "https://x.fbsbx.com/b", "file_size": 99})
    )
    with pytest.raises(AudioError) as declared:
        await small.fetch_media(inbound(media_id="BIG"))
    assert declared.value.problem is AudioProblem.TOO_LARGE
    respx.get(f"{GRAPH}/LIAR").mock(return_value=httpx.Response(200, json={"url": "https://x.fbsbx.com/l"}))
    respx.get("https://x.fbsbx.com/l").mock(return_value=httpx.Response(200, content=b"0123456789"))
    with pytest.raises(AudioError) as streamed:  # size not declared: enforced while streaming
        await small.fetch_media(inbound(media_id="LIAR"))
    assert streamed.value.problem is AudioProblem.TOO_LARGE


@respx.mock
async def test_fetch_media_failures(messenger: WhatsAppMessenger) -> None:
    with pytest.raises(AudioError):
        await messenger.fetch_media(inbound())  # no media id
    respx.get(f"{GRAPH}/NOURL").mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(WhatsAppError, match="no url"):
        await messenger.fetch_media(inbound(media_id="NOURL"))
    respx.get(f"{GRAPH}/GONE").mock(return_value=httpx.Response(200, json={"url": "https://x.fbsbx.com/g"}))
    respx.get("https://x.fbsbx.com/g").mock(
        return_value=httpx.Response(404, json={"error": {"message": "expired"}})
    )
    with pytest.raises(AudioError) as gone:  # looked up again once, then reported as unusable audio
        await messenger.fetch_media(inbound(media_id="GONE"))
    assert gone.value.problem is AudioProblem.UNDECODABLE
    respx.get(f"{GRAPH}/DOWN").mock(return_value=httpx.Response(200, json={"url": "https://x.fbsbx.com/d"}))
    respx.get("https://x.fbsbx.com/d").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(WhatsAppError, match="media download"):
        await messenger.fetch_media(inbound(media_id="DOWN"))


# --- review fixes (M8) -------------------------------------------------------------------------------------


def test_limits_count_utf16_units() -> None:
    from kudiguard.channels.whatsapp import units

    emoji = "😀" * 3000  # 6000 UTF-16 units
    parts = split_text(emoji)
    assert len(parts) == 2 and all(units(p) <= MAX_TEXT for p in parts)
    assert units("ẹ́") == 2 and units("😀") == 2  # combining mark + base; astral emoji


def test_list_body_allows_4096_and_buttons_1024() -> None:
    text = "x" * 2000
    [listed] = build_payloads("1", Outbound(text, options=(MenuOption("lang:en", "English"),)))
    assert listed["interactive"]["body"]["text"] == text and MAX_LIST_BODY == 4096
    assert [p["type"] for p in build_payloads("1", Outbound(text, (Button("a:1", "A"),)))] == [
        "text",
        "interactive",
    ]


def test_button_titles_stay_unique_after_clipping() -> None:
    buttons = (Button("a:1", "What should I do now?"), Button("a:2", "What should I do now!"))
    [payload] = build_payloads("1", Outbound("x", buttons))
    titles = [b["reply"]["title"] for b in payload["interactive"]["action"]["buttons"]]
    assert len(set(titles)) == 2 and all(len(t) <= 20 for t in titles)


def test_empty_text_with_buttons_still_has_a_body() -> None:
    [payload] = build_payloads("1", Outbound("", (Button("a:1", "A"),)))
    assert payload["interactive"]["body"]["text"] == BUTTONS_ONLY_BODY


def test_messages_without_metadata_are_skipped_when_filtering() -> None:
    payload = fixture("inbound_mixed.json")
    del payload["entry"][0]["changes"][0]["value"]["metadata"]
    assert parse_webhook(payload, phone_number_id="PNID123", now=NOW) == []
    assert len(parse_webhook(payload, now=NOW)) == 7  # no filter configured: accepted


@pytest.mark.parametrize("bad", [5, "x", {"a": 1}])
def test_non_list_containers_never_raise(bad: Any) -> None:
    payload = fixture("inbound_mixed.json")
    payload["entry"][0]["changes"][0]["value"]["messages"] = bad
    assert parse_webhook(payload, now=NOW) == []
    payload["entry"][0]["changes"] = bad
    assert parse_webhook(payload, now=NOW) == []


def test_absurd_timestamps_are_treated_as_now() -> None:
    payload = fixture("inbound_mixed.json")
    message = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    message["timestamp"] = float("inf")
    parsed = parse_webhook(payload, now=NOW)
    assert parsed[-1].message_id == "wamid.TEXT1"  # treated as "now": newest


def test_failed_delivery_statuses_are_reported() -> None:
    from kudiguard.channels.whatsapp import parse_failed_deliveries

    payload = fixture("statuses_only.json")
    statuses = payload["entry"][0]["changes"][0]["value"]["statuses"]
    statuses.append({"id": "wamid.OUT2", "status": "failed", "recipient_id": "1",
                     "errors": [{"code": 131047, "title": "Re-engagement message"}]})  # fmt: skip
    statuses.append({"id": "wamid.OUT3", "status": "failed", "errors": "weird"})
    failures = parse_failed_deliveries(payload, phone_number_id="PNID123")
    assert [(f.message_id, f.code) for f in failures] == [("wamid.OUT2", 131047), ("wamid.OUT3", None)]
    assert parse_failed_deliveries({"object": "whatsapp_business_account", "entry": [{"changes": 5}]}) == []


@respx.mock
async def test_sends_are_not_retried_when_meta_may_have_accepted_them(messenger: WhatsAppMessenger) -> None:
    route = respx.post(SEND_URL).mock(side_effect=httpx.ReadTimeout("no answer"))
    with pytest.raises(WhatsAppError):
        await messenger.send("1", Outbound("x"))
    assert route.call_count == 1  # a retry could deliver the reply twice
    route.mock(return_value=httpx.Response(500), side_effect=None)
    with pytest.raises(WhatsAppError):
        await messenger.send("1", Outbound("x"))
    assert route.call_count == 2  # one more call: the 500 was not retried either


@respx.mock
async def test_retry_decisions_follow_meta_error_codes(messenger: WhatsAppMessenger) -> None:
    throttled = {"error": {"message": "Rate limit hit", "code": 130429}}
    route = respx.post(SEND_URL).mock(
        side_effect=[
            httpx.Response(400, json=throttled, headers={"retry-after": "0"}),
            httpx.Response(200, json={}),
        ]
    )
    await messenger.send("1", Outbound("x"))
    assert route.call_count == 2  # HTTP 400 but a retryable code
    route.mock(return_value=httpx.Response(503, json={"error": {"code": 131026}}), side_effect=None)
    with pytest.raises(WhatsAppError):
        await messenger.send("1", Outbound("x"))
    assert route.call_count == 3  # undeliverable: final even though HTTP 503 (one call, no retry)


@respx.mock
async def test_expired_token_is_logged_loudly(
    messenger: WhatsAppMessenger, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post(SEND_URL).mock(
        return_value=httpx.Response(401, json={"error": {"message": "expired", "code": 190}})
    )
    with pytest.raises(WhatsAppError):
        await messenger.send("1", Outbound("x"))
    assert any(r.levelname == "ERROR" and "WHATSAPP_ACCESS_TOKEN" in r.getMessage() for r in caplog.records)


@respx.mock
async def test_media_url_must_be_a_meta_https_host(messenger: WhatsAppMessenger) -> None:
    for url in ("http://lookaside.fbsbx.com/m", "https://evil.example/m", "https://fbsbx.com.evil.example/m"):
        respx.get(f"{GRAPH}/M").mock(return_value=httpx.Response(200, json={"url": url}))
        with pytest.raises(WhatsAppError, match="unexpected host"):
            await messenger.fetch_media(inbound(media_id="M"))
    lookup = respx.calls[-1].request
    assert lookup.url.params["phone_number_id"] == "PNID123"


@respx.mock
async def test_unknown_media_id_is_unusable_audio(messenger: WhatsAppMessenger) -> None:
    respx.get(f"{GRAPH}/NOPE").mock(
        return_value=httpx.Response(404, json={"error": {"message": "no", "code": 100}})
    )
    with pytest.raises(AudioError):
        await messenger.fetch_media(inbound(media_id="NOPE"))


@respx.mock
async def test_media_url_expiring_once_is_looked_up_again(messenger: WhatsAppMessenger) -> None:
    lookup = respx.get(f"{GRAPH}/AGAIN").mock(
        side_effect=[
            httpx.Response(200, json={"url": "https://lookaside.fbsbx.com/old"}),
            httpx.Response(200, json={"url": "https://lookaside.fbsbx.com/new"}),
        ]
    )
    respx.get("https://lookaside.fbsbx.com/old").mock(return_value=httpx.Response(404))
    respx.get("https://lookaside.fbsbx.com/new").mock(return_value=httpx.Response(200, content=b"OggS"))
    assert await messenger.fetch_media(inbound(media_id="AGAIN")) == b"OggS"
    assert lookup.call_count == 2


# --- verification round (M8) ------------------------------------------------------------------------------


@respx.mock
@pytest.mark.parametrize("header", ["nan", "-5", "inf", "Wed, 21 Oct 2026 07:28:00 GMT"])
async def test_odd_retry_after_headers_do_not_break_retries(
    messenger: WhatsAppMessenger, header: str
) -> None:
    route = respx.post(SEND_URL).mock(
        side_effect=[httpx.Response(429, headers={"retry-after": header}), httpx.Response(200, json={})]
    )
    if header == "inf":  # non-finite: falls back to our backoff (0 in this test) instead of waiting forever
        route.side_effect = [httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200, json={})]
    await messenger.send("1", Outbound("x"))
    assert route.call_count == 2


def test_retry_after_parsing() -> None:
    from kudiguard.channels.whatsapp import _retry_after

    def parsed(value: str) -> float | None:
        return _retry_after(httpx.Response(429, headers={"retry-after": value}))

    assert (parsed("2"), parsed("-5"), parsed("nan"), parsed("inf"), parsed("soon")) == (
        2.0, 0.0, None, None, None  # non-finite values fall back to our own backoff
    )  # fmt: skip
    assert _retry_after(None) is None


@respx.mock
async def test_ambiguous_131000_is_not_retried_for_sends(messenger: WhatsAppMessenger) -> None:
    route = respx.post(SEND_URL).mock(return_value=httpx.Response(500, json={"error": {"code": 131000}}))
    with pytest.raises(WhatsAppError):
        await messenger.send("1", Outbound("x"))
    assert route.call_count == 1  # may already have been processed: a retry could double-send
    lookup = respx.get(f"{GRAPH}/RETRYME").mock(
        side_effect=[
            httpx.Response(500, json={"error": {"code": 131000}}),
            httpx.Response(200, json={"url": "https://lookaside.fbsbx.com/r"}),
        ]
    )
    respx.get("https://lookaside.fbsbx.com/r").mock(return_value=httpx.Response(200, content=b"OggS"))
    assert await messenger.fetch_media(inbound(media_id="RETRYME")) == b"OggS"  # GET: safe to retry
    assert lookup.call_count == 2


def test_split_text_rejects_impossible_limits() -> None:
    with pytest.raises(ValueError):
        split_text("😀abc", 1)
    assert split_text("😀😀😀", 2) == ["😀", "😀", "😀"]


def test_parse_helpers_never_raise_even_if_internals_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    from kudiguard.channels import whatsapp

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(whatsapp, "_parse_message", boom)
    assert parse_webhook(fixture("inbound_mixed.json"), now=NOW) == []
    payload = fixture("statuses_only.json")
    payload["entry"][0]["changes"][0]["value"]["statuses"][0]["status"] = "failed"
    monkeypatch.setattr(whatsapp, "DeliveryFailure", boom)
    assert whatsapp.parse_failed_deliveries(payload) == []


def test_messages_without_type_or_sender_are_skipped() -> None:
    payload = fixture("inbound_mixed.json")
    value = payload["entry"][0]["changes"][0]["value"]
    good = value["messages"][0]
    value["messages"] = [{**good, "type": None}, {**good, "from": ""}, {**good, "timestamp": "-3"}]
    assert [m.message_id for m in parse_webhook(payload, now=NOW)] == ["wamid.TEXT1"]  # -3: treated as now


@respx.mock
async def test_media_lookup_server_errors_propagate(messenger: WhatsAppMessenger) -> None:
    respx.get(f"{GRAPH}/BROKEN").mock(return_value=httpx.Response(401, json={"error": {"code": 190}}))
    with pytest.raises(WhatsAppError):
        await messenger.fetch_media(inbound(media_id="BROKEN"))


@respx.mock
async def test_network_failures_on_get_are_retried_then_raised(messenger: WhatsAppMessenger) -> None:
    lookup = respx.get(f"{GRAPH}/FLAKY").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(WhatsAppError, match="network"):
        await messenger.fetch_media(inbound(media_id="FLAKY"))
    assert lookup.call_count == 3


def test_error_bodies_of_any_shape() -> None:
    from kudiguard.channels.whatsapp import _error_of

    assert _error_of(httpx.Response(400, json={"error": "flat string"})).code is None
    assert _error_of(httpx.Response(400, json=["list"])).code is None
    assert _error_of(httpx.Response(400, text="<html>")).status == 400


def test_prefix_within_counts_utf16_units() -> None:
    from kudiguard.channels.whatsapp import _prefix_within

    assert _prefix_within("ab😀c", 3) == 2  # the emoji (2 units) would cross the limit
    assert _prefix_within("abc", 10) == 3  # everything fits
