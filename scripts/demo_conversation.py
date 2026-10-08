"""End-to-end conversation against the real stack (N-ATLAS llama-server + NCAIR1 ASR), printed to the console.

    python scripts/demo_conversation.py [--url http://127.0.0.1:8081]

Runs the production ``Conversation`` with an in-memory store and a console messenger: onboarding, a Hausa
check, a voice note, the reply buttons and feedback. Used as the live smoke test and for the demo video script.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import time
from pathlib import Path

import httpx

from kudiguard.domain import InboundMessage, MessageKind
from kudiguard.engine import Engine
from kudiguard.knowledge.service import knowledge_base_from_packs
from kudiguard.languages import LanguageRegistry
from kudiguard.llm.openai_compat import OpenAICompatLLM
from kudiguard.pipeline import Conversation, Outbound
from kudiguard.privacy import UserHasher
from kudiguard.speech.audio import find_ffmpeg
from kudiguard.speech.natlas_asr import NatlasASR, ct2_loader
from kudiguard.storage.memory import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
USER = "+2348000000000"  # fictional


class ConsoleMessenger:
    def __init__(self) -> None:
        self.last_buttons: dict[str, str] = {}
        self.started = time.perf_counter()

    async def acknowledge(self, message: InboundMessage) -> None:
        return None

    async def send(self, user_id: str, message: Outbound) -> None:
        elapsed = time.perf_counter() - self.started
        print(f"\n  ⬅ [{elapsed:5.1f}s] " + message.text.replace("\n", "\n    "))
        for b in message.buttons:
            print(f"    [ {b.title} ]  ({b.id})")
            self.last_buttons[b.id.split(":")[0]] = b.id
        if message.options:
            print("    ▾ " + " · ".join(o.title for o in message.options))


async def main(url: str) -> None:
    registry = LanguageRegistry.load(ROOT / "data" / "languages")
    hasher = UserHasher(secrets.token_bytes(32))
    async with httpx.AsyncClient() as client:
        asr = NatlasASR(
            registry.asr_models(),
            loader=ct2_loader(
                cache_dir=Path(r"***REMOVED***\ncair\.cache"), whisper_languages=registry.asr_whisper_languages()
            ),
        )
        await asr.preload(["en"])
        engine = Engine(
            registry=registry,
            llm=OpenAICompatLLM(url, client=client),
            knowledge=knowledge_base_from_packs(ROOT / "data" / "packs", client=client),
            asr=asr,
            ffmpeg=find_ffmpeg(),
        )
        messenger = ConsoleMessenger()
        conversation = Conversation(
            engine=engine, store=MemoryStore(hasher), messenger=messenger, hasher=hasher
        )
        counter = 0

        async def say(
            text: str | None = None, *, button: str | None = None, audio: Path | None = None
        ) -> None:
            nonlocal counter
            counter += 1
            kind = MessageKind.BUTTON if button else MessageKind.AUDIO if audio else MessageKind.TEXT
            shown = text or button or (audio.name if audio else "")
            print(f"\n➡ {shown}")
            data = await asyncio.to_thread(audio.read_bytes) if audio else None
            messenger.started = time.perf_counter()
            await conversation.handle(
                InboundMessage("demo", f"m{counter}", USER, kind, text=text, button_id=button, audio=data)
            )

        await say("Sannu")
        await say(button="lang:ha")
        await say(
            "Wani ya kira ni ya ce shi ma'aikacin banki ne, asusuna za a rufe, ya ce in turo masa lambar OTP"
        )
        for action in ("steps", "warn", "helpful"):
            if action in messenger.last_buttons:
                await say(button=messenger.last_buttons[action])
        await say("English")
        await say(audio=ROOT / "tests" / "fixtures" / "voice_note_en_otp.ogg")
        await say("How can I save money every month?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    asyncio.run(main(parser.parse_args().url))
