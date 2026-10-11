"""Rule probes from the E2 review (53bdeed): credential requests vs a bank's own warning, wrong-transfer cues,
impersonation claims vs questions. Each case: text, flag, expected."""

import pytest

from kudiguard.domain import Flag
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.scam_rules import assess

C, W, IMP = Flag.ASKS_CREDENTIALS, Flag.WRONG_TRANSFER, Flag.IMPERSONATION
PROBES = [
    # B1: a negated "do not share" that makes an exception is still a request
    ("Do not share your OTP with anyone except our agent who will call you", C, True),
    ("Never give your PIN to anyone. Give it to him only", C, True),
    ("Read the code to the officer when he calls", C, True),
    ("don't worry, just send the OTP", C, True),
    ("Dont panic, send the code", C, True),
    ("Do not delay, send your PIN", C, True),
    ("Never mind, share your BVN", C, True),
    ("Your OTP is 123456. Do not share it with anyone.", C, False),
    ("Do not share it with anyone", C, False),
    # B2: "the excess / the balance" needs a wrong-transfer cue
    ("I want to transfer the balance of my rent to my landlord", W, False),
    ("How do I pay back the balance of my loan?", W, False),
    ("I forgot to send back the excess change to the customer", W, False),
    ("He sent 50k to my account by mistake and wants me to send back the excess", W, True),
    ("I sent you too much by mistake, please send back the excess", W, True),
    # S1: named or headed customer care is a claim; a question is not
    ("GTBank customer service: your BVN needs urgent update", IMP, True),
    ("Customer care desk. Your account needs verification today", IMP, True),
    ("Access Bank Customer Support Unit wishes to inform you", IMP, True),
    ("Message from customer service: we noticed unusual activity", IMP, True),
    ("Dear customer, kindly contact our customer care line on WhatsApp to avoid restriction", IMP, True),
    ("Hello this is Moniepoint, we need to verify your account", IMP, True),
    ("I am from Zenith Bank, I need your details", IMP, True),
    ("Moniepoint here. Please confirm your details", IMP, True),
    ("Kindly send your account details for verification. Regards, GTBank", IMP, True),
    ("what is my bank's customer care number?", IMP, False),
    # second check (fresh genuine messages): no new false alarms
    ("Mummy, give the code to Daddy so he can open the gate", C, False),
    ("send me the code that came to your phone", C, True),
    ("Hi, this is First Bank fraud desk, a code has been sent to you, tell me the code to confirm", C, True),
    ("Dont panic, send the code to unlock your account", C, True),
    ("The cashier said she will refund the money to him tomorrow", W, False),
    ("He paid you by mistake, please refund the money to him today", W, True),
    ("Zenith Bank here in Ikeja is closed today, go to the Lekki branch", IMP, False),
    ("the agent told me to read the code to him, is that okay?", C, True),  # a live OTP scam being described
]


@pytest.mark.parametrize(("text", "flag", "expected"), PROBES)
def test_review_probes(registry: LanguageRegistry, text: str, flag: Flag, expected: bool) -> None:
    assert (flag in assess([text], registry.lexicon_for("en")).flags) is expected
