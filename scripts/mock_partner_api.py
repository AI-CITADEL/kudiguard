"""Run the reference partner knowledge API over the sample Demo Bank pack.

DEMO_BANK_TOKEN=dev-token python scripts/mock_partner_api.py   # serves http://127.0.0.1:8900
"""

import os
from pathlib import Path

import uvicorn

from kudiguard.knowledge.reference_server import create_app

PACK = Path(__file__).resolve().parents[1] / "data" / "packs" / "demo-bank"

if __name__ == "__main__":
    uvicorn.run(create_app(PACK, token=os.environ.get("DEMO_BANK_TOKEN")), host="127.0.0.1", port=8900)
