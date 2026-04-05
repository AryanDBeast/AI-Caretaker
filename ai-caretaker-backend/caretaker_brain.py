import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a caregiver assistant for a dementia patient.
Be calm, simple, and safe.

Return ONLY JSON:
{
  "reply": "...",
  "detections": {
    "emergency": {"flag": true/false, "type": "string", "score": 0-100},
    "activity": {"flag": true/false, "type": "string"},
    "medicine": {"flag": true/false, "time_of_day": "Morning/Evening/Unknown"}
  }
}
"""


def process_input(user_text: str):
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content
    return json.loads(raw)