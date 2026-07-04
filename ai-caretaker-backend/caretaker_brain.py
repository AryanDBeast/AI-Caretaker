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

CHECK_IN_SYSTEM_PROMPT = """
You are a caregiver assistant for a dementia patient.
The patient has not spoken to you in a while, so you are proactively
checking in on them.

Rules:
- Be warm, calm, and simple. One or two short sentences.
- Gently ask how they are doing or if they need anything.
- If their last statement is provided, you may softly reference it
  (e.g. following up on a meal, a walk, or how they were feeling).
- Do not alarm the patient or mention emergencies, caregivers, or monitoring.

Return ONLY JSON:
{
  "reply": "..."
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


def generate_check_in(last_statement, last_contact_display, hours_since):
    """Generate a proactive check-in message for the patient.

    last_statement: the patient's most recent statement text, or None
    last_contact_display: human-readable time of last contact, or None
    hours_since: float hours since last contact, or None if never spoken
    """
    if last_statement:
        context = (
            f"There has been no contact from the patient in about "
            f"{hours_since:.1f} hours.\n"
            f"Their last statement (at {last_contact_display}) was:\n"
            f'"{last_statement}"\n\n'
            "Write a gentle check-in message for them now."
        )
    else:
        context = (
            "There has been no contact from the patient yet today.\n"
            "Write a gentle check-in message for them now."
        )

    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {"role": "system", "content": CHECK_IN_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content
    data = json.loads(raw)
    reply = str(data.get("reply", "")).strip()
    if not reply:
        reply = "Hello, just checking in on you. How are you feeling right now?"
    return reply
