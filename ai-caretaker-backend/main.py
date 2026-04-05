import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from caretaker_brain import process_input
from database_service import DatabaseService
from fastapi.middleware.cors import CORSMiddleware


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

db = DatabaseService()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConversationRequest(BaseModel):
    patient_id: int
    transcript: str


def build_emergency_sms(e_type: str, score: int, user_text: str, patient_name: str) -> str:
    e_type_clean = e_type.strip().title()
    msg = (
        "🚨Emergency Alert🚨\n\n"
        f"Patient: {patient_name}\n"
        f"Emergency Type: {e_type_clean}\n"
        f"Severity Score: {score}/100\n"
        f'Patient Input: "{user_text}"'
    )
    if len(msg) > 500:
        msg = msg[:497] + "..."
    return msg


def send_verizon_sms_via_gmail(body: str, caregiver_phone: str) -> bool:
    gmail_addr = os.getenv("GMAIL_ADDRESS", "").strip()
    gmail_app_pw = os.getenv("GMAIL_APP_PASSWORD", "").strip()

    if not gmail_addr or not gmail_app_pw:
        print("[EmailSMS] Missing env vars: GMAIL_ADDRESS, GMAIL_APP_PASSWORD")
        return False

    digits_only = "".join(ch for ch in caregiver_phone if ch.isdigit())
    if len(digits_only) != 10:
        print(f"[EmailSMS] Invalid caregiver phone number: {caregiver_phone}")
        return False

    to_addr = f"{digits_only}@vtext.com"

    body = (body or "").strip()
    if len(body) > 160:
        body = body[:157] + "..."

    msg = EmailMessage()
    msg["From"] = gmail_addr
    msg["To"] = to_addr
    msg["Subject"] = ""
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_addr, gmail_app_pw)
            smtp.send_message(msg)
        print(f"[EmailSMS] Sent to {to_addr}.")
        return True
    except Exception as e:
        print(f"[EmailSMS] Failed: {e}")
        return False


@app.post("/conversation")
def conversation(req: ConversationRequest):
    now = datetime.now()

    result = process_input(req.transcript)

    reply = str(result.get("reply", "")).strip()
    if not reply:
        reply = "I’m here to help. Please tell me what happened."

    detections = result.get("detections", {}) or {}

    emergency = detections.get("emergency", {}) or {}
    activity = detections.get("activity", {}) or {}
    medicine = detections.get("medicine", {}) or {}

    activity_type = activity.get("type") if activity.get("flag") else None
    medication_type = medicine.get("time_of_day") if medicine.get("flag") else None
    emergency_type = emergency.get("type") if emergency.get("flag") else None
    severity_score = emergency.get("score") if emergency.get("flag") else None

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    db.append_statement(
        req.patient_id,
        timestamp,
        "User",
        req.transcript,
        activity_type,
        medication_type,
        emergency_type,
        severity_score,
    )

    db.append_statement(
        req.patient_id,
        timestamp,
        "Assistant",
        reply,
    )

    patient_row = db.get_patient_with_caregiver(req.patient_id)

    if patient_row:
        patient_first_name = patient_row[1]
        patient_last_name = patient_row[2]
        caregiver_phone = patient_row[7]

        patient_name = f"{patient_first_name} {patient_last_name}"

        if emergency_type and severity_score is not None and severity_score >= 70:
            sms_body = build_emergency_sms(
                emergency_type,
                severity_score,
                req.transcript,
                patient_name,
            )
            send_verizon_sms_via_gmail(sms_body, caregiver_phone)

    return {
        "reply": reply,
        "detections": detections,
    }

@app.get("/patients")
def get_patients():
    rows = db.list_patients()
    return [
        {"id": r[0], "first_name": r[1], "last_name": r[2]}
        for r in rows
    ]


@app.get("/patients/{patient_id}/statements")
def get_statements(patient_id: int):
    return db.get_recent_statements(patient_id, 20)

@app.get("/caregivers")
def get_caregivers():
    rows = db.list_caregivers()
    return [
        {"id": r[0], "first_name": r[1], "last_name": r[2]}
        for r in rows
    ]