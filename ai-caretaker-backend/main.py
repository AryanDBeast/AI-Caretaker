import os
import smtplib
import threading
import time as time_module
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from caretaker_brain import process_input, generate_check_in
from database_service import DatabaseService
from fastapi.middleware.cors import CORSMiddleware


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# ---------------------------------------------------------------
# Check-in configuration
# ---------------------------------------------------------------
EASTERN = ZoneInfo("America/New_York")   # EST/EDT, handles daylight saving
CHECK_IN_AFTER = timedelta(hours=2)      # check in after 2h of no contact
RESPONSE_WINDOW = timedelta(minutes=15)  # how long the patient has to respond
ACTIVE_START_HOUR = 8                    # 8:00 AM Eastern
ACTIVE_END_HOUR = 20                     # 8:00 PM Eastern (exclusive)
LOOP_SECONDS = 60                        # scheduler wake-up interval
NO_RESPONSE_SEVERITY = 80                # severity score for the alert SMS

# In-memory per-patient scheduler state:
# patient_id -> {"awaiting_since": datetime|None, "last_check_in": datetime|None}
check_in_state = {}

# Check-in messages waiting to be spoken by the device:
# patient_id -> [message, ...]  (device polls GET /patients/{id}/checkin)
pending_speech = {}

state_lock = threading.Lock()

db = DatabaseService()


def now_eastern():
    return datetime.now(EASTERN)


def parse_ts(ts):
    """Normalize a DB timestamp (str or datetime) to an aware Eastern datetime."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=EASTERN)
    return dt.astimezone(EASTERN)


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


# ---------------------------------------------------------------
# Check-in scheduler
# ---------------------------------------------------------------
def handle_patient_check_in(sched_db: DatabaseService, patient_id: int, now: datetime):
    with state_lock:
        state = check_in_state.setdefault(
            patient_id, {"awaiting_since": None, "last_check_in": None}
        )
        awaiting_since = state["awaiting_since"]
        last_check_in = state["last_check_in"]

    last_user = sched_db.get_last_user_statement(patient_id)
    last_user_text = last_user[0] if last_user else None
    last_user_time = parse_ts(last_user[1]) if last_user else None

    # --- Case 1: we sent a check-in and are waiting for a response ---
    if awaiting_since is not None:
        if last_user_time is not None and last_user_time >= awaiting_since:
            # Patient responded — all good
            with state_lock:
                check_in_state[patient_id]["awaiting_since"] = None
            print(f"[CheckIn] Patient {patient_id} responded to check-in.")
            return

        if now - awaiting_since >= RESPONSE_WINDOW:
            # No response within the window — alert the caregiver, then continue
            patient_row = sched_db.get_patient_with_caregiver(patient_id)
            if patient_row:
                patient_name = f"{patient_row[1]} {patient_row[2]}"
                caregiver_phone = patient_row[7]
                minutes = int(RESPONSE_WINDOW.total_seconds() // 60)
                sms_body = build_emergency_sms(
                    "No Response To Check-In",
                    NO_RESPONSE_SEVERITY,
                    last_user_text or f"No response in {minutes} minutes",
                    patient_name,
                )
                send_verizon_sms_via_gmail(sms_body, caregiver_phone)

            # Log the missed check-in so it shows in the statement history
            sched_db.append_statement(
                patient_id,
                now.strftime("%Y-%m-%d %H:%M:%S"),
                "Assistant",
                "Patient did not respond to the check-in. Caregiver has been alerted.",
                None,
                None,
                "No Response To Check-In",
                NO_RESPONSE_SEVERITY,
            )

            with state_lock:
                check_in_state[patient_id]["awaiting_since"] = None
            print(f"[CheckIn] Patient {patient_id} did not respond. Caregiver alerted.")
        return

    # --- Case 2: not awaiting — decide whether a new check-in is due ---
    recent_contact = (
        last_user_time is not None and (now - last_user_time) < CHECK_IN_AFTER
    )
    recent_check_in = (
        last_check_in is not None and (now - last_check_in) < CHECK_IN_AFTER
    )
    if recent_contact or recent_check_in:
        return

    # Time to check in
    hours_since = None
    last_contact_display = None
    if last_user_time is not None:
        hours_since = (now - last_user_time).total_seconds() / 3600.0
        last_contact_display = last_user_time.strftime("%Y-%m-%d %I:%M %p ET")

    try:
        reply = generate_check_in(last_user_text, last_contact_display, hours_since)
    except Exception as e:
        print(f"[CheckIn] AI generation failed for patient {patient_id}: {e}")
        return

    sched_db.append_statement(
        patient_id,
        now.strftime("%Y-%m-%d %H:%M:%S"),
        "Assistant",
        reply,
    )

    # Queue the message for the device to speak
    with state_lock:
        pending_speech.setdefault(patient_id, []).append(reply)
        check_in_state[patient_id]["last_check_in"] = now
        check_in_state[patient_id]["awaiting_since"] = now

    print(f"[CheckIn] Queued check-in for patient {patient_id}: {reply}")


def run_check_in_cycle(sched_db: DatabaseService):
    now = now_eastern()

    # Only operate between 8 AM and 8 PM Eastern
    if not (ACTIVE_START_HOUR <= now.hour < ACTIVE_END_HOUR):
        # Clear pending windows and unspoken messages so nothing fires overnight
        with state_lock:
            for state in check_in_state.values():
                state["awaiting_since"] = None
            pending_speech.clear()
        return

    for row in sched_db.list_patients():
        patient_id = row[0]
        try:
            handle_patient_check_in(sched_db, patient_id, now)
        except Exception as e:
            print(f"[CheckIn] Error for patient {patient_id}: {e}")


def check_in_loop():
    # Separate DB connection — psycopg connections are not thread-safe,
    # so the scheduler thread must not share the API's connection.
    sched_db = DatabaseService()
    print("[CheckIn] Scheduler started (8 AM - 8 PM Eastern, 2h inactivity).")
    while True:
        try:
            run_check_in_cycle(sched_db)
        except Exception as e:
            print(f"[CheckIn] Cycle error: {e}")
        time_module.sleep(LOOP_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=check_in_loop, daemon=True)
    thread.start()
    yield


app = FastAPI(lifespan=lifespan)

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


@app.post("/conversation")
def conversation(req: ConversationRequest):
    now = now_eastern()

    # Any patient message counts as a response to a pending check-in,
    # and cancels any queued check-in that hasn't been spoken yet
    with state_lock:
        if req.patient_id in check_in_state:
            check_in_state[req.patient_id]["awaiting_since"] = None
        pending_speech.pop(req.patient_id, None)

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


@app.get("/patients/{patient_id}/checkin")
def get_pending_checkin(patient_id: int):
    """Polled by the device every few seconds.
    Returns a check-in message exactly once, then removes it from the queue."""
    with state_lock:
        queue = pending_speech.get(patient_id)
        if queue:
            message = queue.pop(0)
            if not queue:
                pending_speech.pop(patient_id, None)
            return {"message": message}
    return {"message": None}


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
