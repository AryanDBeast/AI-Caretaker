import os
import io
import time
import json
import calendar
import threading
import smtplib
from queue import Queue, Empty
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from typing import Optional, Deque, Tuple, List, Dict, Any
from pathlib import Path
from email.message import EmailMessage

import speech_recognition as sr
import pygame
from google.cloud import texttospeech
from openai import OpenAI
from dotenv import load_dotenv

from database_service import DatabaseService

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


@dataclass(frozen=True)
class Config:
    openai_model: str
    openai_api_key: str
    mic_device_index: int
    listen_timeout_sec: int
    phrase_time_limit_sec: int
    context_window_minutes: int
    buffer_maxlen: int

@dataclass(frozen=True)
class ActivePatient:
    patient_id: int
    patient_first_name: str
    patient_last_name: str
    dob: str
    caregiver_id: int
    caregiver_first_name: str
    caregiver_last_name: str
    caregiver_phone: str


# Loads environment variables into a strongly typed config object.
def load_config() -> Config:
    return Config(
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        mic_device_index=int(os.getenv("MIC_DEVICE_INDEX", "1")),
        listen_timeout_sec=int(os.getenv("LISTEN_TIMEOUT_SEC", "25")),
        phrase_time_limit_sec=int(os.getenv("PHRASE_TIME_LIMIT_SEC", "20")),
        context_window_minutes=int(os.getenv("CONTEXT_WINDOW_MIN", "3")),
        buffer_maxlen=int(os.getenv("BUFFER_MAXLEN", "300")),
    )

# Returns a human-readable time context string for the model.
def time_context(now: datetime) -> str:
    day_name = calendar.day_name[now.weekday()]
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p").lstrip("0")
    return f"Current local date/time: {day_name}, {date_str} at {time_str}."


# Infers whether the current time is morning, evening, or unknown.
def infer_time_of_day(now: datetime) -> str:
    hhmm = int(now.strftime("%H%M"))
    if 600 <= hhmm < 1200:
        return "Morning"
    if 1900 <= hhmm < 2200:
        return "Evening"
    return "Unknown"


# Safely converts a value to int with a default fallback.
def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# Formats a datetime into a SQLite-friendly timestamp string.
def to_db_timestamp(now: datetime) -> str:
    return now.strftime("%Y-%m-%d %H:%M:%S")


# Builds the outgoing emergency alert SMS message.
def build_emergency_sms(e_type: str, score: int, user_text: str) -> str:
    e_type_clean = e_type.strip().title()
    msg = (
        "🚨Emergency Alert🚨\n\n"
        f"Emergency Type: {e_type_clean}\n"
        f"Severity Score: {score}/100\n"
        f'Patient Input: "{user_text}"'
    )
    if len(msg) > 500:
        msg = msg[:497] + "..."
    return msg


# Sends an emergency SMS through Gmail to a Verizon SMS gateway.
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

def choose_patient(db: DatabaseService) -> ActivePatient:
    patients = db.list_patients()

    print("\nSelect Patient\n")
    for p in patients:
        print(f"{p[0]}. {p[1]} {p[2]}")

    while True:
        try:
            choice = int(input("\nEnter patient number: "))
            row = db.get_patient_with_caregiver(choice)
            if row:
                return ActivePatient(
                    patient_id=row[0],
                    patient_first_name=row[1],
                    patient_last_name=row[2],
                    dob=row[3],
                    caregiver_id=row[4],
                    caregiver_first_name=row[5],
                    caregiver_last_name=row[6],
                    caregiver_phone=row[7],
                )
        except Exception:
            pass

        print("Invalid selection. Try again.")


class BackgroundLogger:
    # Starts a background worker that executes queued database writes.
    def __init__(self, db: DatabaseService):
        self.db = db
        self.q: Queue = Queue()
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    # Continuously processes queued logging tasks.
    def _run(self):
        while not self._stop.is_set():
            try:
                fn, args = self.q.get(timeout=0.2)
            except Empty:
                continue
            try:
                fn(*args)
            except Exception as e:
                print(f"[BackgroundLogger] write failed: {e}")
            finally:
                self.q.task_done()

    # Adds a database write task to the queue.
    def submit(self, fn, *args):
        self.q.put((fn, args))

    # Stops the background worker loop.
    def stop(self):
        self._stop.set()


class SpeechService:
    # Initializes audio playback and Google Cloud TTS.
    def __init__(self):
        pygame.mixer.init()
        self.tts_client = texttospeech.TextToSpeechClient()

    # Converts text into spoken MP3 audio in memory.
    def synthesize(self, text: str) -> Optional[io.BytesIO]:
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Wavenet-D"
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            resp = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            return io.BytesIO(resp.audio_content)
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    # Plays synthesized audio through pygame.
    def play(self, audio_stream: Optional[io.BytesIO]) -> None:
        if audio_stream is None:
            return
        audio_stream.seek(0)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


class ListenService:
    # Configures the microphone and speech recognizer.
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 2
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    # Listens once from the microphone and returns transcript plus STT time.
    def listen_once(self) -> Tuple[Optional[str], float]:
        try:
            with sr.Microphone(device_index=self.cfg.mic_device_index) as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.cfg.listen_timeout_sec,
                    phrase_time_limit=self.cfg.phrase_time_limit_sec
                )

            text = self.recognizer.recognize_google(audio)

            print(f"You said: {text}")
            return text

        except sr.WaitTimeoutError:
            return None
        except (AssertionError, OSError) as e:
            print(f"Mic error (device changed/disconnected): {e}")
            time.sleep(0.5)
            return None
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, there was an issue with the speech recognition service.")
            return None


class CaretakerBrain:
    # Sets up the LLM client, system prompt, and short-term memory buffer.
    def __init__(self, cfg: Config, active_patient: ActivePatient):
        self.client = OpenAI(api_key=cfg.openai_api_key)
        self.cfg = cfg
        self.active_patient = active_patient
        self.recent_buffer: Deque[Tuple[datetime, str, str]] = deque(maxlen=cfg.buffer_maxlen)
        self.system_prompt = {
            "role": "system",
            "content": (
                f"You are a caregiver assistant speaking with dementia patient "
                f"{active_patient.patient_first_name} {active_patient.patient_last_name}. "
                f"The patient's caregiver is "
                f"{active_patient.caregiver_first_name} {active_patient.caregiver_last_name}. "
                f"Be calm, simple, warm, and safe. Facilitate conversation in a friendly manner. "
                f"Answer in 50 words or less unless you must ask a safety clarifying question.\n\n"
                "CRITICAL OUTPUT RULE:\n"
                "Return ONLY valid JSON (no markdown, no extra text) matching this structure:\n"
                "{\n"
                '  "reply": "what you will say to the patient",\n'
                '  "detections": {\n'
                '    "emergency": {"flag": true/false, "type": "string", "score": 0-100},\n'
                '    "activity": {"flag": true/false, "type": "string"},\n'
                '    "medicine": {"flag": true/false, "time_of_day": "Morning/Evening/Unknown"}\n'
                "  }\n"
                "}\n\n"
                "Detection rules:\n"
                "- Emergency: falls, injury, fire, break-in/stranger, wandering/leaving home, severe confusion, chest pain, 'help', etc.\n"
                "- Activity: walking, eating, showering, bathroom, sleep, exercise, chores, etc.\n"
                "- Medicine: taking meds, pill box, 'I took my medicine', 'did I take it?', etc.\n"
                "If uncertain about emergency, ask a clarifying question in reply AND set emergency.flag=true with a cautious score.\n"
            )
        }

    # Builds the message list for the LLM using recent conversational context.
    def _build_messages(self, user_text: str, dynamic_system: str) -> List[Dict[str, str]]:
        now = datetime.now()
        msgs: List[Dict[str, str]] = [
            self.system_prompt,
            {"role": "system", "content": dynamic_system},
        ]

        for ts, role, message in self.recent_buffer:
            if (now - ts).total_seconds() <= self.cfg.context_window_minutes * 60:
                msgs.append({
                    "role": "assistant" if role == "assistant" else "user",
                    "content": message
                })

        msgs.append({"role": "user", "content": user_text})
        return msgs

    # Parses the model output into a Python dictionary.
    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return json.loads(text[first:last + 1])

        raise ValueError("Could not parse JSON from model output.")

    # Sends the user input to the LLM and returns structured JSON plus latency.
    def respond_structured(self, now: datetime, user_text: str, dynamic_system: str) -> Tuple[Dict[str, Any], float]:

        try:
            completion = self.client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=self._build_messages(user_text, dynamic_system),
                response_format={"type": "json_object"},
            )
        except TypeError:
            completion = self.client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=self._build_messages(user_text, dynamic_system),
            )

        raw = completion.choices[0].message.content or ""
        payload = self._parse_json(raw)

        reply = str(payload.get("reply", "")).strip()
        if not reply:
            reply = "I’m here with you. Can you tell me what you need right now?"
            payload["reply"] = reply

        self.recent_buffer.append((now, "user", user_text))
        self.recent_buffer.append((now, "assistant", reply))

        return payload


# Queues all detection-based database log writes for the current turn.
def queue_detection_logs(
    logger: BackgroundLogger,
    db: DatabaseService,
    patient_id: int,
    payload: Dict[str, Any],
    now: datetime,
    user_text: str
) -> None:
    ts = to_db_timestamp(now)
    det = payload.get("detections", {}) or {}

    e = det.get("emergency", {}) or {}
    if bool(e.get("flag", False)):
        e_type = str(e.get("type", "Unknown")).strip() or "Unknown"
        score = max(0, min(100, safe_int(e.get("score", 0), default=0)))
        statement = f"[{e_type}] {user_text}"
        logger.submit(db.append_emergency, patient_id, ts, score, statement)

    a = det.get("activity", {}) or {}
    if bool(a.get("flag", False)):
        a_type = str(a.get("type", "Unknown")).strip() or "Unknown"
        logger.submit(db.append_activity, patient_id, ts, a_type, user_text)

    m = det.get("medicine", {}) or {}
    if bool(m.get("flag", False)):
        medication_type = str(m.get("time_of_day", "Unknown")).strip() or "Unknown"
        if medication_type not in {"Morning", "Evening", "Unknown"}:
            medication_type = "Unknown"
        if medication_type == "Unknown":
            medication_type = infer_time_of_day(now)
        logger.submit(db.append_medication, patient_id, ts, medication_type, user_text)


# Runs the caretaker application loop.
def main():
    cfg = load_config()

    missing = []
    if not cfg.openai_api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        raise RuntimeError("Missing required env vars: " + ", ".join(missing))

    db = DatabaseService()
    active_patient = choose_patient(db)

    print(f"\nLoaded Patient: {active_patient.patient_first_name} {active_patient.patient_last_name}")
    print(f"Caregiver: {active_patient.caregiver_first_name} {active_patient.caregiver_last_name}")
    print(f"Caregiver Phone: {active_patient.caregiver_phone}\n")

    logger = BackgroundLogger(db)
    brain = CaretakerBrain(cfg, active_patient)
    speaker = SpeechService()
    listener = ListenService(cfg)

    while True:
        now = datetime.now()
        ts = to_db_timestamp(now)

        user_text = listener.listen_once()
        if not user_text:
            continue

        dynamic_system = (
            f"{time_context(now)} "
            f"You are currently assisting patient {active_patient.patient_first_name} "
            f"{active_patient.patient_last_name}. "
            f"The caregiver for this patient is {active_patient.caregiver_first_name} "
            f"{active_patient.caregiver_last_name}. "
            "Safety rule: Do not instruct the patient to leave the home. "
            "If someone is at the door, ask for details and encourage staying inside. "
            "Remember: output ONLY JSON. "
            "If it is morning or evening and medicine has not been mentioned recently, "
            "you MAY gently remind the patient in one short, natural sentence. "
            "Do not repeat reminders every turn."
        )

        payload = brain.respond_structured(now, user_text, dynamic_system)

        reply = str(payload.get("reply", "")).strip()
        if not reply:
            reply = "I’m here with you. What’s going on?"

        det = payload.get("detections", {}) or {}
        e = det.get("emergency", {}) or {}
        if bool(e.get("flag", False)):
            e_type = str(e.get("type", "Unknown")).strip() or "Unknown"
            score = max(0, min(100, safe_int(e.get("score", 0), default=0)))
            sms_body = build_emergency_sms(e_type, score, user_text)
            send_verizon_sms_via_gmail(sms_body, active_patient.caregiver_phone)

        logger.submit(db.append_conversation_pair, active_patient.patient_id, ts, user_text, reply)
        queue_detection_logs(logger, db, active_patient.patient_id, payload, now, user_text)

        audio = speaker.synthesize(reply)

        print(f"Assistant Reply: {reply}")
        speaker.play(audio)


if __name__ == "__main__":
    main()
