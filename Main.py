import os
import io
import time
import json
import calendar
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Deque, Tuple, List, Dict, Any

import speech_recognition as sr
import pygame
import gspread
from google.oauth2.service_account import Credentials
from google.cloud import texttospeech
from openai import OpenAI


# =========================
# OPENAI KEY (HARDCODED)
# =========================
OPENAI_API_KEY = ""


# =========================
# Configuration
# =========================

@dataclass(frozen=True)
class Config:
    caretaker_name: str
    phone_number: str  # reserved for SMS use
    openai_model: str

    google_creds_path: str
    spreadsheet_name: str

    # Sheet tab names
    sheet_medicine: str
    sheet_emergency: str
    sheet_activity: str
    sheet_convo: str
    sheet_metrics: str

    mic_device_index: int
    listen_timeout_sec: int
    phrase_time_limit_sec: int

    # context window for conversation memory
    context_window_minutes: int
    buffer_maxlen: int


def load_config() -> Config:
    caretaker_name = os.getenv("CARETAKER_NAME", "[INSERT CARETAKER NAME]")
    phone_number = os.getenv("CARETAKER_PHONE", "[INSERT PHONE NUMBER]")

    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

    google_creds_path = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/Users/aryandhingra/Documents/Research/bruce-433322-d3bdbe79e165.json"
    )

    listen_timeout = int(os.getenv("LISTEN_TIMEOUT_SEC", "25"))

    return Config(
        caretaker_name=caretaker_name,
        phone_number=phone_number,
        openai_model=openai_model,
        google_creds_path=google_creds_path,
        spreadsheet_name=os.getenv("SPREADSHEET_NAME", "Research Project"),
        sheet_medicine=os.getenv("SHEET_MEDICINE", "Medicine Log"),
        sheet_emergency=os.getenv("SHEET_EMERGENCY", "Emergency Log"),
        sheet_activity=os.getenv("SHEET_ACTIVITY", "Activity Log"),
        sheet_convo=os.getenv("SHEET_CONVO", "Convo History"),
        sheet_metrics=os.getenv("SHEET_METRICS", "Data"),
        mic_device_index=int(os.getenv("MIC_DEVICE_INDEX", "1")),
        listen_timeout_sec=listen_timeout,
        phrase_time_limit_sec=int(os.getenv("PHRASE_TIME_LIMIT_SEC", "20")),
        context_window_minutes=int(os.getenv("CONTEXT_WINDOW_MIN", "3")),
        buffer_maxlen=int(os.getenv("BUFFER_MAXLEN", "300")),
    )


# =========================
# Utilities
# =========================

def fmt_date(now: datetime) -> str:
    return now.strftime("%D")


def fmt_time(now: datetime) -> str:
    return now.strftime("%H:%M")


def time_context(now: datetime) -> str:
    day_name = calendar.day_name[now.weekday()]
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p").lstrip("0")
    return f"Current local date/time: {day_name}, {date_str} at {time_str}."


def infer_time_of_day(now: datetime) -> str:
    hhmm = int(now.strftime("%H%M"))
    if 600 <= hhmm < 1200:
        return "Morning"
    if 1900 <= hhmm < 2200:
        return "Evening"
    return "Unknown"


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# =========================
# Google Sheets Service
# =========================

class SheetsService:
    def __init__(self, cfg: Config):
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(cfg.google_creds_path, scopes=scopes)
        gc = gspread.authorize(creds)

        spread = gc.open(cfg.spreadsheet_name)

        self.medicine = spread.worksheet(cfg.sheet_medicine)
        self.emergency = spread.worksheet(cfg.sheet_emergency)
        self.activity = spread.worksheet(cfg.sheet_activity)
        self.convo = spread.worksheet(cfg.sheet_convo)
        self.metrics = spread.worksheet(cfg.sheet_metrics)

    def append_conversation_pair(self, now: datetime, user_text: str, assistant_text: str) -> None:
        d, t = fmt_date(now), fmt_time(now)
        self.convo.append_rows([
            [d, t, "User", user_text],
            [d, t, "Assistant", assistant_text],
        ])

    def append_medicine(self, now: datetime, time_of_day: str, patient_input: str) -> None:
        self.medicine.append_row([fmt_date(now), fmt_time(now), time_of_day, patient_input])

    def append_emergency(self, now: datetime, e_type: str, score: int, patient_input: str) -> None:
        self.emergency.append_row([fmt_date(now), fmt_time(now), e_type, str(score), patient_input])

    def append_activity(self, now: datetime, a_type: str, patient_input: str) -> None:
        self.activity.append_row([fmt_date(now), fmt_time(now), a_type, patient_input])

    def append_metrics(self, speech_s: float, response_s: float, update_s: float) -> None:
        self.metrics.append_row([f"{speech_s:.3f}", f"{response_s:.3f}", f"{update_s:.3f}"])


# =========================
# Background Sheets Logger (1 worker thread)
# =========================

class BackgroundLogger:
    """
    Runs ALL sheet writes in a single background thread so the main loop
    can move on (especially to TTS playback) without waiting.
    """
    def __init__(self, sheets: SheetsService):
        self.sheets = sheets
        self.q: Queue = Queue()
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

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

    def submit(self, fn, *args):
        self.q.put((fn, args))

    def stop(self):
        self._stop.set()


# =========================
# TTS + Audio Playback
# =========================

class SpeechService:
    def __init__(self):
        pygame.mixer.init()
        self.tts_client = texttospeech.TextToSpeechClient()

    def synthesize(self, text: str) -> Optional[io.BytesIO]:
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Wavenet-D")
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            resp = self.tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            return io.BytesIO(resp.audio_content)
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    def play(self, audio_stream: Optional[io.BytesIO]) -> None:
        if audio_stream is None:
            return
        audio_stream.seek(0)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


# =========================
# STT (Speech Recognition)
# =========================

class ListenService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.recognizer = sr.Recognizer()

        # dementia-friendly pause behavior
        self.recognizer.pause_threshold = 2
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def listen_once(self) -> Tuple[Optional[str], float]:
        try:
            with sr.Microphone(device_index=self.cfg.mic_device_index) as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.cfg.listen_timeout_sec,
                    phrase_time_limit=self.cfg.phrase_time_limit_sec
                )

            t0 = time.perf_counter()
            text = self.recognizer.recognize_google(audio)
            speech_time = time.perf_counter() - t0

            print(f"You said: {text}")
            print(f"Speech Time (sec): {speech_time:.3f}")
            return text, speech_time

        except sr.WaitTimeoutError:
            return None, 0.0
        except (AssertionError, OSError) as e:
            print(f"Mic error (device changed/disconnected): {e}")
            time.sleep(0.5)
            return None, 0.0
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None, 0.0
        except sr.RequestError:
            print("Sorry, there was an issue with the speech recognition service.")
            return None, 0.0


# =========================
# OpenAI + Conversation Memory + Structured Detection
# =========================

class CaretakerBrain:
    def __init__(self, cfg: Config, sheets: SheetsService):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.cfg = cfg
        self.sheets = sheets  # used only for priming

        self.system_prompt = {
            "role": "system",
            "content": (
                f"You are a caregiver for a dementia patient while their family member ({cfg.caretaker_name}) is away. "
                "Be calm, simple, supportive. Keep the patient safe and inside the home. "
                "Answer in 50 words or less unless you must ask a safety clarifying question.\n\n"
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
                "- Activity: walking, eating, showering, bathroom, sleep, exercise, chores, hydration, etc.\n"
                "- Medicine: taking meds, pill box, 'I took my medicine', 'did I take it?', etc.\n"
                "If uncertain about emergency, ask a clarifying question in reply AND set emergency.flag=true with a cautious score.\n"
            )
        }

        self.recent_buffer: Deque[Tuple[datetime, str, str]] = deque(maxlen=cfg.buffer_maxlen)
        self._prime_from_sheet()

    def _prime_from_sheet(self) -> None:
        try:
            rows = self.sheets.convo.get_all_values()
        except Exception as e:
            print(f"Warning: couldn't prime buffer from sheet: {e}")
            return

        now = datetime.now()
        for row in rows[1:]:
            if len(row) < 4:
                continue
            date_str, time_str, role, message = row[:4]
            try:
                ts = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %H:%M")
            except ValueError:
                continue
            if now - ts <= timedelta(minutes=self.cfg.context_window_minutes):
                self.recent_buffer.append((ts, role.lower(), message))

    def _build_messages(self, user_text: str, dynamic_system: str) -> List[Dict[str, str]]:
        now = datetime.now()
        msgs: List[Dict[str, str]] = [
            self.system_prompt,
            {"role": "system", "content": dynamic_system},
        ]

        for ts, role, message in self.recent_buffer:
            if now - ts <= timedelta(minutes=self.cfg.context_window_minutes):
                role_norm = role.strip().lower()
                r = "assistant" if role_norm == "assistant" else "user"
                msgs.append({"role": r, "content": message})

        msgs.append({"role": "user", "content": user_text})
        return msgs

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = text[first:last + 1]
            return json.loads(candidate)

        raise ValueError("Could not parse JSON from model output.")

    def respond_structured(self, now: datetime, user_text: str, dynamic_system: str) -> Tuple[Dict[str, Any], float]:
        """
        Sheets writes are queued in BackgroundLogger from main().
        """
        t0 = time.perf_counter()

        # request JSON mode if supported
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

        response_time = time.perf_counter() - t0

        raw = completion.choices[0].message.content or ""
        payload = self._parse_json(raw)

        reply = str(payload.get("reply", "")).strip()
        if not reply:
            reply = "I’m here with you. Can you tell me what you need right now?"
            payload["reply"] = reply

        # Update buffer for context
        self.recent_buffer.append((now, "user", user_text))
        self.recent_buffer.append((now, "assistant", reply))

        print(f"OpenAI Response Time (sec): {response_time:.3f}")
        return payload, response_time


# =========================
# Detection → Sheet Logging (queued)
# =========================

def queue_detection_logs(logger: BackgroundLogger, sheets: SheetsService, payload: Dict[str, Any], now: datetime, user_text: str) -> None:
    det = payload.get("detections", {}) or {}

    e = det.get("emergency", {}) or {}
    if bool(e.get("flag", False)):
        e_type = str(e.get("type", "Unknown")).strip() or "Unknown"
        score = max(0, min(100, safe_int(e.get("score", 0), default=0)))
        logger.submit(sheets.append_emergency, now, e_type, score, user_text)

    a = det.get("activity", {}) or {}
    if bool(a.get("flag", False)):
        a_type = str(a.get("type", "Unknown")).strip() or "Unknown"
        logger.submit(sheets.append_activity, now, a_type, user_text)

    m = det.get("medicine", {}) or {}
    if bool(m.get("flag", False)):
        tod = str(m.get("time_of_day", "Unknown")).strip() or "Unknown"
        if tod not in {"Morning", "Evening", "Unknown"}:
            tod = "Unknown"
        if tod == "Unknown":
            tod = infer_time_of_day(now)
        logger.submit(sheets.append_medicine, now, tod, user_text)


# =========================
# Main App
# =========================

def main():
    cfg = load_config()

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty. Paste your key into OPENAI_API_KEY at the top.")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.google_creds_path

    sheets = SheetsService(cfg)
    logger = BackgroundLogger(sheets)
    speaker = SpeechService()
    listener = ListenService(cfg)
    brain = CaretakerBrain(cfg, sheets)

    while True:
        now = datetime.now()

        user_text, speech_time = listener.listen_once()
        if not user_text:
            continue

        dynamic_system = (
            f"{time_context(now)} "
            "Safety rule: Do not instruct the patient to leave the home. "
            "If someone is at the door, ask for details and encourage staying inside. "
            "Remember: output ONLY JSON."
        )

        # 1) OpenAI call (must wait)
        payload, response_time = brain.respond_structured(now, user_text, dynamic_system)

        reply = str(payload.get("reply", "")).strip()
        if not reply:
            reply = "I’m here with you. What’s going on?"

        # 2) Queue ALL sheet writes immediately (non-blocking)
        t_log_start = time.perf_counter()

        logger.submit(sheets.append_conversation_pair, now, user_text, reply)
        queue_detection_logs(logger, sheets, payload, now, user_text)
        logger.submit(sheets.append_metrics, speech_time, response_time, 0.0)

        queue_overhead = time.perf_counter() - t_log_start
        print(f"Queued log overhead (sec): {queue_overhead:.3f}")

        # 3) Latency win #2: TTS synthesis overlaps with background logging
        # Synthesize first (network), while logger thread is writing to sheets.
        audio = speaker.synthesize(reply)

        # 4) Speak / play audio (logger continues in background)
        print(f"Assistant Reply: {reply}")
        speaker.play(audio)


if __name__ == "__main__":
    main()
