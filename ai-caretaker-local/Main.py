import io
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pygame
import requests
import speech_recognition as sr
from dotenv import load_dotenv
from google.cloud import texttospeech

from database_service import DatabaseService

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


@dataclass(frozen=True)
class Config:
    mic_device_index: int
    listen_timeout_sec: int
    phrase_time_limit_sec: int
    backend_url: str


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


def load_config() -> Config:
    import os
    return Config(
        mic_device_index=int(os.getenv("MIC_DEVICE_INDEX", "1")),
        listen_timeout_sec=int(os.getenv("LISTEN_TIMEOUT_SEC", "25")),
        phrase_time_limit_sec=int(os.getenv("PHRASE_TIME_LIMIT_SEC", "20")),
        backend_url=os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/"),
    )


def choose_patient(db: DatabaseService) -> ActivePatient:
    patients = db.list_patients()

    print("\nSelect Patient\n")
    for patient in patients:
        print(f"{patient[0]}. {patient[1]} {patient[2]}")

    while True:
        try:
            choice = int(input("\nEnter patient number: "))
            row = db.get_patient_with_caregiver(choice)
            if row:
                return ActivePatient(
                    patient_id=row[0],
                    patient_first_name=row[1],
                    patient_last_name=row[2],
                    dob=str(row[3]),
                    caregiver_id=row[4],
                    caregiver_first_name=row[5],
                    caregiver_last_name=row[6],
                    caregiver_phone=row[7],
                )
        except Exception:
            pass

        print("Invalid selection. Try again.")


def send_to_backend(transcript: str, patient_id: int, backend_url: str) -> dict:
    url = f"{backend_url}/conversation"
    payload = {
        "patient_id": patient_id,
        "transcript": transcript,
    }
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


class SpeechService:
    def __init__(self):
        pygame.mixer.init()
        self.tts_client = texttospeech.TextToSpeechClient()

    def synthesize(self, text: str) -> Optional[io.BytesIO]:
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Wavenet-D",
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            return io.BytesIO(response.audio_content)
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


class ListenService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 2
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def listen_once(self) -> Optional[str]:
        try:
            with sr.Microphone(device_index=self.cfg.mic_device_index) as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.cfg.listen_timeout_sec,
                    phrase_time_limit=self.cfg.phrase_time_limit_sec,
                )

            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text

        except sr.WaitTimeoutError:
            return None
        except (AssertionError, OSError) as e:
            print(f"Mic error: {e}")
            time.sleep(0.5)
            return None
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, there was an issue with the speech recognition service.")
            return None


def main():
    cfg = load_config()

    db = DatabaseService()
    active_patient = choose_patient(db)

    print(f"\nLoaded Patient: {active_patient.patient_first_name} {active_patient.patient_last_name}")
    print(f"Caregiver: {active_patient.caregiver_first_name} {active_patient.caregiver_last_name}")
    print(f"Caregiver Phone: {active_patient.caregiver_phone}\n")

    speaker = SpeechService()
    listener = ListenService(cfg)

    try:
        while True:
            user_text = listener.listen_once()
            if not user_text:
                continue

            payload = send_to_backend(
                transcript=user_text,
                patient_id=active_patient.patient_id,
                backend_url=cfg.backend_url,
            )

            reply = str(payload.get("reply", "")).strip()
            if not reply:
                reply = "I’m here with you. What’s going on?"

            print(f"Assistant Reply: {reply}")
            audio = speaker.synthesize(reply)
            speaker.play(audio)

    finally:
        db.close()


if __name__ == "__main__":
    main()