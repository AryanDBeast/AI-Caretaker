import speech_recognition as sr
import pygame
import io
from google.cloud import texttospeech
from openai import OpenAI
from datetime import datetime, timedelta
import calendar
import gspread
from google.oauth2.service_account import Credentials
import os
import time
from collections import deque

# ----------------------------
# Essential info
# ----------------------------
realName = "[INSERT CARETAKER NAME]"
phone = "[INSERT PHONE NUMBER]"

# SECURITY NOTE: Rotate this key and store it in an env var instead.
client = OpenAI(api_key="[INSERT API KEY HERE]")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '[INSERT JSON FILE HERE]'

SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), scopes=SCOPES)
gc = gspread.authorize(credentials)

# Open spreadsheet ONCE (faster)
spread = gc.open("Research Project")
intake = spread.worksheet("Intake History")
conversation = spread.worksheet("Convo History")
data = spread.worksheet("Data")

predefined_memory = [
    {"role": "system", "content": (
        f"You are a caregiver for a dementia patient will their family member ({realName}) is away"
        f"Answer everything in 50 words or less, in simple words, facilitate conversation, and ensure safety as best you can"
        f"Encourage the patient to do daily activities and rest, but they should not leave the home"
        f"In case of an emergency (falls or strangers) add EMERGENCY and a severity score/100 at the end of your response"
        f"Please thoroughly analyze in emergency detection and ask for clarification if uncertain"
    )}
]

# Initialize Pygame mixer for audio playback ONCE
pygame.mixer.init()

usable_mic = 1  # 1-Airpods, 1-Mac, 0-phone

eveningMed = False
morningMed = False

speech_time = 0.0
response_time = 0.0
update_time = 0.0

# Reuse these objects (faster than recreating every loop)
tts_client = texttospeech.TextToSpeechClient()
recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=usable_mic)

# Make the AI patient (set once)
recognizer.pause_threshold = 2
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

# Keep a rolling buffer of convo history in memory for last ~3 minutes
# Each item: (timestamp_dt, role, message)
recent_buffer = deque(maxlen=300)  # plenty for 3 minutes


def _prime_recent_buffer_from_sheet():
    """
    One-time load to seed recent buffer so first responses still have context
    without repeatedly reading the full sheet.
    """
    try:
        rows = conversation.get_all_values()
    except Exception as e:
        print(f"Warning: couldn't prime buffer from sheet: {e}")
        return

    now = datetime.now()

    for row in rows[1:]:  # skip header
        if len(row) < 4:
            continue
        date_str, time_str, role, message = row[:4]
        try:
            ts = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %H:%M")
        except ValueError:
            continue

        if now - ts <= timedelta(minutes=3):
            recent_buffer.append((ts, role.lower(), message))


_prime_recent_buffer_from_sheet()


def get_recent_history():
    """
    FAST: Uses in-memory buffer instead of reading the full sheet every time.
    Maintains same behavior: includes system prompt + last 3 minutes.
    """
    now = datetime.now()
    msgs = predefined_memory.copy()

    for ts, role, message in recent_buffer:
        if now - ts <= timedelta(minutes=3):
            # role in sheet is "User"/"Assistant" -> lower for API
            msgs.append({"role": role, "content": message})

    return msgs


def get_openai_response(prompt: str) -> str:
    global response_time, update_time

    now = datetime.now()

    recent_history = get_recent_history()
    recent_history.append({"role": "user", "content": prompt})

    # Response timer start
    t0_response = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=recent_history
    )
    response_time = time.perf_counter() - t0_response

    assistant_text = response.choices[0].message.content.strip()

    # ---- SYSTEM UPDATE TIME START (Sheets writes) ----
    t0_update = time.perf_counter()

    date_cell = str(now.strftime("%D"))
    time_cell = str(now.strftime("%H:%M"))

    # Batch append BOTH rows at once (faster than append_row twice)
    conversation.append_rows([
        [date_cell, time_cell, "User", prompt],
        [date_cell, time_cell, "Assistant", assistant_text]
    ])

    update_time = time.perf_counter() - t0_update
    # ---- SYSTEM UPDATE TIME END ----

    # Update in-memory buffer too (so no sheet read needed)
    recent_buffer.append((now, "user", prompt))
    recent_buffer.append((now, "assistant", assistant_text))

    print(f"Response Time (sec): {response_time:.3f}")
    print(f"System Update Time (sec): {update_time:.3f}")

    return assistant_text


def text_to_speech(text: str):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Wavenet-D")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        return io.BytesIO(response.audio_content)
    except Exception as e:
        print(f"An error occurred during text-to-speech conversion: {e}")
        return None


def play_audio(audio_stream):
    if audio_stream is None:
        return
    audio_stream.seek(0)
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def listen_to_microphone():
    """Listen to audio from the selected microphone and convert it to text."""
    global usable_mic, speech_time, recognizer

    # Recreate Microphone each call (prevents stale stream issues after device changes)
    try:
        with sr.Microphone(device_index=usable_mic) as source:
            print("Listening...")
            # Optional: helps stability in noisy environments; small time cost
            # recognizer.adjust_for_ambient_noise(source, duration=0.2)

            audio = recognizer.listen(source, timeout=60, phrase_time_limit=20)

        t0_speech = time.perf_counter()
        text = recognizer.recognize_google(audio)
        speech_time = time.perf_counter() - t0_speech
        print(f"You said: {text}")
        print(f"Speech Time (sec): {speech_time:.3f}")
        return text

    except sr.WaitTimeoutError:
        # No speech heard within timeout
        return None

    except (AssertionError, OSError) as e:
        # Covers "Audio source must be entered..." and device disconnect issues
        print(f"Mic error (likely device changed/disconnected): {e}")
        time.sleep(0.5)
        return None

    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None

    except sr.RequestError:
        print("Sorry, there was an issue with the speech recognition service.")
        return None


def response_decider(user_input: str, time_flag: int):
    my_date = datetime.today()

    # AI response
    response = get_openai_response(user_input)
    print(f"OpenAI Response: {response}")
    play_audio(text_to_speech(response))

    # Medication follow-up (kept exactly as your logic)
    if time_flag == 1:
        response = (
            f"It is currently the morning time on {calendar.day_name[my_date.weekday()]}. "
            f"Please go to your medicine box and take the medicine under "
            f"{calendar.day_name[my_date.weekday()]} morning."
        )
        print(f"OpenAI Response: {response}")
        play_audio(text_to_speech(response))

    elif time_flag == 2:
        response = (
            f"It is currently the evening time on {calendar.day_name[my_date.weekday()]}. "
            f"Please go to your medicine box and take the medicine under "
            f"{calendar.day_name[my_date.weekday()]} evening."
        )
        print(f"OpenAI Response: {response}")
        play_audio(text_to_speech(response))


def intakeTracker():
    global morningMed, eveningMed
    now = datetime.now()
    hhmm = int(now.strftime("%H%M"))

    if 2200 > hhmm > 1900:
        eveningMed = True
        intake.append_row([str(now.strftime("%D")), str(now.strftime("%H:%M")), "Evening"])
        print("Data written successfully!")
    elif 1200 > hhmm > 600:
        morningMed = True
        intake.append_row([str(now.strftime("%D")), str(now.strftime("%H:%M")), "Morning"])
        print("Data written successfully!")


def main():
    global morningMed, eveningMed

    while True:
        now = datetime.now()
        hhmm = int(now.strftime("%H%M"))

        user_input = listen_to_microphone()
        if not user_input:
            continue

        if "took" in user_input and "medicine" in user_input:
            response = "Great, if you have anymore questions feel free to ask!"
            print(f"OpenAI Response: {response}")
            play_audio(text_to_speech(response))
            intakeTracker()
        else:
            if 2200 > hhmm > 2200 and (eveningMed is False):
                response_decider(user_input, 2)
            elif 1200 > hhmm > 600 and (morningMed is False):
                response_decider(user_input, 1)
            else:
                response_decider(user_input, 0)

        # metrics row (kept same) — now update_time is correctly set
        data.append_row([f"{speech_time:.3f}", f"{response_time:.3f}", f"{update_time:.3f}"])


if __name__ == "__main__":
    main()
