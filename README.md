# AI Caretaker

A voice-first, in-home support system for people living with dementia. The patient simply talks to a small device. The system answers conversationally, watches for signs of danger, and keeps caregivers informed through a live dashboard and SMS alerts.

Product site: [ai-caretaker.live](https://ai-caretaker.live)

---

## Why voice-first

Most care tools assume the user can remember to open an app, tap a button, or charge a wearable. People with memory impairment often can't. AI Caretaker removes the interface entirely: there is nothing to learn, so the patient talks to it the way they would talk to a person. Everything structured (logging, triage, alerts) happens behind the scenes for the caregiver.

---

## How it works

```
 Patient speaks
      │
      ▼
┌──────────────────┐   transcript    ┌───────────────────────────┐
│  Local device    │ ──────────────▶ │  Backend (FastAPI)         │
│  (Pi / laptop)   │                 │                           │
│  STT + TTS       │ ◀────────────── │  LLM → structured JSON     │
└──────────────────┘   spoken reply  │   • reply                 │
      ▲                              │   • emergency + severity  │
      │ polls for check-ins          │   • activity / medication │
      └───────────────────────────── │  Check-in scheduler        │
                                     └─────────────┬─────────────┘
                                                   │
                         ┌─────────────────────────┼─────────────────────┐
                         ▼                         ▼                     ▼
                 PostgreSQL (Supabase)     SMS alert to caregiver   Caregiver dashboard
                 conversation + event log  (email-to-SMS gateway)   (Next.js)
```

1. **Listen.** The local device captures speech, transcribes it, and posts the transcript to the backend.
2. **Understand.** The backend sends the conversation to an LLM that returns a structured JSON object: the spoken reply, plus any detected emergency (with a 0–100 severity score), activity, or medication mention.
3. **Act.** Every statement and detection is written to PostgreSQL. Events with severity ≥ 70 trigger an SMS to the patient's assigned caregiver.
4. **Check in.** If a patient has been quiet for 2 hours during the day (8 AM–8 PM), the backend generates a context-aware check-in from their last statement. If there is no reply within 15 minutes, the caregiver is alerted.
5. **Review.** Caregivers sign in to the dashboard to see conversation history, emergencies, and medication events for the patients assigned to them.

---

## Repository layout

```
AI-Caretaker/
├── ai-caretaker-backend/      FastAPI service (deployed on Render)
│   ├── main.py                API routes, auth, SMS alerts, check-in scheduler
│   ├── caretaker_brain.py     LLM prompts and structured-output parsing
│   ├── database_service.py    PostgreSQL access layer
│   └── requirements.txt
│
├── ai-caretaker-local/        Code that runs on the patient's device
│   ├── Main.py                Listen → send → speak loop, check-in polling
│   ├── database_service.py    Patient lookup at startup
│   ├── test_mic.py            Lists microphone device indices
│   └── requirements.txt
│
└── ai-caretaker-frontend/     Caregiver dashboard (Next.js, deployed on Vercel)
    └── app/page.tsx           Login, patient list, logs, emergencies, medications
```

### API

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/conversation` | Process a patient statement; returns the spoken reply and logs detections |
| `GET` | `/patients/{id}/checkin` | Device polls for pending check-in messages to speak |
| `POST` | `/login` / `/logout` | Caregiver session (bcrypt-hashed passwords, bearer tokens) |
| `GET` | `/me/patients` | Patients assigned to the signed-in caregiver |
| `GET` | `/patients/{id}/statements` | Conversation and event history (caregiver-scoped) |

---

## Running it locally

**Backend**

```bash
cd ai-caretaker-backend
pip install -r requirements.txt
export DATABASE_URL=...          # PostgreSQL connection string
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export GMAIL_ADDRESS=...         # sender account for SMS alerts
export GMAIL_APP_PASSWORD=...
uvicorn main:app --reload
```

**Patient device**

```bash
cd ai-caretaker-local
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
export DATABASE_URL=...
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/tts-key.json
python test_mic.py               # find your mic index
export MIC_DEVICE_INDEX=0
python Main.py
```

**Dashboard**

```bash
cd ai-caretaker-frontend
npm install
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

No credentials are stored in this repository. Everything is read from environment variables or a local `.env` file.

---

## Tech stack

| Layer | Tools |
| --- | --- |
| Device | Python, Raspberry Pi 4B, SpeechRecognition (Google Web Speech), Google Cloud Text-to-Speech, pygame |
| Backend | FastAPI, OpenAI API (GPT-4o mini, structured JSON output), psycopg |
| Data | PostgreSQL on Supabase |
| Alerts | SMTP → carrier email-to-SMS gateway |
| Dashboard | Next.js, TypeScript, Tailwind CSS |
| Hosting | Render (API), Vercel (dashboard) |

---

## Evaluation

Results from testing the deployed system:

| Metric | Result |
| --- | --- |
| Speech recognition accuracy | 94.6% |
| Emergency classification accuracy | 76.7% |
| Average end-to-end response latency | 3.24 s |

A two-week in-home pilot with 15 families surfaced three main issues: recognition errors with strong accents, latency spikes on weak home Wi-Fi, and patients rarely starting conversations on their own. The last finding led directly to the proactive check-in scheduler in v1.1.

---

## Version history

The project went through several standalone prototypes before it became this repository. Commit history here starts at v1.0, when the code was consolidated into a single monorepo. Earlier versions lived in separate local projects and were not carried over, so they are summarized below instead.

| Version | Date | What changed |
| --- | --- | --- |
| **v0.1** – Voice prototype | 2024 | Single-file Python assistant: microphone → speech-to-text → LLM → text-to-speech. Built on the voice pipeline from an earlier assistant project. |
| **v0.2** – Caretaker logic | 2024–2025 | Caretaker system prompt, an LLM-raised danger flag, morning/evening medication reminders, and conversation logging to a spreadsheet. Typed-input fallback for testing. |
| **v0.3** – Structured detection | 2025 – early 2026 | Switched from free-text replies to structured JSON (reply + emergency severity + activity + medication). Moved logging from the spreadsheet to PostgreSQL. |
| **v1.0** – Full-stack system | Apr 2026 | Split into three parts: FastAPI backend, patient device client, and Next.js caregiver dashboard. Multi-patient support, SMS alerts, deployed to Render + Vercel. *(first commit in this repo)* |
| **v1.1** – Accounts & check-ins | Jul 2026 | Caregiver login with bcrypt-hashed passwords and per-caregiver patient access. Proactive check-in scheduler with missed-response alerts. Dashboard login and caregiver-scoped views. Fixed Supabase lock timeouts by switching the DB layer to autocommit. |

See [CHANGELOG.md](CHANGELOG.md) for details.

---

## Roadmap

- Push notifications alongside SMS
- Trend charts for activity, mood, and medication adherence
- Better accent robustness and on-device fallback for poor connectivity
- Improved emergency classification
- HIPAA-aligned data handling and persistent session storage
