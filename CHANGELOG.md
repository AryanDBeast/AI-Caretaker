# Changelog

Commit history in this repository begins at v1.0. Earlier versions were developed as separate local prototypes and are summarized at the end of this file.

## v1.1.0 — July 2026

**Caregiver accounts**
- `POST /login` and `POST /logout` with bearer-token sessions; passwords are checked against bcrypt hashes in PostgreSQL.
- `GET /me/patients` returns only the patients assigned to the signed-in caregiver.
- `GET /patients/{id}/statements` now requires login and returns `403` for patients not assigned to the caller.

**Proactive check-ins**
- Background scheduler runs every 60 s, active 8 AM–8 PM Eastern.
- After 2 hours with no patient speech, the LLM writes a short check-in based on the patient's last statement.
- The device polls `GET /patients/{id}/checkin` and speaks the message; it also ignores its own playback so it doesn't transcribe itself.
- If the patient doesn't respond within 15 minutes, the caregiver gets an SMS (severity 80) and the missed check-in is logged.

**Dashboard**
- Login screen, caregiver-scoped patient list, and session handling.

**Fixes**
- Database connections use autocommit. Idle transactions had been holding table locks and causing upstream timeouts on Supabase.

## v1.0.0 — April 2026

First commit in this repository: the system restructured into a monorepo with three components.

- **Backend (FastAPI on Render):** `POST /conversation` sends patient speech to the LLM and gets back structured JSON (reply, emergency flag + 0–100 severity, activity, medication). Every statement is logged to PostgreSQL (Supabase). Emergencies with severity ≥ 70 send an SMS to the assigned caregiver through an email-to-SMS gateway.
- **Patient device:** continuous listen → transcribe → send → speak loop with Google Cloud TTS; loads the active patient profile at startup.
- **Caregiver dashboard (Next.js on Vercel):** conversation logs, emergency events, and medication mentions per patient.
- Multi-patient data model; all credentials moved to environment variables.

## Pre-repository prototypes

| Version | Period | Summary |
| --- | --- | --- |
| v0.3 | 2025 – early 2026 | Structured JSON detections replaced free-text replies; logging moved to PostgreSQL. |
| v0.2 | 2024 – 2025 | Caretaker system prompt, LLM-raised danger flag, medication reminders, conversation logging to a spreadsheet. |
| v0.1 | 2024 | Single-file local voice assistant (speech-to-text → LLM → text-to-speech). |
