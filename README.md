# AI Caretaker

## Overview

AI Caretaker is a full-stack, voice-driven AI system designed to assist dementia patients in real time while providing caregivers with continuous monitoring, structured data logging, and emergency alerts.

The system enables natural, hands-free interaction for patients and supports caregivers through automated tracking, analytics, and notifications.

---

## Motivation

This project was inspired by a real-world scenario involving my grandfather, who has mild dementia. Traditional care solutions often require manual interaction, which is unreliable for patients with memory impairments.

AI Caretaker was built to provide a passive, voice-first solution that prioritizes simplicity for the patient while delivering meaningful insights and alerts to caregivers.

---

## System Architecture

The system is composed of three primary components:

### 1. Local Device (Patient Interface)
- Runs on a laptop or Raspberry Pi
- Captures voice input using speech recognition
- Sends transcripts to backend API
- Plays AI-generated responses

### 2. Backend (FastAPI - Deployed on Render)
- Processes user input using an LLM
- Detects emergencies and assigns severity scores
- Logs all interactions to PostgreSQL (Supabase)
- Sends SMS alerts to caregivers via Gmail → carrier gateway

### 3. Frontend (Next.js - Deployed on Vercel)
- Caregiver dashboard
- Displays conversation logs and analytics
- Provides real-time visibility into patient activity

---

## Features

- Voice-based interaction (no manual input required)
- AI-generated conversational responses
- Emergency detection with severity scoring
- Real-time SMS alerts to caregivers
- Structured logging of:
  - Conversations
  - Activities
  - Medication mentions
- Caregiver dashboard with analytics and logs
- Multi-patient support

---

## Data Flow

Patient Speech → Local Device → Backend API → AI Processing  
                                              ↓  
                                       Database Logging  
                                              ↓  
                                     Emergency Detection  
                                              ↓  
                                    SMS Alert to Caregiver  
                                              ↓  
                                   Frontend Dashboard  

---

## Repository Structure

AI-Caretaker/
│
├── ai-caretaker-backend/
│   ├── main.py
│   ├── caretaker_brain.py
│   ├── database_service.py
│   └── requirements.txt
│
├── ai-caretaker-frontend/
│   ├── app/
│   ├── public/
│   └── package.json
│
├── ai-caretaker-local/
│   ├── Main.py
│   └── supporting modules
│
└── README.md

---

## Technologies Used

- Python (backend + device logic)
- FastAPI (API layer)
- OpenAI API (LLM processing)
- PostgreSQL / Supabase (database)
- SpeechRecognition (STT)
- Google TTS / audio playback (TTS)
- SMTP (Gmail → SMS gateway)
- Next.js + Tailwind (frontend)
- Render (backend hosting)
- Vercel (frontend hosting)

---

## Development Process

This project evolved through multiple architectural iterations:

1. Initial version: fully local Python-based assistant  
2. Added AI-based emergency detection and structured logging  
3. Introduced PostgreSQL database for persistent storage  
4. Refactored into a backend API (FastAPI) for scalability  
5. Implemented SMS alert system for real-time caregiver notifications  
6. Built a full frontend dashboard using Next.js  
7. Deployed system to cloud infrastructure (Render + Vercel)

During development, the repository was restructured from a local-only codebase into a full-stack architecture. This required consolidating and reorganizing the codebase into a clean monorepo format, which resulted in resetting earlier incremental commit history in favor of a clearer, production-ready structure.

---

## Current Status

- Backend fully deployed and operational  
- Database integration complete  
- SMS alert system functional  
- Frontend dashboard live and connected  
- End-to-end system working in real time  

---

## Future Improvements

- Authentication system for caregivers
- Real-time push notifications
- Data visualization (charts and trends)
- Multi-device deployment (Raspberry Pi hardware)
- Improved emergency classification models
- HIPAA-compliant data handling

---

## Author

Aryan Dhingra  
Bergen County Academies
