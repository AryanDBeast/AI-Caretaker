import sqlite3
from datetime import datetime

DB_NAME = "caretaker.db"


class DatabaseService:

    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor = self.conn.cursor()

    def append_conversation_pair(self, patient_id, timestamp, user_text, assistant_text):

        self.cursor.execute("""
        INSERT INTO conversation_log (patient_id, timestamp, speaker, statement)
        VALUES (?, ?, ?, ?)
        """, (patient_id, timestamp, "User", user_text))

        self.cursor.execute("""
        INSERT INTO conversation_log (patient_id, timestamp, speaker, statement)
        VALUES (?, ?, ?, ?)
        """, (patient_id, timestamp, "Assistant", assistant_text))

        self.conn.commit()

    def append_emergency(self, patient_id, timestamp, severity_score, statement):

        self.cursor.execute("""
        INSERT INTO emergency_log (patient_id, timestamp, severity_score, statement)
        VALUES (?, ?, ?, ?)
        """, (patient_id, timestamp, severity_score, statement))

        self.conn.commit()

    def append_activity(self, patient_id, timestamp, activity_type, statement):

        self.cursor.execute("""
        INSERT INTO activity_log (patient_id, timestamp, activity_type, statement)
        VALUES (?, ?, ?, ?)
        """, (patient_id, timestamp, activity_type, statement))

        self.conn.commit()

    def append_medication(self, patient_id, timestamp, medication_type, statement):

        self.cursor.execute("""
        INSERT INTO medication_log (patient_id, timestamp, medication_type, statement)
        VALUES (?, ?, ?, ?)
        """, (patient_id, timestamp, medication_type, statement))

        self.conn.commit()

    def list_patients(self):
        self.cursor.execute("""
        SELECT patient_id, patient_first_name, patient_last_name
        FROM patient
        ORDER BY patient_id
        """)
        return self.cursor.fetchall()


    def get_patient_with_caregiver(self, patient_id):
        self.cursor.execute("""
        SELECT
            p.patient_id,
            p.patient_first_name,
            p.patient_last_name,
            p.dob,
            c.caregiver_id,
            c.caregiver_first_name,
            c.caregiver_last_name,
            c.phone_nbr
        FROM patient p
        JOIN caregiver c
        ON p.caregiver_id = c.caregiver_id
        WHERE p.patient_id = ?
        """, (patient_id,))

        return self.cursor.fetchone()