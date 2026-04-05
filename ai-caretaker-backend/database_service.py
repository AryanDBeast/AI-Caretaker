import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


class DatabaseService:
    def __init__(self):
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment")

        self.conn = psycopg.connect(database_url)
        self.cursor = self.conn.cursor()

    def close(self):
        try:
            self.cursor.close()
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

    def append_statement(
        self,
        patient_id,
        timestamp,
        speaker,
        statement,
        activity_type=None,
        medication_type=None,
        emergency_type=None,
        severity_score=None,
    ):
        self.cursor.execute(
            """
            INSERT INTO statement_log (
                patient_id,
                "timestamp",
                speaker,
                statement,
                activity_type,
                medication_type,
                emergency_type,
                severity_score
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                patient_id,
                timestamp,
                speaker,
                statement,
                activity_type,
                medication_type,
                emergency_type,
                severity_score,
            ),
        )
        self.conn.commit()

    def get_patient_with_caregiver(self, patient_id):
        self.cursor.execute(
            """
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
            WHERE p.patient_id = %s
            """,
            (patient_id,),
        )
        return self.cursor.fetchone()

    def list_patients(self):
        self.cursor.execute(
            """
            SELECT patient_id, patient_first_name, patient_last_name
            FROM patient
            ORDER BY patient_id
            """
        )
        return self.cursor.fetchall()

    def get_recent_statements(self, patient_id, limit=10):
        self.cursor.execute(
            """
            SELECT speaker, statement, "timestamp",
                activity_type, medication_type,
                emergency_type, severity_score
            FROM statement_log
            WHERE patient_id = %s
            ORDER BY "timestamp" DESC
            LIMIT %s
            """,
            (patient_id, limit),
        )
        return self.cursor.fetchall()
    
    def list_caregivers(self):
        self.cursor.execute(
            """
            SELECT caregiver_id, caregiver_first_name, caregiver_last_name
            FROM caregiver
            ORDER BY caregiver_id
            """
        )
        return self.cursor.fetchall()