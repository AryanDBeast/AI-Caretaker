import os
import psycopg


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

    def list_patients(self):
        self.cursor.execute(
            """
            SELECT patient_id, patient_first_name, patient_last_name
            FROM patient
            ORDER BY patient_id
            """
        )
        return self.cursor.fetchall()

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