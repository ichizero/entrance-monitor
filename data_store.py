import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


class DataStore:
    def __init__(self):
        firestore_key = {
            "type": "service_account",
            "project_id": os.environ['FIRESTORE_PROJECT_ID'],
            "private_key_id": os.environ['FIRESTORE_KEY_ID'],
            "private_key": os.environ['FIRESTORE_KEY'].replace(r"\n", "\n"),
            "client_email": os.environ['FIRESTORE_CLIENT_EMAIL'],
            "client_id": os.environ['FIRESTORE_CLIENT_ID'],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.environ['FIRESTORE_CERT_URL']
        }
        cred = credentials.Certificate(firestore_key)
        firebase_admin.initialize_app(cred)

        self.db = firestore.client()

    def add(self, time, name):
        doc_ref = self.db.collection('rooms').document('cilab').collection('attendee').document()
        doc_ref.set({
            "name": self.db.collection('users').document(name),
            "time": int(time.timestamp())
        })
