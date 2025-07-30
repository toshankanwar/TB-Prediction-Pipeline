import os
import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    """
    Initialize Firebase Admin SDK using environment variables.
    Expects the following env vars:
      - FIREBASE_PROJECT_ID
      - FIREBASE_CLIENT_EMAIL
      - FIREBASE_PRIVATE_KEY (with \n replaced by actual newlines)
    """

    # Check if already initialized
    if not firebase_admin._apps:
        project_id = os.getenv('FIREBASE_PROJECT_ID')
        client_email = os.getenv('FIREBASE_CLIENT_EMAIL')
        private_key = os.getenv('FIREBASE_PRIVATE_KEY')

        if not all([project_id, client_email, private_key]):
            raise ValueError("Firebase env vars FIREBASE_PROJECT_ID, FIREBASE_CLIENT_EMAIL, FIREBASE_PRIVATE_KEY must be set")

        # Replace literal '\n' with actual newline characters
        private_key = private_key.replace('\\n', '\n')

        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": project_id,
            "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID', ''),  # optional but good if you want
            "private_key": private_key,
            "client_email": client_email,
            "client_id": os.getenv('FIREBASE_CLIENT_ID', ''),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_X509_CERT_URL', '')
        })

        firebase_admin.initialize_app(cred)

    return firestore.client()

# Usage example:
# db = initialize_firebase()
