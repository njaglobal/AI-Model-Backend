import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def download_images(local_dir="training_data"):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    labels = ["fire", "road"]

    for label in labels:
        path = os.path.join(local_dir, label)
        os.makedirs(path, exist_ok=True)

        # List files in the Supabase bucket directory
        response = supabase.storage.from_(BUCKET).list(f"{label}/")
        for file in response:
            file_name = file["name"]
            print(f"Downloading {label}/{file_name}")
            data = supabase.storage.from_(BUCKET).download(f"{label}/{file_name}")
            with open(f"{path}/{file_name}", "wb") as f:
                f.write(data)

    print("✅ All images downloaded.")
