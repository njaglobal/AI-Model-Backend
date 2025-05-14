import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "incident-uploads"
LOCAL_DIR = "training_data"
METADATA_FILE = os.path.join(LOCAL_DIR, "_downloaded_metadata.json")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def download_images():
    print("🔍 Checking for updated images in Supabase...")
    updated = False
    force_all = not os.path.exists(LOCAL_DIR) or not any(os.scandir(LOCAL_DIR))

    metadata = {} if force_all else load_metadata()
    new_metadata = {}

    folders = ["road", "fire", "none"]

    if force_all:
        print("⚠️  Local training data missing or empty. Re-downloading all images...")

    for folder in folders:
        print(f"📂 Checking folder: {folder}")
        os.makedirs(os.path.join(LOCAL_DIR, folder), exist_ok=True)

        try:
            files = supabase.storage.from_(BUCKET_NAME).list(folder)
        except Exception as e:
            print(f"❌ Error listing folder '{folder}': {e}")
            continue

        if not files:
            print(f"⚠️ No files found in folder: {folder}")
            continue

        for file in files:
            if not isinstance(file, dict):
                print(f"⚠️ Skipping malformed file entry in {folder}: {file}")
                continue

            name = file.get("name")
            if not name:
                print(f"⚠️ Skipping file without name in {folder}")
                continue

            key = f"{folder}/{name}"
            metadata_size = file.get("metadata", {})
            size = metadata_size.get("size", 0) if isinstance(metadata_size, dict) else 0
            new_metadata[key] = size

            if not force_all and metadata.get(key) == size:
                continue  # already downloaded

            print(f"⬇️ Downloading: {key}")
            try:
                content = supabase.storage.from_(BUCKET_NAME).download(key)
                with open(os.path.join(LOCAL_DIR, folder, name), "wb") as f:
                    f.write(content)
                updated = True
            except Exception as e:
                print(f"❌ Failed to download {key}: {e}")

    save_metadata(new_metadata)
    return updated