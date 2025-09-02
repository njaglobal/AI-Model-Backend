# import os
# import json
# from dotenv import load_dotenv
# from supabase import create_client, Client

# load_dotenv()

# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# BUCKET_NAME = "incident-uploads"
# LOCAL_DIR = "training_data"
# METADATA_FILE = os.path.join(LOCAL_DIR, "_downloaded_metadata.json")

# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# def load_metadata():
#     if os.path.exists(METADATA_FILE):
#         with open(METADATA_FILE, "r") as f:
#             return json.load(f)
#     return {}

# def save_metadata(metadata):
#     os.makedirs(LOCAL_DIR, exist_ok=True)
#     with open(METADATA_FILE, "w") as f:
#         json.dump(metadata, f)

# def download_images():
#     print("🔍 Checking for updated images in Supabase...")
#     updated = False
#     force_all = not os.path.exists(LOCAL_DIR) or not any(os.scandir(LOCAL_DIR))

#     metadata = {} if force_all else load_metadata()
#     new_metadata = {}

#     folders = ["road", "fire", "none-accident"]

#     if force_all:
#         print("⚠️  Local training data missing or empty. Re-downloading all images...")

#     for folder in folders:
#         print(f"📂 Checking folder: {folder}")
#         os.makedirs(os.path.join(LOCAL_DIR, folder), exist_ok=True)

#         try:
#             files = supabase.storage.from_(BUCKET_NAME).list(folder)
#         except Exception as e:
#             print(f"❌ Error listing folder '{folder}': {e}")
#             continue

#         if not files:
#             print(f"⚠️ No files found in folder: {folder}")
#             continue

#         for file in files:
#             if not isinstance(file, dict):
#                 print(f"⚠️ Skipping malformed file entry in {folder}: {file}")
#                 continue

#             name = file.get("name")
#             if not name:
#                 print(f"⚠️ Skipping file without name in {folder}")
#                 continue

#             key = f"{folder}/{name}"
#             metadata_size = file.get("metadata", {})
#             size = metadata_size.get("size", 0) if isinstance(metadata_size, dict) else 0
#             new_metadata[key] = size

#             if not force_all and metadata.get(key) == size:
#                 continue  # already downloaded

#             print(f"⬇️ Downloading: {key}")
#             try:
#                 content = supabase.storage.from_(BUCKET_NAME).download(key)
#                 with open(os.path.join(LOCAL_DIR, folder, name), "wb") as f:
#                     f.write(content)
#                 updated = True
#             except Exception as e:
#                 print(f"❌ Failed to download {key}: {e}")

#     save_metadata(new_metadata)
#     return updated


import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET")
BUCKET_MODEL_NAME = "models"
# LOCAL_DIR = "training_data_backup"
LOCAL_DIR = "training_data"
METADATA_FILE = os.path.join(LOCAL_DIR, "_downloaded_metadata.json")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# def load_metadata():
#     """
#     Load metadata from Supabase if it exists; fallback to local _downloaded_metadata.json.
#     Returns a dictionary.
#     """
#     # Fallback: load local metadata if exists
#     if os.path.exists(METADATA_FILE):
#         with open(METADATA_FILE, "r") as f:
#             metadata = json.load(f)
#             print(f"✅ Loaded local metadata.json with {len(metadata)} entries")
#             return metadata

#     # Try to load from Supabase
#     try:
#         content = supabase.storage.from_(BUCKET_MODEL_NAME).download("_downloaded_metadata.json")
#         if content:
#             metadata = json.loads(content.decode("utf-8"))
#             print(f"✅ Found metadata.json in Supabase with {len(metadata)} entries")

#             # Save locally for future use
#             os.makedirs(LOCAL_DIR, exist_ok=True)
#             with open(METADATA_FILE, "w") as f:
#                 json.dump(metadata, f)
#             return metadata
#     except Exception as e:
#         print(f"⚠️ Could not load metadata.json from Supabase: {e}")

#     # Cold start: nothing exists
#     print("⚠️ No metadata found. Starting with empty dataset.")
#     return {}


# def save_metadata(metadata):
#     os.makedirs(LOCAL_DIR, exist_ok=True)
#     with open(METADATA_FILE, "w") as f:
#         json.dump(metadata, f)

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
    """
    Syncs images from Supabase bucket into LOCAL_DIR.
    Returns a list of *newly downloaded* local file paths.
    """
    remove_empty_placeholders(BUCKET_NAME)

    print("🔍 Checking for updated images in Supabase...")
    new_files = []
    force_all = not os.path.exists(LOCAL_DIR) or not any(os.scandir(LOCAL_DIR))

    metadata = {} if force_all else load_metadata()
    new_metadata = {}

    folders = ["road", "fire", "none-accident"]

    if force_all:
        print("⚠️ Local training data missing or empty. Re-downloading all images...")

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
            if name.endswith(".emptyFolderPlaceholder"):
                continue

            key = f"{folder}/{name}"
            metadata_size = file.get("metadata", {})
            size = metadata_size.get("size", 0) if isinstance(metadata_size, dict) else 0
            new_metadata[key] = size

            local_path = os.path.join(LOCAL_DIR, folder, name)

            # Skip if unchanged
            if not force_all and metadata.get(key) == size:
                continue

            print(f"⬇️ Downloading: {key}")
            try:
                content = supabase.storage.from_(BUCKET_NAME).download(key)
                with open(local_path, "wb") as f:
                    f.write(content)
                new_files.append(local_path)
            except Exception as e:
                print(f"❌ Failed to download {key}: {e}")

    save_metadata(new_metadata)

    # ✅ Sanity check: return list of new files
    if not any(Path(LOCAL_DIR).rglob("*.*")):
        print("❌ No training data exists in Supabase bucket.")
        return []

    return new_files


def remove_empty_placeholders(bucket_name, path=""):
    """
    Remove all .emptyFolderPlaceholder files from a Supabase bucket recursively.
    """
    items = supabase.storage.from_(bucket_name).list(path, {"limit": 1000})
    to_delete = []

    for item in items:
        name = item["name"]
        # Delete placeholder files
        if name.endswith(".emptyFolderPlaceholder"):
            to_delete.append(name)
        # Recurse into folders if any (simple heuristic)
        elif not name.lower().endswith((".jpg", ".jpeg", ".png")):
            remove_empty_placeholders(bucket_name, name)

    if to_delete:
        supabase.storage.from_(bucket_name).remove(to_delete)
        print(f"🗑️ Removed {len(to_delete)} .emptyFolderPlaceholder files from '{path or 'root'}'")
   
    