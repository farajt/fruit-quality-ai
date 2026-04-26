# download_model.py

import os, re
import gdown

# Paste your full Google Drive sharing link here
WEIGHTS_URL = "https://drive.google.com/file/d/1lDLrJuQt3GILWXPfO3PIHQsdYnkd5MwA/view?usp=sharing"

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "app", "model", "model_weights.weights.h5"
)

def download_weights():
    if os.path.exists(WEIGHTS_PATH):
        print("✓ Model weights already exist. Skipping.")
        return

    print("⬇️ Downloading model weights from Google Drive...")
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    # Extract file ID and build direct URL
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', WEIGHTS_URL)
    file_id = match.group(1) if match else None
    url = f"https://drive.google.com/uc?id={file_id}" \
          if file_id else WEIGHTS_URL

    gdown.download(url, WEIGHTS_PATH, quiet=False, fuzzy=True)
    print("✓ Done.")

if __name__ == "__main__":
    download_weights()