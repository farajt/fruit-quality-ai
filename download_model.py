# download_model.py — Run once to get model weights from Drive
# Users run this: python download_model.py

import os
import gdown

WEIGHTS_URL = "https://drive.google.com/file/d/1lDLrJuQt3GILWXPfO3PIHQsdYnkd5MwA/view?usp=drive_link"
# You will fill this in after Phase 4 Step 9

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "app", "model", "model_weights.weights.h5"
)

def download_weights():
    if os.path.exists(WEIGHTS_PATH):
        print("✓ Model weights already exist. Skipping download.")
        return
    print("⬇️  Downloading model weights from Google Drive...")
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    gdown.download(WEIGHTS_URL, WEIGHTS_PATH, quiet=False)
    print("✓ Model weights downloaded successfully.")

if __name__ == "__main__":
    download_weights()