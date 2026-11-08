"""
Robust dataset downloader with automatic retry and resume support.
Downloads the Kaggle Chest X-Ray Pneumonia dataset.
"""
import kagglehub
import shutil
import os
import time

MAX_RETRIES = 10

for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"\n--- Attempt {attempt}/{MAX_RETRIES} ---")
        print("Downloading dataset (will resume from where it left off)...")
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print("Dataset downloaded to:", path)

        # Move to project data/ folder
        target_dir = "data"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        source_dir = os.path.join(path, "chest_xray")
        if os.path.exists(source_dir):
            print("Copying dataset to data/ folder...")
            for item in os.listdir(source_dir):
                s = os.path.join(source_dir, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.copytree(s, d)
            print("Done! Dataset is ready in data/")
        else:
            print("Could not find chest_xray folder. Listing contents:")
            for item in os.listdir(path):
                print(f"  {item}")
        break  # success, exit loop

    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt < MAX_RETRIES:
            wait = min(30, 5 * attempt)
            print(f"Waiting {wait}s before retrying...")
            time.sleep(wait)
        else:
            print("All retries exhausted. Please check your internet connection.")
            print("You can also download manually from:")
            print("  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
            print("Extract the 'chest_xray' folder contents into the 'data/' folder.")
