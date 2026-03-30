import urllib.request
import sys
import os

def hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = downloaded * 100 / total_size if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading... {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.1f}%)")
    sys.stdout.flush()

urls = [
    ("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip", "dataset/Flickr8k_text.zip"),
    ("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", "dataset/Flickr8k_Dataset.zip")
]

os.makedirs("dataset", exist_ok=True)

for url, path in urls:
    print(f"\nEvaluating {url} -> {path}...")
    if os.path.exists(path) and os.path.getsize(path) > 10 * 1024 * 1024:
        print(f"Skipping {path}, file already exists and looks valid.")
        continue
        
    print(f"Starting download...")
    try:
        urllib.request.urlretrieve(url, path, reporthook=hook)
        print(f"\nFinished saving {path}")
    except Exception as e:
        print(f"\nFailed to download {path}: {e}")
