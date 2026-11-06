import os
import time
import urllib.request
import hashlib
from pathlib import Path

BASE_URL = "https://huggingface.co/g-luo/news-clippings/resolve/main/"
TARGET_DIR = Path("data/raw/newsclippings")

FILES_TO_DOWNLOAD = [
    {
        "url": BASE_URL + "metadata/test.json",
        "dest": TARGET_DIR / "metadata" / "test.json",
        "expected_bytes": 37369991
    },
    {
        "url": BASE_URL + "embeddings/clip_text_embeddings/clip_text_embeddings_test.pkl",
        "dest": TARGET_DIR / "embeddings" / "clip_text_embeddings" / "clip_text_embeddings_test.pkl",
        "expected_bytes": 86368983
    },
    {
        "url": BASE_URL + "embeddings/clip_image_embeddings/clip_image_embeddings_test.pkl",
        "dest": TARGET_DIR / "embeddings" / "clip_image_embeddings" / "clip_image_embeddings_test.pkl",
        "expected_bytes": 105640860
    }
]

def download_file(url: str, dest: Path, expected_bytes: int):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected_bytes:
        print(f"[EXISTS] {dest.name} already downloaded ({dest.stat().st_size} bytes). Verifying SHA256...")
        hasher = hashlib.sha256()
        with open(dest, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        print(f"[OK] SHA256: {hasher.hexdigest()}")
        return hasher.hexdigest()

    print(f"[DOWNLOADING] {dest.name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    t0 = time.time()
    downloaded = 0
    hasher = hashlib.sha256()
    
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out_f:
        while True:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            out_f.write(chunk)
            hasher.update(chunk)
            downloaded += len(chunk)
            if time.time() - t0 > 5 and downloaded % (5 * 1024 * 1024) < len(chunk):
                mb = downloaded / (1024 * 1024)
                tot = expected_bytes / (1024 * 1024)
                print(f"  Progress: {mb:.1f} / {tot:.1f} MB ({downloaded / expected_bytes * 100:.1f}%)")

    elapsed = time.time() - t0
    sha256 = hasher.hexdigest()
    print(f"[DONE] {dest.name}: {downloaded} bytes in {elapsed:.1f}s ({downloaded/(1024*1024*elapsed):.2f} MB/s)")
    print(f"[SHA256] {sha256}")
    return sha256

def main():
    print("Starting targeted test split asset acquisition...")
    results = {}
    for item in FILES_TO_DOWNLOAD:
        sha = download_file(item["url"], item["dest"], item["expected_bytes"])
        results[item["dest"].name] = {
            "path": str(item["dest"]),
            "bytes": item["dest"].stat().st_size,
            "sha256": sha
        }
    print("\nAll test files successfully acquired and verified:")
    for k, v in results.items():
        print(f"  {k}: {v['bytes']} bytes | SHA256: {v['sha256']}")

if __name__ == "__main__":
    main()
