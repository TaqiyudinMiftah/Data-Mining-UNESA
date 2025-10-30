import os
import time
import glob
import traceback
from typing import List
from roboflow import Roboflow
from tqdm import tqdm  # <<< [1] TAMBAHKAN IMPORT

# =======================
# Konfigurasi
# =======================
WORKSPACE = "cat-detector-zmc90"
PROJECT   = "klasifikasi-makanan-yjo6h"
FOLDER    = "train"                        # folder berisi gambar
EXTS      = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MAX_RETRIES = 6
BASE_SLEEP  = 2.0                          # detik (akan do exponential backoff)
UPLOADED_LOG = "uploaded_files.log"
FAILED_LOG   = "failed_files.log"

# =======================
# Helper: load & append log
# =======================
def load_logged(path: str) -> set:
    s = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s.add(line)
    return s

def append_log(path: str, item: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(item + "\n")

# =======================
# Main
# =======================
def main():
    # Ambil API key dari environment
    api_key = "V36kjEzGLgB93crd7o92"
    if not api_key:
        raise RuntimeError(
            "Environment variable ROBOFLOW_API_KEY tidak ditemukan. "
            "Set dulu, misal di Windows PowerShell: "
            '$env:ROBOFLOW_API_KEY="GANTI_DENGAN_API_KEY_BARU"'
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    # Kumpulkan semua file kandidat
    all_files: List[str] = []
    for ext in EXTS:
        all_files.extend(glob.glob(os.path.join(FOLDER, f"*{ext}")))
    all_files = sorted(all_files)

    if not all_files:
        print(f"Tidak ada file bergambar di '{FOLDER}' dengan ekstensi {EXTS}")
        return

    # Muat file yang sudah sukses & gagal sebelumnya (agar bisa resume)
    already_uploaded = load_logged(UPLOADED_LOG)
    already_failed   = load_logged(FAILED_LOG)  # boleh di-skip atau dicoba lagi

    # Filter: hanya file yang belum tercatat sukses
    to_upload = [p for p in all_files if p not in already_uploaded]

    print(f"Total file ditemukan      : {len(all_files)}")
    print(f"Sudah ter-upload (log)    : {len(already_uploaded)}")
    print(f"Akan dicoba upload lagi   : {len(to_upload)}")

    # <<< [2] GANTI LOOP FOR DENGAN TQDM
    for path in tqdm(to_upload, desc="Uploading files", unit="file", ncols=100):
        # print(f"[{idx}/{len(to_upload)}] Uploading: {path}") # <-- [3] HAPUS INI

        # Retry dengan exponential backoff
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Upload single file (lebih robust dibanding sekaligus folder)
                # Catatan: SDK Roboflow akan menangani hashing & deduplikasi di server
                project.upload(path)
                append_log(UPLOADED_LOG, path)
                
                # <<< [4] GUNAKAN TQDM.WRITE() AGAR TIDAK MERUSAK BAR
                tqdm.write(f"  ✓ Selesai: {path}")
                break
            except Exception as e:
                # Tampilkan ringkas + attempt info
                # <<< [4] GUNAKAN TQDM.WRITE()
                tqdm.write(f"  ⚠️  Gagal attempt {attempt}/{MAX_RETRIES} ({os.path.basename(path)}): {type(e).__name__}")
                # Optional: print(traceback.format_exc())  # jika butuh detail stacktrace

                if attempt == MAX_RETRIES:
                    # Tandai sebagai gagal untuk dicoba ulang nanti
                    append_log(FAILED_LOG, path)
                    # <<< [4] GUNAKAN TQDM.WRITE()
                    tqdm.write(f"  ✗ Ditandai gagal: {path} (cek {FAILED_LOG})")
                else:
                    sleep_s = BASE_SLEEP * (2 ** (attempt - 1))  # exponential backoff
                    # Jeda sedikit acak bisa ditambah untuk menghindari thundering herd
                    time.sleep(sleep_s)

    print("\nUpload selesai ✅")
    print(f"- Log sukses: {UPLOADED_LOG}")
    print(f"- Log gagal : {FAILED_LOG} (bisa diulang jalankan skrip ini lagi)")

if __name__ == "__main__":
    main()