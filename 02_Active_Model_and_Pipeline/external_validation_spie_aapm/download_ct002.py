import urllib.request, zipfile, os, time
from pathlib import Path

raw_test = Path('raw_test')
uid = "1.3.1145223087859355399642916469763443565954368970218"
pid = "LUNGx-CT002"
zip_path = raw_test / f"{pid}.zip"
extract_dir = raw_test / pid

if not extract_dir.exists() or len(list(extract_dir.glob("*.dcm"))) < 272:
    print(f"Downloading {pid} ({uid})...", flush=True)
    t0 = time.time()
    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={uid}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, 'wb') as f:
        bytes_read = 0
        while True:
            chunk = resp.read(1048576)
            if not chunk:
                break
            f.write(chunk)
            bytes_read += len(chunk)
    t_dl = time.time() - t0
    print(f"Downloaded {bytes_read / 1024**2:.1f} MB in {t_dl:.1f}s ({bytes_read / (1024**2 * t_dl):.2f} MB/s)", flush=True)
    
    print("Extracting...", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    print(f"Extracted {len(list(extract_dir.glob('*.dcm')))} DICOM files.", flush=True)
else:
    print(f"{pid} already exists.")
