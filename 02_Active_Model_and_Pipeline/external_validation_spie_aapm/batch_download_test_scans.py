import concurrent.futures, json, os, sys, time, urllib.request, zipfile
from pathlib import Path

EXT_DIR = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/external_validation_spie_aapm")
RAW_TEST = EXT_DIR / "raw_test"
RAW_TEST.mkdir(exist_ok=True, parents=True)

with open(EXT_DIR / "tcia_series_list.json") as f:
    series_list = json.load(f)

test_series = [x for x in series_list if "LUNGx-CT" in x["PatientID"]]
test_series.sort(key=lambda x: x["PatientID"])
print(f"Total test series in catalog: {len(test_series)}", flush=True)

def download_and_extract(item):
    pid = item["PatientID"]
    uid = item["SeriesInstanceUID"]
    expected_count = int(item["ImageCount"])
    target_dir = RAW_TEST / pid
    zip_path = RAW_TEST / f"{pid}.zip"
    
    # Check if already complete
    if target_dir.exists():
        dcms = list(target_dir.glob("*.dcm"))
        if len(dcms) >= expected_count:
            return pid, True, f"Already complete ({len(dcms)}/{expected_count} slices)"
    
    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={uid}"
    headers = {"User-Agent": "Mozilla/5.0 (CLEAR3D-CT Independent Evaluation Client)"}
    
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp, open(zip_path, "wb") as out_f:
                bytes_read = 0
                while True:
                    chunk = resp.read(1048576)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    bytes_read += len(chunk)
            t_dl = time.time() - t0
            
            # Extract zip
            target_dir.mkdir(exist_ok=True, parents=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(target_dir)
            
            dcms = list(target_dir.glob("*.dcm"))
            if len(dcms) >= expected_count:
                if zip_path.exists():
                    zip_path.unlink() # remove zip to save space
                return pid, True, f"Success: {len(dcms)} slices ({bytes_read / 1024**2:.1f} MB in {t_dl:.1f}s)"
            else:
                print(f"[{pid}] Warning: extracted {len(dcms)} < expected {expected_count}, retry {attempt}", flush=True)
        except Exception as e:
            print(f"[{pid}] Attempt {attempt} failed: {e}", flush=True)
            time.sleep(attempt * 2)
    
    return pid, False, "Failed after 3 attempts"

print("Starting concurrent download with 3 workers...", flush=True)
t_total0 = time.time()
results = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    future_to_pid = {executor.submit(download_and_extract, item): item["PatientID"] for item in test_series}
    for i, fut in enumerate(concurrent.futures.as_completed(future_to_pid), 1):
        pid = future_to_pid[fut]
        try:
            pid, ok, msg = fut.result()
            results[pid] = {"ok": ok, "msg": msg}
            print(f"[{i}/{len(test_series)}] {pid}: {msg}", flush=True)
        except Exception as e:
            results[pid] = {"ok": False, "msg": str(e)}
            print(f"[{i}/{len(test_series)}] {pid} Exception: {e}", flush=True)

total_time = time.time() - t_total0
success_count = sum(1 for r in results.values() if r["ok"])
print(f"\nDownload finished in {total_time/60:.2f} min. Success: {success_count}/{len(test_series)}", flush=True)
with open(EXT_DIR / "download_status.json", "w") as f:
    json.dump({"total_time_s": total_time, "success_count": success_count, "results": results}, f, indent=2)