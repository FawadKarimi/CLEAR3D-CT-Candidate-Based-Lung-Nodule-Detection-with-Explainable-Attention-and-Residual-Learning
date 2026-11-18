"""Script to package all SPIE-AAPM test cohort evaluation deliverables into a certified zip archive."""
import hashlib, json, os, sys, zipfile
from pathlib import Path

EXT_DIR = Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/external_validation_spie_aapm')
ZIP_OUT = EXT_DIR / 'clear3d-spie-aapm-test-cohort-cap4000-package-20261109.zip'
MANIFEST_OUT = EXT_DIR / 'package_manifest_final_20261109.json'

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def package():
    files_to_pack = [
        "frozen_evaluation_protocol_cap4000.json",
        "test_ground_truth.json",
        "tcia_series_list.json",
        "download_status.json",
        "batch_download_test_scans.py",
        "evaluate_test_cohort_cap4000.py",
        "scores_cap4000.csv",
        "test_cohort_cap4000_evaluation_summary.json",
        "eval_cap4000.log",
        "generate_test_cohort_froc_and_overlays.py",
        "froc_test_cohort_cap4000.png"
    ]
    overlays_dir = EXT_DIR / "test_cohort_overlays"
    overlay_files = sorted(list(overlays_dir.glob("*.png"))) if overlays_dir.exists() else []

    manifest = {"files": {}, "package_sha256": None, "total_files": 0}

    print(f"Creating zip archive {ZIP_OUT}...")
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in files_to_pack:
            fp = EXT_DIR / fn
            if fp.exists():
                h = sha256_file(fp)
                manifest["files"][fn] = {"sha256": h, "size_bytes": fp.stat().st_size}
                zf.write(fp, arcname=fn)
                print(f"  Added {fn} ({h[:12]}...)")
            else:
                print(f"  WARNING: {fn} not found!")

        for of in overlay_files:
            rel_name = f"test_cohort_overlays/{of.name}"
            h = sha256_file(of)
            manifest["files"][rel_name] = {"sha256": h, "size_bytes": of.stat().st_size}
            zf.write(of, arcname=rel_name)
            print(f"  Added {rel_name}")

    zip_hash = sha256_file(ZIP_OUT)
    manifest["package_sha256"] = zip_hash
    manifest["total_files"] = len(manifest["files"])
    manifest["package_size_bytes"] = ZIP_OUT.stat().st_size

    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Package complete: {ZIP_OUT} (SHA-256: {zip_hash})")
    print(f"Manifest saved to: {MANIFEST_OUT}")

if __name__ == "__main__":
    package()
