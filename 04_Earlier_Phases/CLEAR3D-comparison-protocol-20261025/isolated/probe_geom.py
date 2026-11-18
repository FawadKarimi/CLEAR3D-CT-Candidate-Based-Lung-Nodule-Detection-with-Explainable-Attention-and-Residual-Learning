"""Read one fit scan header and map up to three annotations. No confirmation patients."""
import csv
import json
import os

import numpy as np

PREP = "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared"
ANN = "/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/annotations.csv"
ROOT = "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025"
pid = json.load(open(ROOT + "/runs/phase-cf/split.json"))["fit"][0]
for name in os.listdir(PREP):
    folder = os.path.join(PREP, name)
    if not os.path.isdir(folder):
        continue
    record = json.load(open(os.path.join(folder, "record.json")))
    if record["patient_id"] != pid:
        continue
    image = np.load(os.path.join(folder, "image.npy"), mmap_mode="r")
    geometry = json.load(open(os.path.join(folder, "geometry.json")))
    affine = np.array(geometry["affine"], float)
    center = image[tuple(s // 2 for s in image.shape)]
    print("SCAN", name, record["series_uid"])
    print("SHAPE", image.shape, image.dtype, float(center))
    print("AFFINE")
    print(np.array2string(affine, precision=4))
    shown = 0
    with open(ANN, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["seriesuid"] != record["series_uid"]:
                continue
            world = np.array([float(row["coordX"]), float(row["coordY"]), float(row["coordZ"]), 1.0])
            ijk = np.linalg.inv(affine) @ world
            inside = bool(np.all(ijk[:3] >= 0) and np.all(ijk[:3] < np.array(image.shape)))
            print("ANN", row["diameter_mm"], np.array2string(ijk[:3], precision=2), inside)
            shown += 1
            if shown == 3:
                break
    print("SHOWN", shown)
    break
