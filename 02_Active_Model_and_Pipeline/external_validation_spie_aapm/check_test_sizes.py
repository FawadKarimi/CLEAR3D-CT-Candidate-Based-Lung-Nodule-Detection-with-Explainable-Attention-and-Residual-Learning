import json
with open('tcia_series_list.json') as f:
    data = json.load(f)

test_scans = [x for x in data if 'LUNGx-CT' in x['PatientID']]
total_slices = sum(int(x['ImageCount']) for x in test_scans)
total_bytes = sum(int(x['FileSize']) for x in test_scans)

print(f"Total test scans: {len(test_scans)}")
print(f"Total DICOM slices: {total_slices}")
print(f"Total uncompressed size: {total_bytes / 1024**3:.2f} GB ({total_bytes / 1024**2:.1f} MB)")
print(f"Average slices per scan: {total_slices / len(test_scans):.1f}")
