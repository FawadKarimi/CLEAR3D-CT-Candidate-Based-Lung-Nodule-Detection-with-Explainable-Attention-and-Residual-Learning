import json
with open('tcia_series_list.json') as f:
    data = json.load(f)

for x in data:
    if x['PatientID'] in ['LUNGx-CT003', 'LUNGx-CT004', 'LUNGx-CT005']:
        print(x['PatientID'], x['ImageCount'], f"{int(x['FileSize'])/1024**2:.1f} MB")
