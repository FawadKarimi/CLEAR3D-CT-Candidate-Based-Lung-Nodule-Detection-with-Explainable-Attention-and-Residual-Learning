"""Select the first distinct training candidates eligible under a newer split.

Pure metadata helper: callers must independently bind the split, selection,
row store, draw plan and this file by hash before using the returned indices.
"""

ROLES = ('fit', 'dev', 'confirm', 'excluded_reviewed')
EXCLUDED_PATIENTS = frozenset(('LIDC-IDRI-0138', 'LIDC-IDRI-0029'))


def select_fit(indices, rows, entries, split):
    if len(indices) != 960 or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('expected 960 nonnegative integer draw indices')
    if not isinstance(split, dict) or any(k not in split or not isinstance(split[k], list) for k in ROLES):
        raise ValueError('missing patient split role')
    roles = {}
    for role in ROLES:
        for pid in split[role]:
            if not isinstance(pid, str) or not pid or pid in roles:
                raise ValueError('invalid, duplicate or overlapping patient')
            roles[pid] = role
    scan_to_patient = {}
    for entry in entries:
        scan = entry['scan']
        sid, pid = scan['scan_id'], scan['patient_id']
        if not isinstance(sid, str) or not sid or sid in scan_to_patient:
            raise ValueError('missing or ambiguous scan mapping')
        if not isinstance(pid, str) or not pid:
            raise ValueError('missing patient mapping')
        scan_to_patient[sid] = pid
    result = []
    for draws, label, count in ((indices[:240], 1, 4), (indices[240:], 0, 12)):
        selected = []
        seen = set()
        for index in draws:
            row = rows[index]
            if row.get('label') != label or type(row.get('label')) is not int:
                raise ValueError('draw section disagrees with row label')
            sid = row.get('scan_id')
            if sid not in scan_to_patient:
                raise ValueError('row has unknown scan')
            patient = scan_to_patient[sid]
            if patient not in roles:
                raise ValueError('row patient absent from current split')
            if roles[patient] != 'fit' or patient in EXCLUDED_PATIENTS:
                continue
            if index not in seen:
                seen.add(index)
                selected.append(index)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError('insufficient distinct eligible fit rows')
        result.extend(selected)
    if len(set(result)) != 16:
        raise ValueError('candidate duplicated across classes')
    return result
