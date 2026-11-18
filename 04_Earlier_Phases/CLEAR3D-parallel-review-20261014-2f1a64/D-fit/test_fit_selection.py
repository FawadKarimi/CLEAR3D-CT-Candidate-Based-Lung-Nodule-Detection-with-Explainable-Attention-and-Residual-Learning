import unittest
from fit_selection import select_fit


class Tests(unittest.TestCase):
    def setUp(self):
        self.entries = [{'scan': {'scan_id': x, 'patient_id': p}} for x, p in [
            ('fit', 'p1'), ('dev', 'p2'), ('confirm', 'p3'),
            ('excluded', 'LIDC-IDRI-0138')]]
        self.split = {'fit': ['p1', 'LIDC-IDRI-0138'], 'dev': ['p2'],
                      'confirm': ['p3'], 'excluded_reviewed': []}
        self.rows = {i: {'scan_id': 'fit', 'label': int(i < 240), 'id': str(i)} for i in range(960)}
        self.indices = list(range(960))

    def test_filters_old_dev_and_duplicate_patient_preserves_draw_order(self):
        self.rows[0]['scan_id'] = 'dev'
        self.rows[1]['scan_id'] = 'excluded'
        self.rows[240]['scan_id'] = 'confirm'
        out = select_fit(self.indices, self.rows, self.entries, self.split)
        self.assertEqual(out[:4], [2, 3, 4, 5])
        self.assertEqual(out[4:], list(range(241, 253)))

    def test_overlapping_split_rejected(self):
        split = dict(self.split, dev=['p1'])
        with self.assertRaises(ValueError): select_fit(self.indices, self.rows, self.entries, split)

    def test_scan_mapping_ambiguous_rejected(self):
        with self.assertRaises(ValueError): select_fit(self.indices, self.rows, self.entries + self.entries[:1], self.split)

    def test_unknown_patient_scan_and_bad_label_rejected(self):
        self.rows[0]['scan_id'] = 'missing'
        with self.assertRaises(ValueError): select_fit(self.indices, self.rows, self.entries, self.split)
        self.rows[0]['scan_id'] = 'fit'
        self.rows[0]['label'] = 0
        with self.assertRaises(ValueError): select_fit(self.indices, self.rows, self.entries, self.split)
        entries = self.entries + [{'scan': {'scan_id': 'unknown', 'patient_id': 'absent'}}]
        self.rows[0]['scan_id'] = 'unknown'
        with self.assertRaises(ValueError): select_fit(self.indices, self.rows, entries, self.split)

    def test_insufficient_positives_and_repeated_draws(self):
        indices = [0]*240 + list(range(240,960))
        with self.assertRaises(ValueError): select_fit(indices, self.rows, self.entries, self.split)
        indices = [0, 0, 1, 1, 2, 2, 3] + [3]*233 + list(range(240,960))
        self.assertEqual(select_fit(indices, self.rows, self.entries, self.split)[:4], [0,1,2,3])


if __name__ == '__main__':
    unittest.main()
