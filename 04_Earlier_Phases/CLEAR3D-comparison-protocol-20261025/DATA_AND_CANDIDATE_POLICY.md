# Data and candidate policy

Status: DRAFT. The population is not adopted.

Do not overwrite luna16-prepare-development-fold0-v1. Write a new versioned manifest. Leave the live LIDC3D checkout unchanged.

A reference nodule is eligible when its center is inside the volume and the prespecified inclusion flag is true. Small, subtle, ground-glass, pleural, and vessel-adjacent nodules stay in. The brightness component test in redesign.py is not part of this benchmark.

Candidates come from one frozen coordinate file. Ignored rows are counted. A patient belongs to one split, and every series of that patient stays with the patient. Repeated crops are training views, not extra positive cases. Reference-centred augmentation is training-only and is not added to inference candidates.

After adoption, sample only training patients, declare a patient weight, count unique nodules separately from repeated draws, and mine hard negatives only from training patients. Do not use the 83 unsettled val patients, do not retrieve absent LUNA series, and do not tune on the redesign named test or the 384-row slice.
