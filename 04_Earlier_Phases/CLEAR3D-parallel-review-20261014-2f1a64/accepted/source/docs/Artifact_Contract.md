# Phase 3 return package

Return a directory or archive retaining the following files. Do not return screenshots alone.

| Artifact | Required content |
| --- | --- |
| environment.json | Python, CUDA, GPU, package versions and run device |
| source inventory | source path, DICOM/XML header summary, file hashes, reason-coded exclusions |
| cohort manifests | scan/patient/series IDs, independent split, pixel fingerprints, reference review and access history |
| candidate artifacts | resolved LoG/mask config, threshold search trace, candidates, mask/proposal diagnostics and failures |
| per-run folder | exact config, source hash, architecture signature, seed, epoch history, best checkpoint hash, selection record |
| saved predictions | candidate IDs, LPS centers, raw logits, scores, scan status and inference provenance |
| evaluation files | evaluation cases, FROC threshold points, CPM, frozen operating threshold, candidate summaries |
| statistics | paired-bootstrap draws/intervals, seed summaries, optional permutation and Holm results |
| XAI folder | selected IDs, maps, IG residuals, localization/agreement/perturbation/stability/randomization records, failures |
| timings | preprocessing, proposal, training, inference and XAI timing protocol and raw samples |

Preserve null or undefined values and failure records. State the exact run IDs that are confirmatory H16, C16 and C32. Include the final resolved configuration and test-release record. No outcome will be converted into thesis prose until these artifacts are internally consistent and reviewed.
