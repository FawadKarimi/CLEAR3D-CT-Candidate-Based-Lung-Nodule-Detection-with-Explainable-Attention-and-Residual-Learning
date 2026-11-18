# Image-equivalence exclusion derivative — implementation proposal

Status: HOLD. This patch is prepared for independent review only. It is based on commit c118d7b5570339a83ab845fd664a2c0eb48373f7 and has not been applied to the live source tree. No real migration, freeze, threshold selection, training, inference or publication was executed. The accepted 799-scan preparation remains the authoritative preserved parent. Synthetic tests alone exercised the implementation.

## Population policy

Exclude BOTH scan_6ed2b75c6884e97fa7de (LIDC-IDRI-0138, subset4) and scan_c32a1a3ce4c68dae6b44 (LIDC-IDRI-0029, subset3) from a separately versioned derivative. Reason: confirmed image equivalence. This is NOT biological-patient identity adjudication. Neither recorded identity nor reference list is merged. Both excluded records and all original artifacts remain in the parent and investigation evidence.

Expected development population:797 scans,796 distinct recorded patient IDs,708 train,89 validation. Included reference rows:1074 (946 train,128 validation); excluded rows:31847 (28703 train,3144 validation). No remaining reference object, coordinate, diameter, radius, row ID, duplicate multiplicity or subset assignment changes. The full assignment-metadata population becomes886 series/885 recorded patient IDs. These are identifier counts, not established biological-patient counts.

Expected retained proposals:36183150 = 36280516 - 48683 - 48683, from accepted checkpoint counts. No real proposal file was read to author this proposal. The migration would concatenate retained per-scan proposal streams in original scan order and verify every row ID, scan ID and checkpoint count using the existing verifier. It never materializes the aggregate pool.

## Review-bound ledger and input hashes

The package's exclusion-ledger.PROPOSED.json records both exact scan/study/series/patient identities, their shared pixel fingerprint, individual record hashes, canonical hashes and counts of both reference lists, the investigation report/ZIP hashes, original accepted preparation hashes, input manifest/protocol/assignment hashes, expected counts and unresolved findings. It is a proposed implementation input, not reviewer approval. Its SHA-256 must be supplied explicitly when a future authorized migration is invoked. The ledger itself, its source evidence and parent metadata are rechecked; the ledger is checked again before copy and before the new preparation marker is committed.

Original scans.jsonl:49d9df4826a505622359743d6b21519b6b31b98b6758c70619252071e18a3abf.
Original development manifest:6944f057a269d5d52c973763d46b670c4028ffb73c06760e99cceaf524c0b963.
Accepted integrity report:f31bacd388d390b54b6944d1fd62876ca3359364e218858fcc88307734952a6c.
Investigation report:e9830336f49178714d403d55c1986e6c12f40601506cdf46d2c73a9327abd7b9.
Investigation ZIP:6ab6661ae0821ce4dedff1099493cc6b39923044be0a4ec16272ebdc025fd7c1.
Additional exact hashes are in the ledger and package manifests; no future output hash is fabricated.

## Files reused versus rewritten

| File | Proposed operation | Binding/check |
|---|---|---|
| Parent799 preparation, raw sources and existing evidence | Read-only; preserve | Original SHA-256, metadata and Git checks; no rename/delete/edit |
| Retained image.npy, valid.npy, lung_mask.npy, base_mask.npy | Byte-copy or Linux reflink-auto copy | Fresh source/destination SHA-256 and size equal parent checkpoint; separate inode |
| Retained geometry.json | Byte-identical copy | Same parent hash; coordinate conventions unchanged |
| Retained proposals.jsonl | Byte-identical streamed copy | Same parent hash; IDs/order preserved |
| Retained record.json | Rewrite only four prepared-array paths | All other fields preserved; exact new record hash |
| Retained checkpoint.json | Rewrite | New preparation key, same cohort-record hash, new record/file inventory, parent checkpoint hash and ledger ancestry |
| cohort_development.jsonl | New filtered ordered manifest | Exactly retained original records; canonical cohort hash; never a withheld manifest |
| assignments.csv | New filtered assignment table | Same fields and values for all remaining series; original subset correction retained |
| protocol.json | New schema3 derivative protocol | Exact role memberships/IDs/hashes/counts for all10folds; no inherited approval flag |
| exclusion-ledger.json,release-holds.json,reuse-map.json | New explicit provenance | Hash-bound into plan and preparation metadata |
| preparation_plan.json | New derivative plan | Parent key, ledger/protocol/code/config/cohort bindings; new canonical preparation key |
| scans.jsonl | New aggregate of rewritten records | Exact record equality and order checks |
| proposal_pool.jsonl | New concatenation of retained proposal streams | Streaming row equality/count/order checks, new aggregate SHA-256 |
| preparation.json | New final commit marker, written LAST | Existing artifact verification plus derivative metadata/count/ancestry checks |

No hardlinks or symlinks are used. Reflink-auto uses cp --reflink=auto and falls back to a separate byte copy; it does not guarantee storage savings on the university filesystem. Copy mode needs additional storage for all retained arrays and proposal copies plus the new aggregate. No image loading/resampling/mask computation or candidate generation is performed by migration. Raw DICOM bytes may be rehashed by the unchanged integrity verifier during an AUTHORIZED future migration; no DICOM pixels are decoded. Image arrays are reused as bytes, not regenerated.

## All-ten-fold recalculation

The implementation filters the assignment rows for the two series, then computes each role's full series UID list, recorded patient-ID list, counts and canonical membership hash from the original protocol's explicit subset lists. It verifies a complete role partition and original counts. Metadata-only expected identities/counts are supplied in expected-protocol-identity-checks.json. The original protocol is preserved; derived schema3 cannot be passed to the old schema2 LUNA16 cohort builder. That rejection is intentional: this is a distinct population, not a falsely approved official protocol.

For folds0,1,5,6,7,8,9,remove two training entries. Fold2 removes one training entry (0138) and one validation entry (0029); fold3 removes one validation entry (0138) and one test-role entry (0029); fold4 removes one test-role entry (0138) and one training entry (0029). The fold0 withheld population is unchanged. Test-role identity mappings for these known development scans do not involve opening withheld CT/labels. Other-fold CT preparation/migration is outside this implementation's execution scope: the entry point requires a successful fold0 development parent.

No stale candidate-class counts are copied into the new protocol. Reference-row deltas follow the excluded records (33 and31 excluded,0included each). Full withheld-label totals are not independently recomputed or claimed. Full identity and series/patient counts use assignment metadata only.

## Duplicate and release gates

The existing verify_bundle identity rules are extracted into verify_scan_identities without weakening them; verify_bundle still checks declared file hashes and calls the same rules. freeze_bundle calls these rules before threshold selection; direct select_candidates also rejects duplicate images first. This rejects duplicate scan IDs, patient-role conflicts, missing fingerprints and repeated fingerprints even within one role.

Derivative verification adds exact ledger/protocol/assignment/manifest/plan/reuse-map/hold-file hashes; checkpoint ancestry; retained cohort fields; array paths; complete ten-fold role membership and counts; exclusion absence; and the new plan key. It supplements the original full artifact/source/proposal verifier. Removing derivative ancestry from a preparation marker is rejected while checkpoints declare ancestry.

A future freeze still needs the existing exact-scans approval and reviewer/audit fields. A derivative additionally needs review.preparation_sha256, review.derivative_binding_sha256 and nonempty attributable resolution evidence for EVERY carried hold. Future frozen bundle file inventories include derivative metadata and preparation hashes. No such approval artifact is supplied here. These fields are content-binding gates, not a digital-signature or reviewer-authenticity system; an accountable independent review process is still required.

## Findings that remain open

The excluded pair alone is addressed. The19excluded-center outliers are unchanged; CSV-origin proof remains inherited rather than a fresh authenticated development-only row comparison; the irregular-spacing series and source-scope policy still need review. Complete attributable prior-test-access disclosure is missing: previous full-cohort annotation parsing, source inventory/audits and26previously decoded repeated-position images are documented, with withheld overlap unresolved. The package retains the prior review verbatim and a six-item unresolved ledger. No new withheld CT/labels were accessed or flags changed.

## Modified population and evaluation disclosure

This is an independent LIDC3D derivative of LUNA16, with886assignment series instead of888 and797fold0development scans instead of799. It retains the preexisting patient-grouped subset correction for patient0332 rather than reverting to official subsets. Exclusions can change sample weighting and false-positive denominators even though neither excluded scan has included nodules. Exclusion-region differences are not merged into another series. Report exact population/assignment hashes and per-role counts with every future result.

The custom one-to-one maximum-cardinality matching, duplicate-detection FP handling, empirical-staircase CPM, patient bootstrap and cap400 are not equivalent to official LUNA16 evaluation. Negative excluded diameters retain their radius5mm semantics and strict-radius flags. No claim of direct official benchmark comparability or biological identity resolution is justified. This document supersedes incompatible XML/70-15-15 design descriptions only for this proposed derivative; it does not silently change the project-wide approved protocol.

## Migration sequence — NOT EXECUTED

1. Independently review the exact patch, ledger hash, synthetic results, population policy, storage plan and unresolved findings. Obtain separate authorization to apply the patch and execute migration; release approval is a separate decision.
2. Recheck the pinned base commit and source hashes; apply the reviewed patch in an authorized checkout. Preserve the original preparation and all previous audit files. Do not run prepare_candidates/resume on the derivative; its deliberately different plan represents reuse, not a new image-preprocessing run.
3. Reserve a new versioned output path, for example artifacts/luna16-development-image-equivalence-exclusions-v1/prepared. Confirm adequate free space; no existing destination may be reused. Source/config/protocol changes require a new reviewed proposal.
4. Only AFTER authorization, use scripts/migrate_image_exclusions.py with --parent, --ledger, --ledger-sha256, --output, --reuse-mode and explicit --execute. The package intentionally does not execute this command or provide an approval file. Without --execute the entry point fails before I/O migration; ledger hash is mandatory.
5. Verify retained-file hash equality, counts, complete proposal aggregation, all-fold metadata, parent preservation and absence of excluded IDs. Capture invocation/environment/log/exit status, complete new hashes and a fresh acceptance report. New record/checkpoint/scans/preparation hashes depend on the chosen output path, ledger/code bindings and exact serialization and are not precomputed here.
6. If copying/verifying fails, no final preparation.json marker is published; leave the incomplete NEW directory for diagnosis and choose a new path for a later separately reviewed attempt. Never repair by altering the parent, weakening a guard or forging approval. The implementation is deliberately non-resumable; storage cleanup requires a separate decision.
7. Submit migration evidence and unresolved reference/test-access resolutions for independent review. Freeze, validation threshold selection, training, inference and publication remain separately gated and are not authorized by a successful migration.

## Reproduction limits

Synthetic tests run only against temporary toy source bytes, arrays and references. They exercise copy/reflink semantics on this filesystem, not full-cohort I/O time/storage, source availability at future execution, or every real checkpoint's fields. The final patch dry-apply is checked against an isolated copy of the exact baseline. No real migration dry-run or real proposal aggregation was performed. Outputs with new hashes do not yet exist.
