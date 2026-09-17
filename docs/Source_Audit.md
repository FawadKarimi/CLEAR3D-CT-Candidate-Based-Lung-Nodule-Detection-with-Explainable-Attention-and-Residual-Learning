# Read-only source and annotation linkage audit

`audit-source` diagnoses an existing LIDC3D header inventory against a SQLite annotation database. It does not reconstruct volumes, resolve ambiguous slices, create labels or exclusions, freeze a cohort, generate candidates, release tests, or train a model. Every run sets `source_approved: false` and `training_released: false`. A completed audit means diagnostic operations completed, including the recording of unresolved findings. It is never source approval. Research audit and any slice-selection policy need separate review.

The module is independent of ConsenDxFormer and LungDxFormer. It uses neither their code nor their derived artifacts. The installed annotation database is an input to inspection, not an approved replacement for the thesis XML reference standard. Neither annotation IDs nor annotation counts are treated as reader identities or distinct nodule counts. Reader identity and coverage of small-nodule and non-nodule annotations remain explicitly unresolved; no negative labels are created.

## Invocation and safe operation

Use a new output directory outside the raw data, annotation database/package, configuration, and code/package source trees. The command accepts the existing `inventory` JSON shape (`source_root`, `slices`, `errors`) without requiring a regenerated or fully hashed archive. Input paths in the inventory must be absolute paths valid on the host executing the audit. Use `configs/source_audit.yaml`, not the training reference configuration. Relative CLI arguments resolve from the recorded working directory.

Proposed university server invocation — **NOT EXECUTED**:

```bash
cd /experiment-storage/chaidir/fawad-dsai/LIDC3D
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  /experiment-storage/chaidir/env/env-310-fawad/bin/python -B -m lidc3d.cli audit-source \
  --inventory artifacts/inventory.json \
  --annotation-db /experiment-storage/chaidir/env/env-310-fawad/lib/python3.10/site-packages/pylidc/pylidc.sqlite \
  --config configs/source_audit.yaml \
  --out artifacts/source-audit-20260917-01
```

The installed console-script equivalent is `lidc3d audit-source` with the same four options. The source-tree invocation avoids installing/upgrading the project or changing shared packages. `-B` and `PYTHONDONTWRITEBYTECODE=1` suppress Python bytecode-cache writes, including in package trees. No pylidc import, configuration read/write, loader call, installation modification, or global NumPy patch occurs. Package versions are obtained through distribution metadata, without importing PyTorch or pylidc.

The observed server PyTorch 2.3.0+cu121 is below this project's declared `torch>=2.4` dependency. This is a separately reported environment incompatibility; nothing here changes the constraint or upgrades that environment. The audit itself imports NumPy, pydicom and PyYAML, plus Python standard-library modules, and does not use PyTorch. The reported server Python 3.10.17 / NumPy 2.2.6 / pydicom 3.0.2 / pylidc 0.2.3 combination was **not tested** in this implementation task.

SQLite is opened via a file URI with `mode=ro`, `query_only=ON`, `trusted_schema=OFF`, memory temporary storage, and a read transaction. Existing WAL, SHM or journal sidecars cause failure; use a quiescent source and have its owner handle any recovery separately. The audit does not recover or clean a database. The recorded database hash is recomputed from the actual input; the previously observed `995989985bb17106808c40572ccac2ce0b6434b91283d4f773cdb967d47443cb` is context, never a hard-coded acceptance criterion. Previously reported counts and patients are likewise not acceptance rules.

The output directory must not exist (including a dangling symlink). The tool checks lexical paths, resolved paths and resolved ancestors and protects the raw root, all inventoried DICOM parent directories and their resolved targets, the database's parent, the configuration's parent, the LIDC3D source tree, Git metadata, Python package/stdlib trees, and additional absolute `protected_roots` in the audit configuration. Every artifact uses exclusive file creation. Git status uses `--no-optional-locks` to avoid index refresh writes. Inventory, database, and configuration hashes are compared before/after. Each inspected DICOM is hashed before/after its read. Concurrent writers are unsupported: final hashes detect persistent input changes but cannot prove absence of transient changes that are reverted. Path checks do not provide a security boundary against a hostile concurrent symlink replacement. Read-only mount/permissions and a quiescent source remain useful operating conditions.

Normal CLI execution prints three summary lines; preflight exceptions print two. Successful diagnostics exit 0 even with ambiguous links or conflicting images. Decoding/read/schema/integrity failures, inspected inventory contradictions, invalid inclusion values, changed inputs, and artifact write failures exit 1. Details belong in files. Unsafe/existing output, unreadable initial inputs, or invalid configuration fail before output creation. Failures after safe directory creation normally retain all five artifacts with `audit_status: failed`; JSONL files can contain partial findings. Disk/write failures or process termination may leave an incomplete set. `audit_summary.json` is written last; absence of it or any of the other four artifacts is never completion. Do not retry into a partial directory.

## Diagnostic rules

All tolerances are explicitly required and recorded in provenance. Defaults: repeated full position componentwise absolute tolerance `1e-6 mm`; LPS z absolute tolerance `1e-4 mm`; orientation absolute tolerance `1e-4`; in-plane spacing absolute tolerance `1e-6 mm`; in-plane drift `0.1 mm`; slice spacing absolute tolerance `0.1 mm` plus relative `0.01`; HU absolute tolerance `1e-6` with relative tolerance zero. Values must be finite and nonnegative; orientation tolerance must be less than 0.1 to support a nondegenerate normal. These are diagnostic settings and do not amend ingestion or define acceptance/release criteria.

Identity comparisons use exact patient/study/series strings in both directions. Differences are also reported at patient and patient/study levels. Duplicate database identities and SOP UIDs are retained and reported. No series or slice is selected or excluded.

Every series, including equal-count series, compares all database zval rows against all finite CT LPS z values. Each zval lists every CT record within tolerance; the reverse uncovered CT list is also recorded. Sorted multisets compare multiplicity when counts agree; agreement is diagnostic, not unique identity. Duplicate scan identities are compared separately rather than merged into a fictitious scan. DB-only and CT-only series remain visible.

`ImagePositionPatient[2]` (LPS z) and `dot(ImagePositionPatient, normal)` are distinct output fields. The normal is the normalized cross product of the two DICOM orientation vectors. Database z-values are compared only to LPS z. Nonfinite/invalid position, invalid/nonorthonormal orientation, nonaxial geometry, variable orientation/spacing/dimensions, localizers, in-plane drift and irregular spacing are flagged; no reconstruction or fallback sort is attempted. Geometry for the full archive comes from inventory headers. Frame count and pixel encoding are validated only on repeated-position files because those fields are not in the existing inventory contract.

Repeated positions use all three LPS components within the configured absolute tolerance. Groups are connected components, preserving tolerance chains; a component spanning more than tolerance on any axis is explicitly `tolerance_chain_ambiguous`. Sharing only z does not create a repeated-position group. The group and member ordering never selects a preferred image.

Only these groups are decoded. Each member retains original/resolved file paths, before/after SHA256, SOP/identity, spatial tags, InstanceNumber, available acquisition/reconstruction identifiers, rescaling, padding, encoding, warnings, and errors. Missing tags are JSON null. Raw decoded arrays are compared exactly and independently of rescaling. Padding masks exclude the inclusive interval from PixelPaddingValue to PixelPaddingRangeLimit (either direction). Padding-mask equality is separate from raw equality. HU is computed in float64 from an explicit finite positive slope and finite intercept, using CT HU when RescaleType is absent; a non-HU type or LUT/real-world mapping is unsupported. HU comparison uses only the intersection of valid pixels and records its size and maximum absolute difference. Unequal validity masks can coexist with equal HU over that intersection. Empty intersections, errors, or unsupported transforms yield null comparisons, never a vacuous equality. No averages, replacements, or choices by InstanceNumber/file order/contour similarity are made.

## SQLite schema and linkage

The actual `sqlite_master` schema and required-table `PRAGMA table_info` are recorded on each run. Required ordinary tables are `scans`, `annotations`, `contours`, and `zvals`, each with a sole `id` primary key. The supported columns/affinities are:

- `scans`: integer `id`; text `patient_id`, `study_instance_uid`, `series_instance_uid`.
- `annotations`: integer `id`, `scan_id`.
- `contours`: integer `id`, `annotation_id`; integer/boolean `inclusion`; real/float/numeric `image_z_position`; text `dicom_file_name`.
- `zvals`: integer `id`, `scan_id`; real/float/numeric `val`.

Extra columns are preserved in schema metadata and do not imply supported reader/coverage semantics. Views or virtual tables cannot replace required tables. `quick_check` plus explicit identity/value/orphan checks guard the used data. Contours are left-joined to annotations and scans so broken relations cannot disappear through an inner join. Orphans or invalid scan/zval identities fail the audit with details in provenance, before linkage results are claimed.

Adapter column names were cross-checked against upstream pylidc v0.2.3 [Scan](https://github.com/pylidc/pylidc/blob/v0.2.3/pylidc/Scan.py), [Annotation](https://github.com/pylidc/pylidc/blob/v0.2.3/pylidc/Annotation.py), [Contour](https://github.com/pylidc/pylidc/blob/v0.2.3/pylidc/Contour.py), and [Zval](https://github.com/pylidc/pylidc/blob/v0.2.3/pylidc/Zval.py) definitions. This is upstream reference information, not inspection of the university's installed code. The tool does **not** inspect installed pylidc source or claim its loader behavior; `installed_pylidc_source_inspected` is false. If installed-source inspection is added later, it must record the inspected bytes' hashes and separately describe actual code behavior. No slice-selection routine is executed.

Each contour preserves `dicom_file_name`, `image_z_position`, and `inclusion` under `original`, including null/unusual values. Candidate lookup is scoped to the matching patient/study/series. Filename candidates require the **entire unchanged string** to equal an inventoried basename, absolute path, or raw-root-relative POSIX path. Literal SOP candidates require the entire unchanged string to equal an inventoried SOP UID. The tool does not remove `.dcm`, parse a filename as a UID, normalize case, strip directories, or use fuzzy/similarity matching. Filesystem names and literal UIDs remain separately attributed candidate sets. A missing exact link is not repaired using position.

Classification precedence:

1. `filename_position_contradiction`: any exact candidate lies outside the contour's finite z tolerance; retain every candidate, including potentially consistent alternatives.
2. `ambiguous_exact_link`: multiple exact candidates, duplicate database identity, or a nonunique SOP UID anywhere in the inventory. Position never breaks the tie.
3. `exact_link_source_unverified`: an exact candidate's inspected source could not be fully inspected or contradicted its inventory entry.
4. `exact_link_geometry_unresolved`: exact evidence exists but contour z or series geometry cannot support the positional check.
5. `verified_exact_inventory_link`: one exact candidate, a unique SOP UID, one matching database scan identity, consistent supported diagnostic geometry, and compatible contour z. Verification is **only correspondence to the hashed inventory**; it does not authenticate the original XML or cryptographically verify an uninspected file.
6. `z_only_unverified`: at least one positional candidate but no exact identity evidence. This stays unverified even with exactly one candidate.
7. `missing_link`: no exact or positional candidate.

Every repeated group reports `neither`, `one`, or `multiple` separately for exact candidate members and verified members. It also lists z-only candidates and per-contour/annotation references, preserving inclusion/exclusion flags. “Neither” means no such reference in this database under these rules, not a verified absence in XML. A filename collision can produce multiple exact candidates and zero verified members. These results do not establish reader counts or nodule counts.

## Artifact schema version 1

All five files are required. JSONL objects each carry `schema_version: 1`; the two JSON documents carry the same field. Numbers are finite JSON numbers; unsupported nonfinite input values are retained as `{"nonfinite": "nan"}` (or the corresponding infinity string). Null comparison results mean unavailable, not false or equal. Record IDs are 1-based ranks of canonical inventory rows sorted by identity, source path, SOP UID and canonical row content; they identify inventory records, not selected slice indices.

- **`audit_summary.json`**: final `audit_status` (`completed`/`failed`), always-false approval/release flags, fixed coverage limitations, counts, bidirectional identity differences (arrays of exact identity tuples), duplicate SOP and database identity registers, preserved inventory errors, contour classification counts, and operational errors. Database annotations/contours/zvals and inspected DICOM records are counted separately. Inventory errors remain findings and limit completeness, even if this diagnostic run completes.
- **`series_audit.jsonl`**: one row per union patient/study/series identity. `identity`, `identity_status`, `ct_count`, all `database_scan_ids`, annotation count, linkage counts, repeated-group IDs, duplicate SOP IDs; `geometry` contains status/findings and every record's source path, SOP, full position, LPS z, normal and projection; `z_comparisons` contains each scan ID's CT/DB counts/delta, sorted-multiset flag, all zval candidate lists, uncovered CT/zval IDs and ambiguous zval IDs.
- **`duplicate_positions.jsonl`**: one row per repeated full-position connected component. `group_id`, `identity`, `position_span_lps_mm`, tolerance-chain flag, unresolved resolution, `members` with metadata/hash/read/decode evidence; `pairwise` with record-ID pairs, raw/padding/HU equality (boolean or null), common valid count, max HU difference and comparison status; `annotation_references` with all matching contour IDs, annotation IDs, inclusion flags, candidate/verified member IDs and cardinalities. Member IDs join to the series position records.
- **`contour_linkage.jsonl`**: one row per joined contour in a schema-valid database. `contour_id`, `annotation_id`, `scan_id`, `identity`, unchanged `original` fields, separate filename/literal-SOP/z candidate ID lists, their union, contradictory candidate IDs, identity-ambiguity flag, classification `status`, nullable `verified_record_id`, inclusion validity and unresolved reader identity. Schema/integrity failure may leave this file empty; summary status controls completeness.
- **`provenance.json`**: UTC start/end, status, exact invocation and cwd, full audit configuration, Python/platform/package versions, SQLite mode/schema/columns/integrity findings, protected paths, each input path and before/after hash/unchanged flag, Git revision/dirty/status and source-file hashes, explicit limited DICOM hashing scope and no-installed-source-inspection/no-loader flags, plus hashes of the three JSONL files. Git unavailable is recorded as null plus an error rather than an invented clean revision. The summary is written after provenance as the final marker.

Order is deterministic for identical inputs: identity tuples, then canonical record/group IDs; database IDs order zvals and contours; candidate IDs, schema objects, flags and error iteration are stable. Provenance timestamps, paths, invocation, code dirty state and hashes intentionally vary with execution/context. The JSONL findings are byte-identical for reordered inventory inputs containing the same rows at the same paths.

## Remaining limits and verification

The header inventory may be stale or incomplete. Only repeated-position files are read/hashed and rechecked against it; there is no full-archive cryptographic verification, directory rescan, XML parsing, or general header reread. Decoder and header failures remain explicit. Geometry support is conservative; oblique/tilted or varying series need separate analysis. LUT-based intensity transforms are not interpreted. The audit does not infer what the installed loader retained when generating its database, and does not prove database filename provenance or XML completeness.

Tests use synthetic uncompressed DICOMs and temporary SQLite databases only. They exercise missing/matching series, equal-count position mismatch, repeated images, separate rescale/padding behavior, ambiguous/missing/z-only/exact links, collisions scoped by series, unsupported schema/geometry, decoding errors, read-only inputs, SQLite write refusal, overwrite/symlink protections, limited decoding/hashing, input tampering, stable ordering and CLI status/line budgets. Compressed decoder availability, production dataset scale, Linux filesystem behavior, real database variants, university paths and clinical evidence are not established by these tests. No server run is claimed.

Use a fresh temporary directory for each test invocation (pytest may otherwise remove an existing `--basetemp`). Actual local verification commands/results and the full review patch are delivered in the task handoff. Existing scientific modules and rejection rules remain unchanged.
