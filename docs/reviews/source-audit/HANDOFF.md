# LIDC3D source audit correction handoff

## Current correction candidate: schema v2

This correction starts from `55f65753bf96cc4667ea96679c112019e45df67e` on `audit/source-annotation-linkage` in `FawadKarimi/LIDC3D`. It is a new descendant commit for review, not an amendment to that published commit. The containing GitHub commit identifies the exact candidate revision. The branch is pushed normally without force; main is not modified or merged. The prior baseline remains `abf35d8b566791e309cc85972389344069159be7`.

No university-server execution, real patient-data audit, research training, candidate generation, cohort freeze, test release, deployment, or source approval occurred. No packages were installed/upgraded and no shared configuration was changed. The existing full suite includes its pre-existing tiny synthetic training/resume unit test; that is software verification only.

Changed files in this correction (5):

1. `src/lidc3d/source_audit.py`
2. `tests/test_source_audit.py`
3. `docs/Source_Audit.md`
4. `docs/reviews/source-audit/HANDOFF.md`
5. `README.md`

### Corrected evidence semantics

Output artifacts now use `schema_version: 2`; the unchanged audit configuration still uses version 1. Ordinary exact links use `exact_inventory_record_link_source_not_rechecked`, with an `exact_inventory_record_id` but no `verified_inspected_record_id`. A source file can receive `verified_inspected_source_link` only if it was actually opened in this audit, its identity and position agree with the inventory and matching scan/contour context, the link is unique, its before/after file hashes match, and all inspected-member and group-comparison evidence is complete and consistent. No ordinary-slice DICOM read, hash or pixel decoding was added.

The inspected-source state requires supported inventory and actual group geometry and available/equal raw-pixel, padding-mask and valid-HU comparisons. Pixel conflict, unavailable comparison, tolerance-chain ambiguity, identity/hash/read error or captured warning blocks that state. A clean-looking peer does not become a preferred slice when another member is uncertain. Every group retains all members and `resolution: unresolved_no_slice_selected`, including groups with fully consistent content.

Z-only evidence is `z_only_inventory_link`, never exact-source evidence even when only one positional candidate exists. Ambiguous/missing/contradictory/unsupported links remain explicit and separate. `exact_link_inspection_unresolved` replaces the old source-unverified category. Repeated-group fields explicitly say `verified_inspected_members`, `verified_inspected_member_cardinality` and `verified_inspected_record_ids`. Exact-candidate inspection details and blocker counts are retained separately.

`inventory_hash_scope` explicitly states that hashing the inventory establishes immutability of those JSON bytes during the audit, not cryptographic verification of every referenced DICOM. Old schema-v1 `verified_exact_inventory_link` claims must not be upgraded to the inspected-source state. Regenerate v2 artifacts in a new directory; no automatic or in-place schema migration is provided. The full classification precedence, fields and migration decision are in [Source_Audit.md](../../Source_Audit.md).

### Raw boundary and warnings

All inventory DICOM paths must be absolute, contain no `..` components, and resolve beneath the resolved raw `source_root`. Direct escapes and escaping symlinks fail before any DICOM is opened/hashed and before creating outputs. Contained symlinks are allowed. The guard is repeated immediately before inspection; reads/hashes use the validated resolved path. This is a quiescent-filesystem policy, not protection against hostile concurrent path replacement.

All Python warnings caught during DICOM read/tag access/pixel conversion are retained, along with WARNING-or-higher pydicom log messages. There is no harmless-warning whitelist. Any captured warning blocks inspected-source verification, including on otherwise equal pixel data. Warnings may still allow diagnostic completion; `source_approved` and `training_released` remain false in every artifact set. The scoped logging handler is removed after inspection.

### Verification actually executed for this correction

Environment: Windows 11, `Windows-11-10.0.22621-SP0`; CPython 3.13.7 (MSC v.1944, 64-bit AMD64); NumPy 2.3.2; pydicom 3.0.1; pytest 9.0.1; torch 2.9.1; SciPy 1.16.2; scikit-image 0.25.2; scikit-learn 1.7.2; PyYAML 6.0.2. No pylidc import/installation is needed: all annotation databases are temporary synthetic SQLite fixtures.

The exact successful PowerShell test commands, including output capture and exit-code propagation, were:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
py -3.13 -m pytest -q tests/test_source_audit.py --basetemp .stage/correction-targeted-r2 2>&1 | Tee-Object -FilePath .stage/correction-targeted-r2.log
exit $LASTEXITCODE
```

Exit 0, `61 passed in 22.63s`. Zero failures and zero skips. This includes every revised linkage state, actual pydicom pixel warnings, logger-only warnings, complete/failed/hash-changed inspections, direct actual-position checks, conflicting comparisons, and raw-source boundary cases. Actual escaping and contained file-symlink tests both ran on this host. Their only allowed skip is an actual OS permission/capability failure creating a file symlink (EPERM, EACCES, ENOSYS, ENOTSUP, or Windows privilege error 1314); unrelated errors fail the test. Directory aliases/junctions remain covered as well.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
py -3.13 -m pytest -q --basetemp .stage/correction-full-r1 2>&1 | Tee-Object -FilePath .stage/correction-full-r1.log
exit $LASTEXITCODE
```

Exit 0:

```text
........................................................................ [ 75%]
........................                                                 [100%]
96 passed in 26.76s
```

Zero failures, zero skips. The full suite includes all 61 audit tests and all 35 pre-existing tests. Each command used a fresh basetemp; these logs and synthetic files remain ignored local evidence, not patient artifacts or additional committed files.

The first correction-specific targeted attempt also remains recorded:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
py -3.13 -m pytest -q tests/test_source_audit.py --basetemp .stage/correction-targeted-r1 2>&1 | Tee-Object -FilePath .stage/correction-targeted-r1.log
exit $LASTEXITCODE
```

Exit 1, `2 failed, 52 passed in 19.50s`, no skips. One assertion expected no Python warnings during a logger-only test, but previously unclosed fixture SQLite connections emitted ResourceWarnings during collection by the interpreter. Fixture connections now use `contextlib.closing` as well as transaction context management; warnings are not filtered out of the audit. The second test's simulated escaping path initially caused the existing output-path guard to reject the output before the intended read-boundary guard ran. The external synthetic target was moved into its own directory, allowing the test to exercise the read guard. Additional identity/hash/geometry regressions increased the successful targeted count to 61.

```powershell
git diff --check
git diff --exit-code 55f65753bf96cc4667ea96679c112019e45df67e -- src/lidc3d/ingest.py src/lidc3d/training.py src/lidc3d/pipeline.py src/lidc3d/provenance.py src/lidc3d/cli.py configs/source_audit.yaml pyproject.toml
```

Both exited 0. `git diff --check` emitted only Git's Windows line-ending conversion warnings, with no whitespace errors. The second command confirms unchanged ingestion, training/release workflow, existing output helpers, CLI wiring, diagnostic configuration and package requirements.

### Review of read-only and research boundaries

Local manual diff review confirms the source audit opens SQLite only with URI `mode=ro` and `query_only=ON`. Raw files are read through the contained pydicom path and existing read-only SHA256 helper; there are no new source writes, dataset transforms, loader calls, package modifications, cohort filters, selection rules or release paths. Production writes remain limited to exclusively created output artifacts. Boundary regressions instrument both DICOM reads and hashes to assert that outside paths are never accessed. Existing read-only-input and SQLite-write-refusal tests remain in the passing suite.

Tests deliberately create/modify their own synthetic DICOM/SQLite fixtures, including a controlled hash-tamper test; those writes are test setup, not source-audit behavior. No data artifact, SQLite database, environment, archive, checkpoint, credential or server output is included in the publication. `source_approved` and `training_released` remain literal false values, and the test fixture checks those flags in all emitted result sets. Both repeated-position members are retained in the successful inspected-source case; every member is blocked in the conflicting-group case. There is no preferred-slice field or automatic conflict resolution.

### Remaining limitations

Only repeated-position DICOMs are opened/hashed. Other source evidence remains inventory-only, which cannot establish ordinary-file integrity or archive completeness. XML/reader identity, small-nodule/non-nodule coverage, database filename provenance, real production scale, compressed decoders, Linux filesystem behavior and the actual university environment remain untested or unresolved. All captured warnings are conservatively blocking, including a warning that might eventually prove harmless; no whitelist is inferred. Source verification is limited to the individual contour/file evidence and group checks, not global database zval alignment, cohort eligibility, reference-standard approval or permission to train. Concurrent writers and hostile path changes remain outside the supported operating assumptions.

The reported server PyTorch 2.3.0+cu121 remains below the unchanged project dependency `torch>=2.4`. The audit does not use/import torch and no package change is authorized or performed. The server command retained in the historical section below remains **NOT EXECUTED**; any future run must use a new output directory and separate review. No university-server command was run during this correction.

## Original implementation and first publication (historical)

The following record preserves the original schema-v1 work, test results and publication at `55f65753bf96cc4667ea96679c112019e45df67e`. Its schema/verification wording is historical and is superseded by the schema-v2 evidence rules above. Its earlier 74-test result is not the current 96-test result.

At the original implementation handoff, implementation and synthetic verification were complete, and no commit or push had been performed. This tracked copy accompanies publication of the review branch. No server audit, patient-data processing, cohort selection, research training, candidate generation, freeze, test release, deployment, or merge has been performed. The existing suite includes its previously present tiny synthetic training/resume unit test; this is software verification, not a research experiment. No environment packages or shared configuration were changed.

### Repository and patch

- Repository: https://github.com/FawadKarimi/LIDC3D
- Exact implementation base and HEAD at the original handoff: `abf35d8b566791e309cc85972389344069159be7`
- Branch: `audit/source-annotation-linkage`
- Initial workspace: empty Git repository, no HEAD, remote, tracked files or user changes. This difference was reported before editing. The expected GitHub main commit was fetched and checked out on the requested branch without reset/overwrite. There were no applicable AGENTS.md files in the workspace or its ancestors.
- Original complete patch: `.stage/source-audit-handoff/source-audit.patch` (local evidence, not committed). The published GitHub commit provides the reviewable diff.
- Patch SHA256: `c3e10a7b3a26a02e5243515671a47cc0bcbcaff34e810a456345b42a4ae564df`
- At the original handoff, all changes were uncommitted and the new files had Git intent-to-add entries. The publication request authorizes committing and pushing this candidate to the review branch only.

Implementation files (6), unchanged from the tested candidate:

1. `README.md`: link to audit operating guide.
2. `configs/source_audit.yaml`: explicit diagnostic tolerances and optional extra protected roots.
3. `docs/Source_Audit.md`: operating instructions, five artifact schemas, classification precedence, limitations, and proposed server command.
4. `src/lidc3d/cli.py`: eight lines of lazy command wiring.
5. `src/lidc3d/source_audit.py`: focused audit module.
6. `tests/test_source_audit.py`: synthetic DICOM/temporary SQLite tests.

Existing ingestion/scientific modules, rejection rules, and dependency constraints are unchanged. Patch statistics: six files, 1,315 inserted lines. The original handoff folder and local synthetic outputs remain ignored `.stage` artifacts. This publication adds only one tracked file: `docs/reviews/source-audit/HANDOFF.md`.

### Verification actually executed

Environment: Windows; CPython 3.13.7; NumPy 2.3.2; pydicom 3.0.1; pytest 9.0.1; torch 2.9.1; SciPy 1.16.2; scikit-image 0.25.2; scikit-learn 1.7.2; PyYAML 6.0.2. No pylidc package is installed locally; audit tests use only temporary SQLite fixtures.

PowerShell environment/setup used for the successful commands:

```powershell
New-Item -ItemType Directory -Path .stage -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE='1'
```

Targeted tests:

```powershell
py -3.13 -m pytest -q tests/test_source_audit.py --basetemp .stage/audit-tests-r4
```

Exit 0, `39 passed in 16.38s`. Captured output: `.stage/source-audit-handoff/targeted-tests.log` (local evidence, not committed).

Existing suite, with its files unchanged:

```powershell
py -3.13 -m pytest -q tests/test_evaluation.py tests/test_geometry_and_data.py tests/test_ingest_and_guards.py tests/test_models_xai.py --basetemp .stage/existing-tests-r1
```

Exit 0, `35 passed in 12.73s`. Captured output: `.stage/source-audit-handoff/existing-suite.log` (local evidence, not committed).

Final full suite, including all audit tests after the final Git read-only refinement:

```powershell
py -3.13 -m pytest -q --basetemp .stage/full-suite-final
```

Exit 0:

```text
........................................................................ [ 97%]
..                                                                       [100%]
74 passed in 20.77s
```

Captured output: `.stage/source-audit-handoff/full-suite.log` (local evidence, not committed). Each test invocation used a fresh basetemp. In the actual shell calls output was captured with `2>&1 | Tee-Object -FilePath <log>` and the pytest exit code propagated with `exit $LASTEXITCODE`.

```powershell
git diff --check
git apply --reverse --check .stage/source-audit-handoff/source-audit.patch
```

Both exit 0. No whitespace errors from the required `git diff --check`; Git printed its Windows LF-to-CRLF conversion warnings. Captured diff-check output: `.stage/source-audit-handoff/diff-check.log` (local evidence, not committed). Reverse application was checked only; no patch was applied to or removed from the worktree.

Earlier development attempts are not hidden:

- Initial targeted run: `1 failed, 26 passed in 11.96s`. The new padding fixture needed an explicit `SS` value representation before pydicom could serialize it; corrected in the fixture. Existing pytest temporary-directory cleanup also printed Windows directory-not-empty warnings.
- The next targeted run specified `.stage/audit-tests-r2` before `.stage` existed: `27 errors in 1.53s`, all fixture setup path errors. Creating `.stage` and using a new path resolved that invocation issue.
- Intermediate targeted run: `27 passed in 10.72s`; additional edge-case tests subsequently brought the final targeted count to 39.
- A packaging-only check with `git -c core.autocrlf=false diff --check` returned 2 because overriding the checkout's normal CRLF normalization treated existing CRLF text as trailing whitespace. The standard requested `git diff --check` passes; the exported patch uses normal Git normalization. Diagnostic output: `.stage/source-audit-handoff/diff-check-crlf-diagnostic.log` (local evidence, not committed). No Git configuration was changed.

### Schemas and rules

The full contract is in [docs/Source_Audit.md](../../Source_Audit.md). All five exclusively created artifacts are version 1:

- `audit_summary.json`: final diagnostic completion/failure, always-false source/training approval, identity differences, duplicate registers, counts, operational errors and coverage limits.
- `series_audit.jsonl`: union of CT/database series, all CT positions, separate LPS z and normal projection, per-scan count and positional comparisons, geometry flags and group links.
- `duplicate_positions.jsonl`: every repeated-position member, source hashes/metadata/errors, independent raw/padding/HU comparisons and annotation-reference cardinalities.
- `contour_linkage.jsonl`: unchanged contour filename/z/inclusion, joined IDs, all separate exact-filename/literal-SOP/z candidates, contradictions, status and nullable verified inventory record.
- `provenance.json`: input hashes before/after, config, versions, invocation, Git state/source hashes, actual SQLite schema, protected paths and limited DICOM hash scope.

Every series is audited, including equal-count series. Matching counts do not establish position or identity alignment. z-only candidates stay unverified, even if unique. Filename collisions retain all candidates; position cannot resolve the tie. Exact inventory matches require consistent identity/geometry and become unverified if inspected source evidence contradicts the inventory. Repeated-position groups remain unresolved, even with equal pixels. No slice is selected, averaged, excluded or converted to a negative label.

Normal CLI output is three lines; preflight exceptions are two. Read-only SQLite and input hashes are verified. Decoding/schema/read/integrity errors produce failure, not success. The summary is written last; incomplete artifact sets never represent completion.

### Limits and server handoff

Only repeated-position DICOMs are opened/hashed. Other source/position evidence comes from the inventory. The entire CT archive is not cryptographically verified. XML coverage, actual reader identity, small-nodule/non-nodule completeness and installed loader behavior remain unresolved. Installed pylidc source is not inspected or executed. Linux filesystem behavior, compressed decoders, real SQLite variants, patient-data scale and server compatibility are not tested. Concurrent source writers or adversarial symlink replacement are outside the operating assumptions.

Observed server PyTorch 2.3.0+cu121 is below the existing package requirement `torch>=2.4`. The audit does not import/use torch; no installation or upgrade is proposed. The exact reported server environment has not been exercised.

Proposed server invocation — **NOT EXECUTED**:

```bash
cd /experiment-storage/chaidir/fawad-dsai/LIDC3D
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  /experiment-storage/chaidir/env/env-310-fawad/bin/python -B -m lidc3d.cli audit-source \
  --inventory artifacts/inventory.json \
  --annotation-db /experiment-storage/chaidir/env/env-310-fawad/lib/python3.10/site-packages/pylidc/pylidc.sqlite \
  --config configs/source_audit.yaml \
  --out artifacts/source-audit-20260917-01
```

That output directory must be new. A completed diagnostic run still does not approve the source, replace the XML reference standard, or release training. Stop here for separate research review and any future slice-selection policy.


### Review-branch publication

The user authorized publication of the existing candidate on `audit/source-annotation-linkage` only. Before staging, the six-file `git diff --binary` was byte-for-byte identical to the original archived patch, including its SHA256 above. All 16 Python source files also matched the source hashes recorded in provenance produced by the final successful test run. The preserved results remain 39 targeted tests, 35 existing tests, and 74 tests in the final combined run; they are not newly executed test results. No implementation/configuration/test files changed during publication, so no test rerun was needed. Only this handoff's location, publication context, and local-evidence link presentation changed.

The staged publication contains exactly the six implementation files listed above plus this tracked report. The staged implementation diff matched the archived tested patch byte-for-byte, and `git diff --cached --check` exited 0. All seven staged files were checked as text; no excluded artifacts or credential-pattern matches were found. Publication uses a normal push of this branch only, followed by verification of its remote SHA against local HEAD. The commit containing this report is the authoritative candidate revision; the GitHub history supplies its full SHA without a self-referential hash in the report. No dataset, SQLite database, credential, environment, archive, checkpoint, or server-generated patient artifact is included.

## Future LIDC3D coding handoffs

For future LIDC3D coding handoffs, publish the reviewed candidate on a review branch and keep review documents tracked under `docs/reviews/` in GitHub. Verify repository/branch/base and worktree first; preserve unrelated changes. Stage only the intended source/configuration/test/documentation files and tracked handoff. Confirm that the published files are covered by the reported tests; rerun relevant checks and update actual results if they changed. Review the staged diff, check whitespace and excluded artifact categories, commit, push without force, and verify the remote commit. Return the full commit SHA, GitHub commit link, exact committed-file list, and tracked handoff link. This workflow does not authorize merging into main, deployment, or university-server execution.
