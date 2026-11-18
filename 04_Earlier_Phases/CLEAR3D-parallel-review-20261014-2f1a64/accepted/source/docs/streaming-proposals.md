# Streaming proposal preparation and release

The `prepare` CLI does not retain the whole cohort proposal pool. Scan
checkpoint validation still holds one scan's proposals, then discards them.
Only records and counts remain across scans. Aggregation rereads per-scan JSONL
files in cohort order and preserves canonical serialization and proposal IDs.
Complete resume and release verification compare the aggregate against scan
files row by row, rejecting missing, extra, reordered or inconsistent rows.
Hash and source checks remain required.

`freeze_bundle` verifies the disk pool before selection. Threshold search reads
validation scans only; final selection reads each scan separately. Sorting,
threshold inclusivity and caps are unchanged. The selected capped candidates
are retained in memory. CT arrays, cohort records and a single scan proposal
list still use memory; total peak RSS must be measured on the server.

For compatibility, `prepare_candidates` returns a materialized proposal list
by default. Large Python callers must pass `materialize_pool=False`, as the CLI
does. That returns a reiterable `PreparedProposalPool`; it has iteration and
length, but no random indexing. The existing preparation schema stays v1;
the implementation hash changes and old outputs cannot resume with new code.
Use a fresh output directory and preserve the accepted pilot baseline.

## Server validation

From the repository root with its virtual environment active:

```
PYTHONPATH=src python -u scripts/validate_streaming_pilot.py
```

This script pins the accepted 20-scan manifest and baseline marker hashes,
prepares a fresh `artifacts/luna16-prepare-pilot20-streaming-v2/prepared` tree,
uses the existing exact preparation comparator for all 833188 proposals, and
checks completed resume without recomputation or any prepared-file hash,
mode, size or modification-time change. It refuses an existing destination.
Timing and peak RSS for each phase are in `prepare.log`, `comparison.log`
and `resume.log`; it emits `streaming-validation.json` only after those phases.
It does not train, freeze, prepare the full cohort or access withheld scans.

Local validation: 245 tests passed in a temporary virtual environment with
existing scratch dependencies available to subprocesses. Server equivalence
and peak memory have not yet been measured for this implementation.
