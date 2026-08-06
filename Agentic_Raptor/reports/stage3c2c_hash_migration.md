# Stage 3C.2c - V1 Hash Migration & OPAMP-Generator Re-deduplication

## Defect & remediation
Role-blind WL hashing (core, since Stage 3A) ignored block_role -> could merge
graphs differing only in signed-gm / ff-fb / functional roles. Fixed hasher
(block_role in WL labels) is the single source of truth
(CircuitGraph.structural_hash); no duplicate hashers exist; no legacy fallback
is exposed (a v1 hash survives only as stored `legacy_graph_hash` metadata).

## Migration (executed; artifacts/stage3c2c/)
- Pre-migration snapshot: artifacts/corpus_snapshots/pre_stage3c2c_v1_hash_migration/
  (220 files, SHA-256 manifest, hasher_before=wl_role_blind_v1).
- 57/57 v1 families migrated: graph_hash -> role-aware v2; legacy_graph_hash
  preserved; hash_version=2; hash_algorithm=wl_role_aware_v2. Changed: 56,
  unchanged: 1. Atomic metadata writes; snapshot untouched.
- AnalogGym: all 17 topology IDs preserved; netlists hashed & unchanged ->
  electrical_evidence_preserved for 17/17; no simulation rerun; Level-4 records
  keep binding via topology_id + netlist artifacts (graph hash = identity
  metadata, per the graph/netlist/simulation identity distinction).

## OPAMP-Generator re-extraction & role-aware dedup (from all 1,500 rows)
valid 1,500 / invalid 0 -> unique role-aware graphs **15** (audit-confirmed,
not targeted); duplicates 1,485; variant-pair analysis found 0 shape-equal
pairs differing only in role/sign/ff-fb (distinct role-profiles also differ in
shape in this dataset). Corrected families topology_og2_0001..0015 in
artifacts/topology_registry_v1_hash_v2/ with relationship=legacy_family_split
from legacy family topology_0057 (v1 clustering had collapsed the opamp-gen
variants into effectively one active family); the legacy ID is retained as
lineage, never reused for descendants.

## Lineage & consistency
v1_to_hash_v2_lineage.jsonl: legacy_family_split (15 records, conf 0.9,
source-row bridged) + hash_only_migration (remaining families, conf 1.0).
Consistency checker (fails loudly): hash_version==2 everywhere, no stale
active hash, legacy hash present, no duplicate corrected IDs, no source row in
two active families -> 0 problems. CktGNN v1 stays invalidated/excluded; v2
registry already hash_version-2 semantics by construction.

## SPICE cache policy
Existing cache key = topology hash + sizing + analyses + corner + simulator
fingerprint. Old entries become misses under v2 hashes (classified
legacy_graph_hash_cache_entry conceptually); misses can only cost re-runs,
never wrong results. Recommended compound key (netlist/testbench/model/config/
simulator) documented for Stage 3D adoption.

## Splits
Family-level split labels ride on topology_ids, which are preserved
(AnalogGym + non-opamp families). The 15 og2 descendants inherit ONE split
group (legacy topology_0057's split) to prevent leakage; recorded in lineage.

## Tests
6 new migration tests (legacy preserved, evidence preserved, snapshot,
lineage + one-row-one-family, 4-way sign hash distinction, no v1 active).
Full suite: **255 passed, 1 skipped** - nothing weakened.

## Limitations
og2 families are structural only (unvalidated); registry_v1_hash_v2 holds the
opamp descendants while datasets/topology_library retains migrated originals -
the combined operational registry assembly is Stage 3C.2b; RAG record files
still carry v1 hashes in datasets/topology_rag (rebuild scheduled with 3C.2b).
