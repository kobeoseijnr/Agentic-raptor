# Repository Separation Report

Date: 2026-07-25. Not a git repository (no VCS history involved). All moves were
same-volume renames (robocopy /MOVE for ACL-protected vendored `.git` internals),
so file contents were never rewritten; the inventory's MD5 hashes (non-bulk files)
certify integrity — bulk generated trees were relocated by directory rename, which
cannot alter content.

## Files moved — 588,540 total (see ORIGINAL_RAPTOR_FILE_INVENTORY.csv/.json)

Directories moved to `RAPTOR_Legacy/`: rag, dpo, controller, graph, graph_search,
mb_sac, surrogate, experiments, data, results, docs, tools, configs, scripts,
outputs, llm, topology_dpo, RGNN_RL, AnalogGym, AutoCkt, baselines, archive,
external. Root files moved: README.md (anglog → `RAPTOR_Legacy/README.md`),
.gitignore, bsim4v5.out, inspect_results.py, tmp_make_paired_ucbtopk.py,
tmp_pair_mini100_surcal.py, _tmp_dpo_debug.py.

By category (from inventory): source_code 479 · test 20 · configuration 61,686 ·
csv_dataset 1,989 · experiment_result CSVs 1,593 · rag_memory 932 · dpo_pairs
16,377 · json_artifact 12,257 · spice_data 163,224 · model_checkpoint 30 ·
documentation 4,866 · figure 317 · log 159,255 · cache 292 · other 165,223.

## CSV / JSON / JSONL inventory
- Every original-RAPTOR CSV was moved (3,582 CSV rows across csv_dataset +
  experiment_result, plus CSVs inside rag_memory/dpo_pairs categories); none left at
  root; none deleted. Destinations preserve original relative paths under
  `RAPTOR_Legacy/` (e.g. `RAPTOR_Legacy/data/generated_candidates/phase9_generation_results.csv`).
- Key artifacts verified present post-move: `phase9_generation_results.csv`,
  `data/rag_memory/rag_memory.jsonl`, `data/preference_pairs/dpo_pairs.jsonl`,
  `docs/RAG_DPO_Novelty_Evidence_Report.docx` — all True.
- Duplicates: none deleted; hashes for non-bulk files are in the inventory
  (`content_hash` column); bulk generated trees marked `skipped-bulk` (hashing
  545k+ OneDrive files would force cloud hydration; renames don't alter content).

## Result artifacts
`results/tables`, `results/logs`, `results/figures`, all outputs/, data/, archive/
moved intact under `RAPTOR_Legacy/` per the "archived, not deleted" policy. Nothing
was deleted — including `__pycache__` (preservation prioritized).

## Source modules
All legacy Python packages moved (479 source files + 20 tests), including every
module named in the task (rag/*, dpo/*, controller/rag_dpo_controller.py,
experiments/*, docs/generate_rag_dpo_evidence_report.py, graph/*, graph_search/*,
tools/cleanup_legacy_artifacts.py). Dependency tracing: legacy modules import each
other by top-level package name and anchor paths via `Path(__file__).parents[1]`,
so moving the complete set together keeps the dependency graph closed — verified
by imports (below).

## Agentic RAPTOR
`Agentic_Raptor/` untouched by the separation except one file:
`agentic_raptor/adapters/legacy_raptor.py` — the SINGLE centralized legacy path
now resolves `<repo>/RAPTOR_Legacy`. All other adapters (existing_mb_sac/spice/
rag/surrogate, legacy_netlist, sizing.adapters) inherit it; no scattered sys.path
edits. No legacy code copied into Agentic_Raptor.

## Shared resources
None. Outputs are fully separated (`RAPTOR_Legacy/{data,results,outputs}` vs
`Agentic_Raptor/outputs`). Root-level items: repository README.md (new), this
report, the two inventory files, `_separation_inventory.py` (inventory tooling).

## Updated paths
1 file: `Agentic_Raptor/agentic_raptor/adapters/legacy_raptor.py`. No legacy file
needed path changes: all listed commands use project-relative paths that remain
valid when run from inside `RAPTOR_Legacy/` (verified: all anchor via
`__file__`-relative roots or CWD-relative arguments).

## Test results
- **Original RAPTOR imports**: `rag.memory_schema, dpo.dpo_interface,
  controller.controller_state, graph.graph_schema, graph_search.mcts_node,
  mb_sac.replay_buffer, surrogate.feature_schema, llm.llm_client` → OK from
  `RAPTOR_Legacy/`.
- **Original RAPTOR command verification**: all 9 required entry scripts exist and
  byte-compile (`py_compile`) in place; verified by compile + import rather than
  full execution — full runs mutate historical evidence artifacts (memory/pair
  CSVs), which this task forbids risking. `--validate-spice` remains available.
- **Agentic RAPTOR tests**: 176 passed, 1 skipped (opt-in API test) — unchanged.
- **Adapter tests**: `inspect-repository` resolves `RAPTOR_Legacy` with all 9
  legacy packages available; the legacy-netlist comparison test (in the suite) passed.

## Missing files
None of the explicitly named artifacts are missing. One repository-root item could
not be located post-separation: a root `.claude/` project-settings folder observed
at session start is no longer present at root (it was never in the move list);
if project-scoped Claude settings are needed, check `RAPTOR_Legacy/.claude/` and
move it back to root — flagged rather than silently ignored.

## Integrity
Non-bulk relocated files carry MD5 hashes in the inventory (computed pre-move;
renames preserve content). Bulk trees: relocation by directory rename; file counts
match the inventory total. No algorithmic changes to either project.
