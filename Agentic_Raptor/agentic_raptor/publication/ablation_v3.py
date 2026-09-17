"""A0-A9 component-ablation framework (declarative config, provenance,
result schema). This module defines WHAT each experiment is; it does not
run large campaigns itself -- see run_ablation_v3.py at the repo root for
the CLI driver, and Part 14 of the ablation spec for why paper-scale runs
are gated behind an explicit human approval step, not this file.

Ten primary experiments only (A0-A9). Do not add secondary mechanism
experiments here -- those live in `secondary_studies.py`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_raptor.electrical.pvt_eval import PvtConfig

ROOT = Path(__file__).resolve().parents[2]


def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str,
                  separators=(",", ":")).encode()).hexdigest()[:16]


# ============================================================================
# ExperimentBudget -- ONE object A0-A8 inherit from; an ablation may change
# only what its removed component logically requires (Part: FAIRNESS).
# ============================================================================
@dataclass(frozen=True)
class ExperimentBudget:
    proposal_count_k: int = 5                 # == TARGET_K in run_raptor_v2
    validated_candidate_budget: int = 5        # candidates the validator may pass through
    topologies_sent_to_sizing: int = 2         # == SELECT_K; unchanged by A5
    mcts_simulations: int | None = None        # None == engine default (topology_rl.stage3e1 MCTSSettings); 0 for A5
    sizing_repeats: int = 1
    max_optimization_spice_calls: int = 32     # == the `budget` passed to size_and_predict per branch
    max_sizing_steps: int = 32                 # one real SPICE call per sizing step in this engine
    final_verification_calls: int = 1          # 2 when calibrate/low_confidence triggers a backup verify
    pvt: PvtConfig = field(default_factory=PvtConfig)
    wallclock_limit_s: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pvt"] = asdict(self.pvt)
        return d

    def hash(self) -> str:
        return _stable_hash(self.to_dict())


# ============================================================================
# AblationConfig -- ONE clean switch object per experiment. Component
# isolation logic lives HERE and in run_raptor_v2.py's kwargs; nothing
# ablation-specific is scattered through unrelated modules.
# ============================================================================
@dataclass(frozen=True)
class AblationConfig:
    ablation_id: str            # "A0".."A9"
    name: str
    question: str
    use_llm: bool = True
    use_rag: bool = True
    use_sft: bool = True        # False -> base Qwen3-4B, no adapter
    use_exclusion_conditioning: bool = True
    use_mcts: bool = True
    use_puct: bool = True       # coupled with use_mcts (see __post_init__)
    use_sac: bool = True
    sizing_method: str = "sac"  # "tpe_lite" | "grid" | "random" when use_sac=False
    use_surrogate: bool = True
    use_sizing_ranker: bool = True
    # Stage 8 (2026-08-12, second deployment): default flipped back to True.
    # Stage 7.1/7.2A found the ORIGINAL 11-feature learned DPO ranker
    # LEARNED_DPO_NOT_JUSTIFIED, which briefly flipped this default False.
    # Stage 7.2B then found a richer 46-feature representation
    # (POST_SAC_FEATURES_V2) genuinely beats the deterministic selector
    # (77.80% vs 74.66% run-grouped DEV ranker-authority accuracy, 77
    # wins/45 losses/0 catastrophic errors -- see artifacts/publication_v3/
    # stage7_2b_dpo_repair/STAGE7_2B_REPORT.json), so DPO was re-justified
    # and FULL (A0) once again carries the learned Level-2 ranker by
    # default, matching run_raptor_v2.run_pipeline's restored default.
    use_dpo: bool = True
    learning_mode: str = "frozen"   # A0-A8 evaluation is always frozen
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)
    requires_puct_value_checkpoint: bool = True   # False only for A5 (search="none")
    # Stage 8 (second deployment): A8 (NO_LEARNED_DPO) is meaningful again
    # now that FULL once more carries a learned DPO to remove -- see
    # A8_NO_DPO below, un-retired.
    retired: bool = False
    retired_reason: str | None = None

    def __post_init__(self):
        # The current v2 search implementation has exactly two modes:
        # one-root MCTS+PUCT together, or "none" (direct prior top-K). There
        # is no code path that runs MCTS without PUCT selection or vice
        # versa, so splitting these two switches apart would silently do
        # nothing different from leaving them coupled -- refuse rather than
        # imply a distinction that doesn't exist.
        if self.use_mcts != self.use_puct:
            raise ValueError(
                f"{self.ablation_id}: use_mcts != use_puct is not "
                "separable in the current v2 search implementation "
                "(single search='none'|'one_root' switch)")
        if not self.use_sac and self.sizing_method == "sac":
            raise ValueError(
                f"{self.ablation_id}: use_sac=False requires a non-'sac' "
                "sizing_method")
        if self.use_sac and self.sizing_method != "sac":
            raise ValueError(
                f"{self.ablation_id}: sizing_method={self.sizing_method!r} "
                "with use_sac=True is contradictory")

    def config_hash(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k != "budget"}
        return _stable_hash(d)

    def to_run_pipeline_kwargs(self) -> dict:
        """Exact kwargs for run_raptor_v2.run_pipeline. This is the ONLY
        place ablation switches translate into pipeline parameters.

        SELECTOR CHANGE (2026-08-15, user decision after the GATE2 +
        determinism findings): use_mcts=True now maps to
        search="bandit_top2" -- the hash-pinned BANDIT_TOP2_V1 linear
        contextual bandit replaces AlphaZero as FULL's topology selector.
        Grounds: AlphaZero never added a pass in any campaign (old-gate,
        GATE2, F0-F3, per-seed retention 24/24 but zero gains, 97%
        destructive edits), while the bandit passed its offline
        spec-disjoint gate (12/12 vs prior 1/12) and matched the best arm
        on TRAIN electrically with a unique pass. use_mcts=False stays
        search="none" (direct prior top-K), so A5 now ablates the LEARNED
        TOPOLOGY SELECTOR (bandit vs deterministic prior). True AlphaZero
        remains available via run_pipeline(search="one_root"/"one_root_cc"
        /"bandit_top2_az") as an opt-in research mode."""
        return {
            "conditioning": ("exclusion" if self.use_exclusion_conditioning
                             else "temperature"),
            "search": "bandit_top2" if self.use_mcts else "none",
            "ranker_mode": "dpo" if self.use_dpo else "deterministic",
            "sizing_method": self.sizing_method,
            "use_surrogate": self.use_surrogate,
            "use_sizing_ranker": self.use_sizing_ranker,
            "use_llm": self.use_llm,
            "use_rag": self.use_rag,
            "learning_mode": self.learning_mode,
            "sizing_repeats": self.budget.sizing_repeats,
            "budget": self.budget.max_optimization_spice_calls,
        }


# ============================================================================
# Declarative YAML round-trip (Part: CONFIGURATION FILES)
# ============================================================================
CONFIG_DIR = ROOT / "experiments" / "configs"


def config_to_yaml_dict(c: AblationConfig) -> dict:
    d = asdict(c)
    d["budget"]["pvt"] = asdict(c.budget.pvt)
    return d


def config_from_yaml_dict(d: dict) -> AblationConfig:
    d = dict(d)
    bud = dict(d.get("budget") or {})
    pvt_d = dict(bud.pop("pvt", None) or {})
    bud["pvt"] = PvtConfig(**pvt_d) if pvt_d else PvtConfig()
    d["budget"] = ExperimentBudget(**bud)
    return AblationConfig(**d)


def dump_yaml_configs(out_dir: Path | None = None) -> list[Path]:
    """Write experiments/configs/A0_full.yaml .. A9_no_self_improvement.yaml.
    Idempotent -- re-running overwrites with the current in-code definition,
    which is the source of truth; the YAML files are a declarative MIRROR
    for inspection/CLI use, not hand-edited independently of this module."""
    import yaml as _yaml
    out_dir = out_dir or CONFIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"A0": "A0_full", "A1": "A1_no_llm", "A2": "A2_no_rag",
            "A3": "A3_no_sft", "A4": "A4_no_diversity",
            "A5": "A5_no_mcts_puct", "A6": "A6_no_sac",
            "A7": "A7_no_surrogate", "A8": "A8_no_dpo",
            "A9": "A9_no_self_improvement"}
    paths = []
    for aid, cfg in PRIMARY_EXPERIMENTS.items():
        p = out_dir / f"{names[aid]}.yaml"
        p.write_text(_yaml.safe_dump(config_to_yaml_dict(cfg), sort_keys=False),
                    encoding="utf-8")
        paths.append(p)
    return paths


# ============================================================================
# THE TEN PRIMARY EXPERIMENTS
# ============================================================================
def _budget(**overrides) -> ExperimentBudget:
    return ExperimentBudget(**overrides)


A0_FULL = AblationConfig(
    "A0", "FULL_AGENTIC_RAPTOR",
    "Reference system: does the complete pipeline out-perform every "
    "component-removed variant?")

A1_NO_LLM = AblationConfig(
    "A1", "NO_LLM",
    "Does generative LLM topology synthesis outperform retrieval of "
    "existing topologies?", use_llm=False)

A2_NO_RAG = AblationConfig(
    "A2", "NO_RAG",
    "Does retrieval-augmented circuit knowledge improve the LLM proposer?",
    use_rag=False)

A3_NO_SFT = AblationConfig(
    "A3", "NO_SFT",
    "Does domain-specific topology fine-tuning improve the base LLM?",
    use_sft=False)

A4_NO_DIVERSITY = AblationConfig(
    "A4", "NO_EXCLUSION_CONDITIONED_DIVERSITY",
    "Does exclusion-conditioned generation create USEFUL topology "
    "diversity?", use_exclusion_conditioning=False)

# 2026-08-11: root-level PUCT retired, replaced by TRUE_ALPHAZERO (see
# artifacts/publication_v3/ROOT_LEVEL_PUCT_RETIRED.json). This arm is now
# A5/NO_ALPHAZERO -- same validated LLM seed pool, bypasses AlphaZero
# topology refinement entirely, uses direct_prior_select_two()'s
# predeclared deterministic prior-ranking baseline (never the retired
# selector). The Python identifier stays A5_NO_MCTS_PUCT (renaming it
# would cascade through every import site for no functional benefit --
# use_mcts/use_puct=False -> to_run_pipeline_kwargs()["search"]=="none"
# already means exactly this); the DATA (name/question) reflects the new
# architecture.
# 2026-08-15 (GATE3 era): FULL's selector is now BANDIT_TOP2_V1 (see
# to_run_pipeline_kwargs), so this arm's question changed from "does
# AlphaZero add value" to "does the LEARNED topology selector add value
# over the deterministic prior ranking". The id stays A5 for table
# continuity across campaigns; the name/question reflect what it now
# measures. (AlphaZero's own verdict is closed: never added a pass in
# any campaign -- STAGE9_GATE2_ANALYSIS.json.)
A5_NO_MCTS_PUCT = AblationConfig(
    "A5", "NO_TOPOLOGY_SELECTOR",
    "Does the learned bandit topology selector (BANDIT_TOP2_V1) add value "
    "beyond the deterministic prior ranking (search='none')?",
    use_mcts=False, use_puct=False, requires_puct_value_checkpoint=False)

A6_NO_SAC = AblationConfig(
    "A6", "NO_SAC",
    "Does RL-based continuous sizing outperform a strong non-RL "
    "optimizer?", use_sac=False, sizing_method="tpe_lite")

A7_NO_SURROGATE = AblationConfig(
    "A7", "NO_MODEL_BASED_SURROGATE",
    "Does the learned model/surrogate improve SAC sample efficiency? "
    "(SAC remains ENABLED -- this is not A6.)",
    use_surrogate=False, use_sizing_ranker=False)

A8_NO_DPO = AblationConfig(
    "A8", "NO_LEARNED_DPO_RANKER",
    "Un-retired (Stage 8, 2026-08-12, second deployment): Stage 7.2B "
    "re-justified the learned DPO ranker (POST_SAC_FEATURES_V2,  "
    "DPO_REJUSTIFIED), so FULL (A0) once again carries a learned Level-2 "
    "ranker for this arm to meaningfully remove. Same full pipeline, same "
    "hard safety gate, same candidate inputs, same MB-SAC, same budgets -- "
    "only the learned DPO is disabled in favor of the predeclared "
    "deterministic selector. Does NOT remove the hard safety gate. "
    "Question: does learned post-SAC ranking (DPO V2) improve final "
    "candidate selection over the hard safety gate + deterministic "
    "selector alone?",
    use_dpo=False, retired=False)

A9_STATIC = AblationConfig(
    "A9", "NO_SELF_IMPROVEMENT_STATIC",
    "Does measured design experience improve future RAPTOR "
    "performance? (run via generation_state.py, not this A0-A8 paired "
    "runner)", learning_mode="static")

PRIMARY_EXPERIMENTS: dict[str, AblationConfig] = {
    c.ablation_id: c for c in
    (A0_FULL, A1_NO_LLM, A2_NO_RAG, A3_NO_SFT, A4_NO_DIVERSITY,
     A5_NO_MCTS_PUCT, A6_NO_SAC, A7_NO_SURROGATE, A8_NO_DPO, A9_STATIC)}

#: A0-A8 only -- the paired-evaluation set. A9 is a separate longitudinal
#: design (Part: IMPORTANT EXPERIMENTAL SEPARATION) and must never be mixed
#: into an A0-A8 paired sweep.
COMPONENT_ABLATIONS: dict[str, AblationConfig] = {
    k: v for k, v in PRIMARY_EXPERIMENTS.items() if k != "A9"}


# ============================================================================
# RESULT SCHEMA -- adapts run_pipeline's EXISTING trace (which already
# carries nominal/fom/pvt/spice_usage from the FoM/PVT work) into the
# structure this ablation framework's report/statistics layer expects.
# Deliberately NOT a new duplicate result dataclass mirrored in code AND
# json -- this is a pure reshape of the trace dict already on disk.
# ============================================================================
STOCK_FAMILIES = ("1s_none", "2s_none", "2s_miller", "2s_rc",
                  "3s_none", "3s_miller", "3s_rc")


def _selected_family(trace: dict) -> str | None:
    """Family of the design that reached authoritative verification."""
    sr = trace.get("stage8_ranker") or {}
    sel_hash = sr.get("selected_topology_hash")
    for r in (trace.get("stage5_alphazero") or {}).get("ranking") or []:
        if r.get("canonical_graph_hash") == sel_hash:
            return r.get("canonical_family")
    return None


def build_result_record(trace: dict, config: AblationConfig, *,
                        experiment_id: str, pipeline_seed: int,
                        spec_index: int | None = None,
                        spec_hash: str | None = None,
                        generation_id: int | None = None) -> dict:
    """(spec_index, ablation_id, pipeline_seed) -> the Part-10 RESULT SCHEMA.

    spec_index (not spec_id) is the correct grouping key -- spec_id is a
    display name (e.g. "t_easy_topology_0008") that multiple DISTINCT
    corpus records can share (confirmed: heldout idx=0 and idx=1 both
    display as "t_easy_topology_0008" with different UGBW targets -- and
    the same pattern holds corpus-wide: see
    agentic_raptor/publication/spec_registry.py, which found 28/29 heldout,
    74/85 train and 26/28 blindtest records share a non-unique context_id).
    spec_index is the corpus row's position (unambiguous but split-order
    dependent); spec_hash is a content hash of the exact prompt text
    (unambiguous AND order-independent) from spec_registry.spec_hash().
    Both are carried so a result row is identifiable even if the corpus
    file is ever regenerated in a different order.
    """
    spec = trace.get("stage1_spec", {}).get("spec", {})
    nominal = trace.get("nominal") or {}
    fom = trace.get("fom") or {}
    pvt = trace.get("pvt") or {}
    su = trace.get("spice_usage") or {}
    manifest = {"config_hash": config.config_hash(),
               "budget_hash": config.budget.hash(),
               "proposer_checkpoint": trace.get("proposer_checkpoint"),
               "ranker_checkpoint_hash":
               trace.get("stage8_ranker", {}).get("ranker_checkpoint_hash")}
    return {
        "experiment_id": experiment_id,
        "ablation_id": config.ablation_id,
        "ablation_name": config.name,
        "spec_id": trace.get("stage1_spec", {}).get("spec_id"),
        "spec_index": spec_index,
        "spec_hash": spec_hash,
        "topology_id": spec.get("topology_id"),
        "split": trace.get("stage1_spec", {}).get("split"),
        # LLM-VALUE PROBE (2026-08-17): which topology WON, so the analysis
        # can count passes whose winning structure lies OUTSIDE the stock
        # template library (composition beyond retrieval)
        "selected_family": _selected_family(trace),
        "selected_hash": (trace.get("stage8_ranker") or {}).get("selected_topology_hash"),
        "pipeline_seed": pipeline_seed,
        "generation_id": generation_id,
        "configuration_hash": config.config_hash(),
        "budget_hash": config.budget.hash(),
        "manifest_hash": _stable_hash(manifest),
        "nominal": {
            "gain_db": nominal.get("gain_db"), "pm_deg": nominal.get("pm_deg"),
            "ugbw_hz": nominal.get("ugbw_hz"), "idd_a": nominal.get("idd_a"),
            "ibias_a": nominal.get("ibias_a"), "power_w": nominal.get("power_w"),
            "c_load_f": nominal.get("c_load_f"),
            "complete_pass": nominal.get("complete_pass"),
            "failure_reasons": nominal.get("failure_reasons")},
        "fom": {"fom_value": fom.get("fom_value"),
               "fom_version": fom.get("fom_version"),
               "fom_units": fom.get("fom_units")},
        "pvt": {"total_corners": pvt.get("total_pvt_corners"),
               "passed_corners": pvt.get("passed_pvt_corners"),
               "failed_corners": pvt.get("failed_pvt_corners"),
               "pvt_pass_percent": pvt.get("pvt_pass_percent"),
               "robust_complete_pass": pvt.get("robust_complete_pass"),
               "worst_gain_db": pvt.get("worst_gain_db"),
               "worst_pm_deg": pvt.get("worst_pm_deg"),
               "worst_ugbw_hz": pvt.get("worst_ugbw_hz"),
               "max_idd_a": pvt.get("max_idd_a"),
               "max_power_w": pvt.get("max_power_w"),
               "worst_spec_margin": pvt.get("worst_spec_margin")},
        "spice": {"optimization_calls": su.get("optimization_spice_calls"),
                 "final_verification_calls":
                 su.get("final_nominal_verification_calls"),
                 "pvt_calls": su.get("pvt_spice_calls"),
                 "total_calls": su.get("total_spice_calls")},
        # None (not False) when the selected design's sizing trajectory
        # never hit exact_spec_pass within budget -- this is "never
        # happened," not a boolean.
        "calls_to_first_pass": (trace.get("stage6_sizing", {})
                                .get(trace.get("stage8_ranker", {})
                                    .get("selected_design", ""), {})
                                .get("calls_to_first_pass")),
        "runtime_s": trace.get("seconds"),
        "trace_result": trace.get("result"),
    }
