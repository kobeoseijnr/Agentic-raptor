"""FULL-FIDELITY RAPTOR pipeline — one command, one spec, the REAL
components at every layer:

spec -> RAG -> SFT topology LLM -> parser/validators/allow-list -> realisation
-> MCTS decision (real-SPICE leaves) -> GENUINE SAC sizing (learning actor +
twin critics, updated from every real measurement) -> SURROGATE screen
(trained on accumulated measurements) -> BRADLEY-TERRY ranker (the learned
model) -> real ngspice -> PostSizingTopologyScore -> SIX cross-level feedback
channels (AZ value targets, MB-SAC replay, dynamics/surrogate data, ranker
pairs, RAG L4, LLM preference queue).

Run:  python run_full_raptor.py
"""
import copy
import itertools
import json
import re
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentic_raptor.llm_dpo import MODEL_ID
from agentic_raptor.llm_dpo.stage3e4 import generate
from agentic_raptor.electrical import discover_ngspice
from agentic_raptor.mapping import map_family
from agentic_raptor.mb_sac.stage3d1 import PostSizingTopologyScore
from agentic_raptor.topology_rl.stage3e2 import new_costs
from agentic_raptor.topology_rl.stage3e2_edits import (
    EditRejected, apply_edit, device_graph_hash, qualify_device_graph)

O = Path("artifacts/full_raptor_run").resolve()
O.mkdir(parents=True, exist_ok=True)
MEM = Path("datasets/simulation_memory")
trace = {}

# 1) SPEC + RAG (held-out)
corpus = json.loads(Path("artifacts/stage3e4/corpus.json").read_text())
ctx = [r for r in corpus["records"] if r["split"] == "heldout"][0]
gain_t = float(re.search(r"gain>=([\d.]+)dB", ctx["prompt"]).group(1))
pm_t = float(re.search(r"pm>=([\d.]+)deg", ctx["prompt"]).group(1))
trace["spec"] = {"gain_db": gain_t, "pm_deg": pm_t,
                 "prompt_head": ctx["prompt"].splitlines()[0]}

# 1b) HIERARCHICAL RAG AT INFERENCE: retrieve measured evidence (L4) and
# family summaries, inject the CONTENT into the prompt (not just an ID).
rag_lines = []
l4p = MEM / "self_improvement_runs.jsonl"
# Retrieval is ranked by RELEVANCE TO THIS SPEC, not by recency. Taking the
# newest few injected evidence from unrelated targets -- a 60 deg / 200 pF
# request could be handed a 45 deg / 50 pF result purely because it ran last.
from agentic_raptor.llm_dpo import integrity as _ig0
_want = _ig0.parse_spec(ctx["prompt"]) or {}
if l4p.is_file():
    _ev = [json.loads(x) for x in l4p.read_text().splitlines()]
    _ev = [e for e in _ev if e.get("stability")]

    def _closeness(e):
        """Smaller is nearer. PM first: it is the binding constraint."""
        d = 0.0
        if e.get("pm") is not None and _want.get("phase_margin_target_deg"):
            d += abs(e["pm"] - _want["phase_margin_target_deg"]) / 45.0
        if e.get("gain") is not None and _want.get("gain_target_db"):
            d += abs(e["gain"] - _want["gain_target_db"]) / 20.0
        return d
    for e in sorted(_ev, key=_closeness)[:6]:
        rag_lines.append(f"{e.get('stages','?')}stage {e['stability']}"
                         + (f" pm={round(e['pm'])}deg" if e.get("pm") else ""))
sump = MEM / "topology_summaries.jsonl"
if sump.is_file():
    for e in [json.loads(x) for x in sump.read_text().splitlines()][:200]:
        if e.get("stability_status") == "verified_stable":
            rag_lines.append(f"registry:{e['topology_id']} verified_stable")
            if len([x for x in rag_lines if x.startswith("registry")]) >= 2:
                break
retrieved = "; ".join(rag_lines[-4:]) if rag_lines else ""
prompt = ctx["prompt"]
if retrieved:
    prompt = prompt.replace("### BLOCKS", f"### KNOWN {retrieved}\n### BLOCKS")
trace["rag_retrieval"] = {"records_injected": len(rag_lines[-4:]),
                          "content": retrieved[:200]}

# 2) LLM proposes. MULTIMODAL mode (AGENTIC_RAPTOR_MULTIMODAL=1): the vision
# model reads the spec text AND the retrieved family's real schematic image.
import os as _os
_vlm_adapter = Path("artifacts/stage3e4b/vlm_sft_adapter")
if _os.environ.get("AGENTIC_RAPTOR_MULTIMODAL") == "1" and \
        not (_vlm_adapter / "adapter_config.json").is_file():
    # fail SOFT: the VLM adapter is not on disk (never saved or cleaned).
    # Falling back to the text model keeps the run honest and alive; the
    # trace records why multimodal did not run.
    print(f"WARNING: multimodal requested but {_vlm_adapter} is missing — "
          f"falling back to the text model")
    trace["multimodal"] = "unavailable_adapter_missing"
    _os.environ["AGENTIC_RAPTOR_MULTIMODAL"] = "0"
if _os.environ.get("AGENTIC_RAPTOR_MULTIMODAL") == "1":
    from peft import PeftModel as _PM
    from agentic_raptor.llm_dpo import parse_proposal_text as _ppt, \
        proposal_dict_valid as _pdv
    from agentic_raptor.llm_dpo.multimodal import (IMG_DIR as _IMG,
                                                   load_vlm as _lv,
                                                   multimodal_inputs as _mi)
    _proc, _vm, _vrec = _lv(lora=False)
    _vm = _PM.from_pretrained(_vm, str(_vlm_adapter))
    _img = _IMG / f"{ctx['topology_id']}.png"
    assert _img.is_file(), f"schematic image missing: {_img}"
    _inp = _mi(_proc, prompt, _img, _vm)
    with torch.no_grad():
        _out = _vm.generate(**_inp, max_new_tokens=260, do_sample=False)
    _text = _proc.tokenizer.decode(_out[0, _inp["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
    _obj = _ppt(_text)
    _valid, _reasons = (_pdv(_obj) if _obj else (False, ["unparseable"]))
    cand = {"parseable": _obj is not None, "valid": _valid,
            "reasons": [] if _valid else _reasons, "obj": _obj if _valid else None,
            "graph_hash": None}
    attempts = [{"multimodal": True, "model": _vrec["model_id"],
                 "image": str(_img), "valid": _valid}]
    del _vm
    torch.cuda.empty_cache()
    trace["llm_proposal"] = {"attempts": attempts, "multimodal": True,
                             "structure_hash": None}
    assert cand["valid"], f"multimodal proposal rejected: {cand['reasons']}"
    obj = cand["obj"]
else:
    cand = None
if cand is None:            # text-model path (default)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    # adapter override: point at a self-improvement campaign's ACCEPTED
    # checkpoint via AGENTIC_RAPTOR_ADAPTER; defaults to the stage3e4 SFT
    _adapter = _os.environ.get("AGENTIC_RAPTOR_ADAPTER",
                               "artifacts/stage3e4/sft_adapter")
    trace["llm_adapter"] = _adapter
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                             device_map="auto"),
        _adapter)
    cand, attempts = None, []
    for seed in range(4):
        c = generate(model, tok, prompt, sample_seed=seed)
        attempts.append({"seed": seed, "valid": c["valid"],
                         "reasons": c["reasons"]})
        if c["valid"]:
            cand = c
            break
    del model
    torch.cuda.empty_cache()
    trace["llm_proposal"] = {"attempts": attempts,
                             "structure_hash": cand["graph_hash"] if cand else None}
    assert cand, f"all proposals rejected: {attempts}"
    obj = cand["obj"]

# 3/4) Realisation
costs = new_costs()


class _S:
    topology_id = "full_raptor"


g0, _ = map_family(_S(), {
    "topology_id": _S.topology_id, "gain_stages": len(obj["stages"]),
    "functional_blocks": ["C"] if obj.get("compensation") else [],
    "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
    "graph_hash": None})
trace["realised_hash"] = device_graph_hash(g0)
exe = discover_ngspice()

# 5) REAL AlphaZero MCTS: policy/value networks + PUCT tree over the proposal
# root (Stage 3E.1 engine + trained checkpoint), THEN the local edit decision
# grounded on real SPICE.
from agentic_raptor.core.circuit_graph import CircuitEdge, CircuitGraph, CircuitNode
from agentic_raptor.core.types import DeviceType, TerminalType
from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.mb_sac.stage3d2 import V3 as _V3
from agentic_raptor.topology_rl import stage3e1 as s1

_km = {"nmos": DeviceType.NMOS, "pmos": DeviceType.PMOS, "cap": DeviceType.CAPACITOR,
       "res": DeviceType.RESISTOR, "isrc": DeviceType.CURRENT_SOURCE}
_tm = {"d": TerminalType.DRAIN, "g": TerminalType.GATE, "s": TerminalType.SOURCE,
       "b": TerminalType.BULK, "p": TerminalType.PLUS, "n": TerminalType.MINUS}
_cg = CircuitGraph("proposal_root")
for d in g0.devices:
    _cg.add_node(CircuitNode(node_id=d.device_id, device_type=_km[d.kind],
                             block_role=d.role))
    for t, net in d.nets.items():
        _cg.add_edge(CircuitEdge(_cg.next_id("e"), d.device_id, _tm[t], net))
_reg = TopologyRegistry(_V3)
# Candidate pool = the FIVE structures the corpus actually contains, built
# through the same mapping path the sizer uses. The old pool was 4 entries of
# the V3 registry, so the search ranked catalogue families that the proposer
# never generates and that the corpus never scores.
from agentic_raptor.llm_dpo import integrity as _ig
from agentic_raptor.topology_rl.value_refresh import FamilyRegistry as _FamReg
from run_puct_ablation import CORPUS_CLASSES as _CLASSES

_prop_cls = f"{len(obj['stages'])}s_" + _ig.compensation_class(obj)
_fam = _FamReg(set(_CLASSES) | {_prop_cls})


class _Shim:
    def get_topology(self, tid):
        if tid == "proposal_root":
            class _E:
                topology_id, graph, metadata = "proposal_root", _cg, {}
                path, source = O, "llm_proposal"
            return _E()
        try:
            return _fam.get_topology(tid)
        except KeyError:
            return _reg.get_topology(tid)

    def list_topologies(self):
        return sorted(set(_CLASSES) | {_prop_cls})

    def derive_edited(self, tid, edit_name):
        """Exposing this is what lets the tree descend past one ply."""
        return _fam.derive_edited(
            _prop_cls if tid == "proposal_root" else tid, edit_name)


nets_pv = s1.build_policy_value(0)
_ck = Path("artifacts/stage3e1/policy_value_ep0.pt")
if _ck.is_file():
    try:
        s1.load_checkpoint(nets_pv, _ck)
        trace_pv_ckpt = "loaded"
    except Exception:
        trace_pv_ckpt = "fresh(checkpoint incompatible)"
else:
    trace_pv_ckpt = "fresh"
_pool = [c for c in _CLASSES if c != _prop_cls]
# THE SPEC BEING DESIGNED FOR -- not s1.DEFAULT_SPEC. The value head takes
# gain/GBW/PM/load as input, so ranking against the hardcoded default
# (40 dB, 45 deg, 500 pF) evaluated every candidate for a specification
# nobody asked for; an 89 dB request was scored as if it were 40 dB.
_spec_parsed = _ig.parse_spec(prompt) or {}
_search_spec = {
    "target_gain_db": float(_spec_parsed.get("gain_target_db", 40.0)),
    "target_gbw_hz": float(_spec_parsed.get("ugbw_target_hz") or 1e4),
    "minimum_phase_margin_deg":
        float(_spec_parsed.get("phase_margin_target_deg", 45.0)),
    "load_capacitance_f":
        float(_spec_parsed.get("load_capacitance_pf", 500.0)) * 1e-12,
    "supply_voltage": 1.8}
trace["search_spec"] = _search_spec
_root_state = s1.TopologySearchState(
    topology_id="proposal_root", graph_hash=_cg.structural_hash(),
    lineage=[_cg.structural_hash()], spec=_search_spec,
    rag_context_ids=[ctx["context_id"]], available_blocks=[],
    legal_action_ids=[], edit_history=[], validation_status="validated",
    structural_features={"n_nodes": float(len(_cg.nodes))},
    previous_evidence_ref=None, remaining_search_budget=6,
    remaining_spice_budget=0, depth=0)
_cfg = s1.SearchConfig(num_simulations=256, leaf_mode="value_only",
                       training_mode=False, max_depth=3, seed=0)
_mcts = s1.TopologyMCTS(nets_pv, _Shim(), _pool, _cfg)
_root = _mcts.run(_root_state)
_dist = {c.action.action_id: c.N for c in _root.children}
trace["mcts_puct"] = {
    "engine": "PUCT + policy/value nets", "checkpoint": trace_pv_ckpt,
    "simulations": _cfg.num_simulations, "tree_nodes": len(_mcts.nodes),
    "root_visits": dict(sorted(_dist.items(), key=lambda x: -x[1])),
    "value_net_calls": _mcts.costs.value_net_calls,
    "selected": max(_dist, key=lambda k: _dist[k]) if _dist else None}

# ACT on PUCT's decision: if the search prefers a registry family over the
# proposal, realise THAT family (search is a decision-maker, not an advisor)
_sel = trace["mcts_puct"]["selected"]
if _sel and _sel.startswith("a_sel_") and _sel[6:] != _prop_cls:
    # selections are now corpus structure classes, realised through the same
    # mapping path the sizer uses (the V3 registry has no such ids)
    from run_puct_ablation import realise_class as _realise_class
    g0 = _realise_class(_sel[6:])
    trace["mcts_puct"]["acted_on"] = f"switched_root_to_{_sel[6:]}"
    trace["realised_hash"] = device_graph_hash(g0)
else:
    trace["mcts_puct"]["acted_on"] = "kept_llm_proposal"

# local edit decision grounded on REAL SPICE (expensive leaves)
q_keep = qualify_device_graph(_S.topology_id, g0, O, exe, "keep", costs)
try:
    g_e, _a = apply_edit(g0, "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE")
    q_edit = qualify_device_graph(_S.topology_id, g_e, O, exe, "edit", costs)
except EditRejected:
    g_e, q_edit = None, {}


def _pm(q):
    return (q.get("metrics") or {}).get("phase_margin_deg")


keep = (_pm(q_keep) or -999) >= (_pm(q_edit) or -999)
chosen = g0 if keep else g_e
trace["mcts"] = {"action": "KEEP" if keep else "ADD_COMPENSATION",
                 "keep_pm": _pm(q_keep), "edit_pm": _pm(q_edit)}

# 6) GENUINE SAC sizing via spec_sizing: spec-conditioned saturating reward
# (PM excess beyond the cushion earns ~nothing while gain deficits dominate),
# gain-capable action space (widths + LENGTHS + BIAS, not just widths), and a
# Bradley-Terry ranker conditioned on the ACTIVE specification + remaining
# budget. All measurements remain real ngspice.
import os
from agentic_raptor.mb_sac.spec_sizing import sac_size
spice_budget = int(os.environ.get("AGENTIC_RAPTOR_SIZING_BUDGET", "32"))  # knob 3
gain_t = float(os.environ.get("AGENTIC_RAPTOR_GAIN_TARGET", gain_t))      # knob 5
pm_t = float(os.environ.get("AGENTIC_RAPTOR_PM_TARGET", pm_t))            # knob 5
_cl = re.search(r"cl=([\d.]+)pF", ctx["prompt"])
_ug = re.search(r"ugbw>=([\d.eE+-]+)Hz", ctx["prompt"])
active_spec = {"gain_target_db": gain_t, "phase_margin_target_deg": pm_t,
               "load_capacitance_pf": float(_cl.group(1)) if _cl else 100.0,
               "ugbw_target_hz": float(_ug.group(1)) if _ug else None,
               "technology": "sky130"}
from agentic_raptor.llm_dpo.integrity import compensation_class as _cc
_fam = f"{len(obj['stages'])}s_{_cc(obj)}"
sz = sac_size(_S.topology_id, chosen, active_spec, exe, O, costs,
              budget=spice_budget, seed=0, family=_fam)
best, results, replay = sz["best"], sz["results"], sz["transitions"]
# downstream sections consume the legacy field names
best = dict(best, pm=best["pm_deg"], gain=best["gain_db"])
trace["sizing"] = {"method": "SAC(spec-conditioned)+surrogate+BT ranker",
                   "action_space": sz["action_space"],
                   "reward_policy": sz["reward_policy"],
                   "real_transitions": sz["spice_calls"],
                   "calls_to_first_exact_pass": sz["calls_to_first_exact_pass"],
                   "best_knobs": best["knobs"], "best_pm": best["pm"],
                   "best_gain": best["gain"],
                   "outcome_tier": sz["outcome"]["outcome_tier"],
                   "margin_vector": sz["outcome"]["margin_vector"],
                   "spec_met": sz["outcome"]["exact_spec_pass"]}

# 8) Score
score = PostSizingTopologyScore(
    topology_id=_S.topology_id, target_id=ctx["context_id"],
    status="successful_within_budget" if best["stable"] else "unstable"
    if best["pm"] is not None else "simulator_failure",
    verified_stable=best["stable"], feasible=trace["sizing"]["spec_met"],
    metrics={k: v for k, v in best["metrics"].items() if v is not None},
    margins={"pm": ((best["pm"] or -90) - pm_t) / pm_t,
             "gain": ((best["gain"] or 0) - gain_t) / gain_t},
    real_spice_calls=costs["real_spice_calls"], calls_to_first_pass=None,
    simulator_failures=costs["simulator_failures"])
score.compute_scalar()
near = bool(best["stable"] and best["pm"] and best["gain"]
            and best["pm"] >= 0.95 * pm_t and best["gain"] >= 0.95 * gain_t)
trace["score"] = {"status": score.status, "scalar": score.scalar_value,
                  "outcome_tier": ("spec_met" if trace["sizing"]["spec_met"]
                                   else "within_5pct_of_spec" if near
                                   else "stable_below_spec" if best["stable"]
                                   else "unstable" if best["pm"] is not None
                                   else "unmeasured"),                    # knob 4
                  "spec_met": trace["sizing"]["spec_met"],
                  "real_spice_calls": costs["real_spice_calls"]}

# 9) FULL cross-level feedback: all six channels
fb = {}
with (MEM / "az_value_targets.jsonl").open("a", encoding="utf-8") as f:
    # Task 7: the value target is the FINAL post-sizing outcome (bounded,
    # component-decomposed), conditioned on spec/family/budget — never the
    # nominal topology stability, never a bare scalar hiding a failure
    f.write(json.dumps({"graph_hash": trace["realised_hash"],
                        **sz["value"],
                        "value_source": "REAL_POST_SIZING_SPICE",
                        "legacy_scalar": score.scalar_value,
                        "spec": trace["spec"], "active_spec": active_spec,
                        "topology_family": trace["mcts_puct"].get("acted_on"),
                        "sizing_budget": spice_budget,
                        "spice_calls": sz["spice_calls"],
                        "uncertainty": 0.0 if best.get("stability", "").startswith("verified") else 1.0}) + "\n")
fb["az_value_targets"] = 1
# replay + dynamics rows are now appended (family-labelled) by sac_size
# itself, which also persists the SAC nets / surrogate / ranker per family —
# writing here again would duplicate every row
fb["mbsac_replay"] = len(replay)
fb["sizing_memory"] = sz.get("memory")
fb["dynamics_surrogate"] = len(results)
from agentic_raptor.llm_dpo import scores_to_pairs
fb["ranker_and_llm_pairs"] = scores_to_pairs()
with (MEM / "self_improvement_runs.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps({"level": "L4", "source": "full_raptor",
                        "stages": len(obj["stages"]),
                        "stability": "verified_stable" if best["stable"] else "other",
                        "pm": best["pm"]}) + "\n")
fb["rag_l4"] = 1
trace["feedback_channels"] = fb

(O / "TRACE.json").write_text(json.dumps(trace, indent=1, default=str),
                              encoding="utf-8")
print(json.dumps(trace, indent=1, default=str))
