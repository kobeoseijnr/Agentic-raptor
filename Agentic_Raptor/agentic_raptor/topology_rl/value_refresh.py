"""Retrain the AlphaZero VALUE head on measured POST-SIZING outcomes.

The stage3e1 policy/value checkpoint predates the sizing repair: its value
head was trained on nominal-stability-era targets, so PUCT ranks topologies
with stale knowledge. This module rebuilds value training examples from the
accumulated `az_value_targets.jsonl` entries whose targets came from real
post-sizing ngspice outcomes, fine-tunes the value head (policy target is the
trivial keep-distribution — value-only training), and reports a decision-
quality evaluation on a held-out 20% split:

  * Spearman rank correlation between predicted scalar and measured value
  * pairwise ranking accuracy (does the net order candidate A above B when
    A's measured post-sizing value is higher?)

Both metrics are reported for the STALE checkpoint and the RETRAINED one, so
the improvement claim is measured, not asserted. The old checkpoint is backed
up before overwriting.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
AZ_FILE = _ROOT / "datasets/simulation_memory/az_value_targets.jsonl"
#: measured wrong-tier outcomes (build_counterfactual_values.py). Kept in a
#: separate file and OFF by default: the accumulated campaign targets are
#: 354/355 spec-compatible, so including these changes what the value head
#: learns -- opt in explicitly rather than silently shifting past ablations.
CF_FILE = _ROOT / "datasets/simulation_memory/az_counterfactual_targets.jsonl"
CKPT = _ROOT / "artifacts/stage3e1/policy_value_ep0.pt"
OUT = _ROOT / "artifacts/value_refresh"

_FAMILY_RE = re.compile(r"^(\d)s_(\w+)$")


def family_circuit_graph(stages: int, comp: str):
    """Canonical CircuitGraph for a structure family (for the value net's
    graph embedding), built through the same mapping path the pipeline uses."""
    from agentic_raptor.core.circuit_graph import (CircuitEdge, CircuitGraph,
                                                   CircuitNode)
    from agentic_raptor.core.types import DeviceType, TerminalType
    from agentic_raptor.mapping import map_family
    from agentic_raptor.topology_rl.stage3e2_edits import (EditRejected,
                                                           apply_edit)

    comp = {"miller_cap": "miller", "rc_nulling": "rc"}.get(comp, comp)

    class _S:
        topology_id = f"vh_{stages}s_{comp}"
    g, _ = map_family(_S(), {
        "topology_id": _S.topology_id, "gain_stages": stages,
        "functional_blocks": ["C"] if comp != "none" else [],
        "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
        "graph_hash": None})
    if comp == "rc":
        try:
            g, _a = apply_edit(g, "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE")
        except EditRejected:
            pass
    return device_graph_to_circuit_graph(g, _S.topology_id)


def device_graph_to_circuit_graph(g, topology_id: str):
    """DeviceCircuitGraph -> CircuitGraph (the search/value-net view).

    Shared so a topology proposed by the LLM is converted by exactly the same
    rules as a corpus family. Converting them differently would make their
    structural hashes incomparable, and the search ranks by that hash.
    """
    from agentic_raptor.core.circuit_graph import (CircuitEdge, CircuitGraph,
                                                   CircuitNode)
    from agentic_raptor.core.types import DeviceType, TerminalType
    km = {"nmos": DeviceType.NMOS, "pmos": DeviceType.PMOS,
          "cap": DeviceType.CAPACITOR, "res": DeviceType.RESISTOR,
          "isrc": DeviceType.CURRENT_SOURCE}
    tm = {"d": TerminalType.DRAIN, "g": TerminalType.GATE,
          "s": TerminalType.SOURCE, "b": TerminalType.BULK,
          "p": TerminalType.PLUS, "n": TerminalType.MINUS}
    cg = CircuitGraph(topology_id)
    for d in g.devices:
        cg.add_node(CircuitNode(node_id=d.device_id, device_type=km[d.kind],
                                block_role=d.role))
        for t, net in d.nets.items():
            cg.add_edge(CircuitEdge(cg.next_id("e"), d.device_id, tm[t], net))
    return cg


class FamilyRegistry:
    """Registry shim: family ids -> canonical CircuitGraph entries."""

    def __init__(self, families):
        self._cache = {}
        for fam in families:
            m = _FAMILY_RE.match(fam)
            if not m:
                continue

            class _E:
                pass
            e = _E()
            e.topology_id = fam
            e.graph = family_circuit_graph(int(m.group(1)), m.group(2))
            e.metadata = {}
            self._cache[fam] = e

    def get_topology(self, tid):
        return self._cache[tid]

    def list_topologies(self):
        return sorted(self._cache)

    def _register(self, tid, graph):
        class _E:
            pass
        e = _E()
        e.topology_id, e.graph, e.metadata = tid, graph, {}
        self._cache[tid] = e
        return tid

    def derive_edited(self, tid: str, edit_name: str) -> str:
        """Materialise `tid` + one compensation edit as its own entry.

        Gives the search a genuinely different child graph to descend into,
        which is what makes depth > 1 mean anything. Raises (EditRejected)
        when the edit does not apply to this structure -- the validator
        turns that into an honest rejection reason.
        """
        from agentic_raptor.core.circuit_graph import (CircuitEdge,
                                                       CircuitGraph,
                                                       CircuitNode)
        from agentic_raptor.core.types import DeviceType, TerminalType
        from agentic_raptor.topology_rl.stage3e2_edits import apply_edit
        child = f"{tid}#{'add' if edit_name.startswith('ADD') else 'rep'}"
        if child in self._cache:
            return child
        m = _FAMILY_RE.match(tid.split("#", 1)[0])
        if not m:
            raise KeyError(f"not a family id: {tid}")
        from agentic_raptor.mapping import map_family

        class _S:
            topology_id = f"vh_{tid}"
        stages, comp = int(m.group(1)), m.group(2)
        comp = {"miller_cap": "miller", "rc_nulling": "rc"}.get(comp, comp)
        dg, _ = map_family(_S(), {
            "topology_id": _S.topology_id, "gain_stages": stages,
            "functional_blocks": ["C"] if comp != "none" else [],
            "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
            "graph_hash": None})
        dg, _a = apply_edit(dg, edit_name)          # raises if inapplicable
        km = {"nmos": DeviceType.NMOS, "pmos": DeviceType.PMOS,
              "cap": DeviceType.CAPACITOR, "res": DeviceType.RESISTOR,
              "isrc": DeviceType.CURRENT_SOURCE}
        tm = {"d": TerminalType.DRAIN, "g": TerminalType.GATE,
              "s": TerminalType.SOURCE, "b": TerminalType.BULK,
              "p": TerminalType.PLUS, "n": TerminalType.MINUS}
        cg = CircuitGraph(child)
        for d in dg.devices:
            cg.add_node(CircuitNode(node_id=d.device_id,
                                    device_type=km[d.kind],
                                    block_role=d.role))
            for t, net in d.nets.items():
                cg.add_edge(CircuitEdge(cg.next_id("e"), d.device_id,
                                        tm[t], net))
        return self._register(child, cg)


def load_targets(include_counterfactual: bool = False):
    """Post-sizing value targets with a usable family label.

    include_counterfactual adds the measured wrong-tier outcomes, without
    which the head sees only spec-compatible pairings and cannot learn that
    a mismatched structure class is a bad choice.
    """
    entries = []
    files = [AZ_FILE] + ([CF_FILE] if include_counterfactual else [])
    for path in files:
        if not path.is_file():
            continue
        for x in path.read_text(encoding="utf-8").splitlines():
            if not x.strip():
                continue
            e = json.loads(x)
            if e.get("value_source") != "REAL_POST_SIZING_SPICE":
                continue
            if not _FAMILY_RE.match(e.get("topology_family") or ""):
                continue
            if not isinstance(e.get("spec"), dict) or "value_target" not in e:
                continue
            entries.append(e)
    return entries


def _example(e, budget_frac: float = 1.0):
    """Value-only stage3e1 training example from one target entry."""
    from dataclasses import asdict
    from agentic_raptor.topology_rl.stage3e1 import TopologySearchState
    spec = e["spec"]
    st = TopologySearchState(
        topology_id=e["topology_family"], graph_hash=e.get("graph_hash", "x"),
        lineage=[e.get("graph_hash", "x")],
        spec={"target_gain_db": spec.get("gain_target_db", 40.0),
              "target_gbw_hz": spec.get("ugbw_target_hz") or 1e4,
              "minimum_phase_margin_deg":
                  spec.get("phase_margin_target_deg", 45.0),
              "load_capacitance_f":
                  spec.get("load_capacitance_pf", 100.0) * 1e-12,
              "supply_voltage": 1.8},
        rag_context_ids=[e.get("context_id", "")], available_blocks=[],
        legal_action_ids=["a_keep"], edit_history=[],
        validation_status="validated",
        structural_features={}, previous_evidence_ref=None,
        remaining_search_budget=int(8 * budget_frac),
        remaining_spice_budget=int(e.get("sizing_budget", 16)
                                   - e.get("spice_calls", 0)),
        depth=0)
    return {"state": asdict(st), "legal_action_ids": ["a_keep"],
            "visit_distribution": {"a_keep": 1.0},
            "value_target": float(e["value_target"]),
            "spec": st.spec, "split": "train"}


def evaluate(nets, examples, reg) -> dict:
    """Spearman + pairwise ranking accuracy of the value scalar against the
    measured post-sizing value targets."""
    from agentic_raptor.topology_rl.stage3e1 import TopologySearchState
    import torch
    preds, targs = [], []
    with torch.no_grad():
        for ex in examples:
            st = TopologySearchState(**ex["state"])
            preds.append(float(nets["value_forward"](st, reg)["scalar"]))
            targs.append(ex["value_target"])
    n = len(preds)

    def ranks(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rp, rt = ranks(preds), ranks(targs)
    mp, mt = sum(rp) / n, sum(rt) / n
    num = sum((a - mp) * (b - mt) for a, b in zip(rp, rt))
    den = (sum((a - mp) ** 2 for a in rp)
           * sum((b - mt) ** 2 for b in rt)) ** 0.5
    spearman = num / den if den else 0.0
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(targs[i] - targs[j]) < 0.05:
                continue                    # measured tie: not a ranking case
            total += 1
            correct += int((preds[i] > preds[j]) == (targs[i] > targs[j]))
    return {"n": n, "spearman": round(spearman, 4),
            "pairwise_accuracy": round(correct / total, 4) if total else None,
            "ranked_pairs": total}


def _ident_hash(e) -> float:
    ident = "|".join(str(e.get(k, "")) for k in
                     ("graph_hash", "context_id", "topology_family",
                      "campaign_id", "generation"))
    return (int(hashlib.sha256(ident.encode("utf-8")).hexdigest()[:12], 16)
            % 10_000) / 10_000


def example_bucket(e, holdout_frac: float = 0.2,
                   val_frac: float = 0.15) -> str:
    """Three-way stable split: holdout / val / train.

    `val` exists so training can early-stop on data it is not fitting
    WITHOUT touching the reported hold-out. Measured need: from the current
    checkpoint, hold-out spearman peaks around epoch 10 (0.8050) and decays
    to 0.7824 by epoch 30 -- a fixed 30-epoch schedule trains straight past
    the best model and reports the decayed one as "no improvement".
    """
    h = _ident_hash(e)
    if h < holdout_frac:
        return "holdout"
    if h < holdout_frac + val_frac:
        return "val"
    return "train"


def example_split(e, holdout_frac: float = 0.2) -> str:
    """Stable per-example split, decided by CONTENT not list position.

    refresh() previously shuffled indices of a growing list with a fixed
    seed, so an example could sit in `train` one campaign and `holdout` the
    next. Measured on the current 355 targets: 69% of a later hold-out had
    been trained on earlier. That inflates the stale checkpoint's score and
    makes an honest retrain read as a regression (the observed
    0.903 -> 0.891, improved=False). Hashing identity keeps an example on
    the same side forever, however much data arrives later.
    """
    return "holdout" if _ident_hash(e) < holdout_frac else "train"


def _discrimination(nets, entries, reg) -> dict:
    """The decision the search actually needs: does the head score a
    spec-COMPATIBLE class above an incompatible one for the same spec?"""
    import torch
    from agentic_raptor.topology_rl.stage3e1 import TopologySearchState
    comp, mismatch = [], []
    with torch.no_grad():
        for e in entries:
            st = TopologySearchState(**_example(e)["state"])
            try:
                v = float(nets["value_forward"](st, reg)["scalar"])
            except KeyError:                  # family absent from registry
                continue
            tier = e["spec"].get("gain_target_db", 0)
            want = 1 if tier < 30 else 2 if tier < 70 else 3
            (comp if int(e["topology_family"][0]) == want
             else mismatch).append(v)
    if not comp or not mismatch:
        return {"note": "one side unpopulated", "compatible_n": len(comp),
                "mismatch_n": len(mismatch)}
    mc, mm = sum(comp) / len(comp), sum(mismatch) / len(mismatch)
    return {"compatible_n": len(comp), "mismatch_n": len(mismatch),
            "mean_value_compatible": round(mc, 4),
            "mean_value_mismatch": round(mm, 4),
            "margin": round(mc - mm, 4),
            "separates_correctly": mc > mm}


def _policy_examples():
    """AlphaZero POLICY-head training examples from recorded P8 search visit
    distributions: cross-entropy target = normalized visit counts (canonical
    AlphaZero pi_MCTS target), value target = the arm's measured outcome.
    This makes BOTH heads learn -- value from outcomes, policy from search."""
    from dataclasses import asdict
    from agentic_raptor.topology_rl.stage3e1 import TopologySearchState
    # both search batteries record pi_MCTS the same way: the P battery (L8
    # proposer) and the rescue battery (L2 proposer). Harvesting both means
    # one run repopulates the policy head instead of leaving it starved.
    sources = [(_ROOT / "artifacts/publication/puct_ablation"
                / "puct_ablation_rows.jsonl",
                _ROOT / "artifacts/publication/puct_ablation/proposals.json",
                "P8"),
               (_ROOT / "artifacts/publication/puct_rescue"
                / "puct_rescue_rows.jsonl",
                _ROOT / "artifacts/publication/puct_rescue/proposals.json",
                "R8")]
    out = []
    pairs = []
    for rows_f, props_f, arm in sources:
        if rows_f.is_file() and props_f.is_file():
            props = json.loads(props_f.read_text())
            for x in rows_f.read_text(encoding="utf-8").splitlines():
                if x.strip():
                    pairs.append((json.loads(x), props, arm))
    for r, props, want_arm in pairs:
        visits = r.get("visits") or {}
        prop = props.get(r.get("context_id") or "")
        if r.get("arm") != want_arm or not visits or not prop:
            continue
        # proposal's structure class = the state's topology identity
        from agentic_raptor.llm_dpo.integrity import (candidate_identity,
                                                      compensation_class)
        obj = prop["obj"]
        prop_cls = f"{len(obj['stages'])}s_{compensation_class(obj)}"
        spec = prop["spec"]
        total = sum(visits.values()) or 1
        dist = r.get("distance")
        value = max(0.0, 1.0 - dist) if dist is not None else None
        if value is None:
            continue
        st = TopologySearchState(
            topology_id=prop_cls,
            graph_hash=candidate_identity(obj)["canonical_graph_hash"],
            lineage=[prop_cls],
            spec={"target_gain_db": spec["gain_target_db"],
                  "target_gbw_hz": spec.get("ugbw_target_hz") or 1e4,
                  "minimum_phase_margin_deg":
                      spec["phase_margin_target_deg"],
                  "load_capacitance_f":
                      spec["load_capacitance_pf"] * 1e-12,
                  "supply_voltage": 1.8},
            rag_context_ids=[r["context_id"]], available_blocks=[],
            legal_action_ids=sorted(visits), edit_history=[],
            validation_status="validated", structural_features={},
            previous_evidence_ref=None, remaining_search_budget=6,
            remaining_spice_budget=0, depth=0)
        out.append({"state": asdict(st),
                    "legal_action_ids": sorted(visits),
                    "visit_distribution": {k: v / total
                                           for k, v in visits.items()},
                    "value_target": value, "spec": st.spec,
                    "split": "train",
                    "policy_families": sorted(
                        {k[6:] for k in visits if k.startswith("a_sel_")}
                        | {prop_cls})})
    return out


def refresh(epochs: int = 30, holdout_frac: float = 0.2,
            seed: int = 0, include_counterfactual: bool = False,
            patience: int = 8) -> dict:
    """Retrain the value head on post-sizing targets; report stale-vs-
    retrained decision quality on the held-out split; back up and overwrite
    the checkpoint the full pipeline loads."""
    from random import Random
    from agentic_raptor.topology_rl import stage3e1 as s1
    entries = load_targets(include_counterfactual)
    if len(entries) < 10:
        return {"skipped": f"only {len(entries)} usable targets"}
    rng = Random(seed)
    # content-hashed split: an example never changes sides as data grows
    train_ex, val_ex, hold_ex = [], [], []
    for e in entries:
        {"holdout": hold_ex, "val": val_ex,
         "train": train_ex}[example_bucket(e, holdout_frac)].append(
             _example(e))
    if len(hold_ex) < 4:                       # tiny corpus: fall back
        hold_ex, train_ex = train_ex[:4], train_ex[4:]
    if len(val_ex) < 4:                        # no room for early stopping
        val_ex = list(hold_ex[:0])             # stays empty -> fixed epochs
    # AlphaZero policy learning: recorded search visit distributions join
    # training as cross-entropy targets for the POLICY head (value examples
    # carry a trivial keep-distribution, so only these move the policy)
    pol_ex = _policy_examples()
    train_ex += pol_ex
    fams = {e["topology_family"] for e in entries}
    for ex in pol_ex:
        fams |= set(ex.get("policy_families", []))
    reg = FamilyRegistry(fams)

    nets = s1.build_policy_value(seed)
    stale_meta = {}
    if CKPT.is_file():
        try:
            stale_meta = s1.load_checkpoint(nets, CKPT)
        except Exception as exc:
            stale_meta = {"load_failed": str(exc)[:120]}
    before = evaluate(nets, hold_ex, reg)
    reports = []
    import copy

    import torch
    opt = torch.optim.Adam(nets["params"], lr=5e-4, weight_decay=1e-4)
    best_val, best_state, best_epoch, stale_epochs = None, None, 0, 0
    for ep in range(epochs):
        rng.shuffle(train_ex)
        reports.append(s1.train_step(nets, train_ex, reg, lr=5e-4, opt=opt))
        if not val_ex:
            continue
        v = evaluate(nets, val_ex, reg)["spearman"]
        if best_val is None or v > best_val:
            best_val, best_epoch, stale_epochs = v, ep + 1, 0
            best_state = (copy.deepcopy(nets["encoder"].state_dict()),
                          copy.deepcopy(nets["heads"].state_dict()))
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:      # report/save the best model, not the last
        nets["encoder"].load_state_dict(best_state[0])
        nets["heads"].load_state_dict(best_state[1])
    after = evaluate(nets, hold_ex, reg)
    OUT.mkdir(parents=True, exist_ok=True)
    # ACCEPTANCE GATE. This used to overwrite unconditionally, so a refresh
    # that measurably degraded the net still replaced it -- and campaigns
    # call refresh() every generation, which means the value head was being
    # walked downhill run after run. Measured on 355 targets: incumbent
    # hold-out spearman 0.8021, retrained 0.7731 even with early stopping,
    # and from-scratch tops out at 0.7238. The incumbent carries learning
    # from far more data than any single snapshot holds, so the honest
    # policy is to keep it unless the challenger actually wins.
    improved = (after["spearman"] > before["spearman"]
                and (after["pairwise_accuracy"] or 0)
                >= (before["pairwise_accuracy"] or 0))
    had_ckpt = CKPT.is_file()
    backup = None
    if improved or not had_ckpt:
        if had_ckpt:
            backup = CKPT.with_name(f"policy_value_ep0.pre_postsizing_"
                                    f"{time.strftime('%Y%m%d_%H%M%S')}.pt")
            backup.write_bytes(CKPT.read_bytes())
        s1.save_checkpoint(nets, CKPT, {
            "refresh": "post_sizing_value_targets",
            "targets_total": len(entries), "train": len(train_ex),
            "holdout": len(hold_ex), "epochs": epochs, "seed": seed,
            "stale_meta": {k: str(v)[:80]
                           for k, v in (stale_meta or {}).items()},
            "timestamp": time.time()})
    report = {"targets_total": len(entries), "train": len(train_ex),
              "split_policy": "content_hash_stable",
              "val": len(val_ex), "epochs_run": len(reports),
              "best_epoch_by_val": best_epoch,
              "best_val_spearman": (round(best_val, 4)
                                    if best_val is not None else None),
              "early_stopped": len(reports) < epochs,
              "optimizer": "persistent_adam",
              "counterfactual_included": include_counterfactual,
              "counterfactual_targets": sum(
                  1 for e in entries if e.get("counterfactual")),
              "mismatch_discrimination": _discrimination(nets, entries, reg),
              "policy_visit_records_available": len(_policy_examples()),
              "holdout": len(hold_ex), "epochs": epochs,
              "stale_checkpoint_eval": before,
              "retrained_eval": after,
              "value_loss_first_last": [reports[0]["value_loss"],
                                        reports[-1]["value_loss"]],
              "checkpoint": str(CKPT), "backup": str(backup),
              "improved": improved,
              "checkpoint_written": bool(improved or not had_ckpt),
              "acceptance_gate": ("accepted: beat the incumbent on holdout"
                                  if improved else
                                  "rejected: incumbent kept (refresh is "
                                  "non-destructive)"),
              "families": sorted({e["topology_family"] for e in entries})}
    (OUT / "REPORT.json").write_text(json.dumps(report, indent=1),
                                     encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--counterfactual", action="store_true",
                    help="include measured wrong-tier targets (off by "
                         "default so past ablations keep their value net)")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    print(json.dumps(refresh(epochs=a.epochs,
                             include_counterfactual=a.counterfactual),
                     indent=1))
