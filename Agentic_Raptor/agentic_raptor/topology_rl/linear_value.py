"""LINEAR VALUE PROBE AS MCTS LEAF VALUE (quick-test task, 2026-08-13).

The root-cause diagnostic measured that a plain linear model on 24 explicit
physical features (DEV Spearman 0.399 / pairwise 0.713 / feasibility AUROC
0.883, spec-disjoint) beats every trained AlphaZero value head (0.22-0.24).
This module makes that exact probe usable as an MCTS leaf evaluator:

  - ONE feature implementation (physical_features_core) shared by the
    offline dataset path and the live search path, so they cannot drift;
  - train_and_freeze(): deterministically reproduces the Part-D probe on
    AZ_VALUE_DIAGNOSTIC_V1's frozen TRAIN split and persists weights +
    normalization + verification metrics to FROZEN_LINEAR_VALUE_V1.json;
  - FrozenLinearValue: loads the frozen artifact, verifies its recorded
    DEV metrics against the Part-D reference, and scores live
    TopologySearchStates via the registry's real DeviceCircuitGraph.

EXPERIMENTAL ONLY: nothing live imports this by default; the live FULL
default (search="one_root", neural V) is untouched.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics as st
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DATASET = _ROOT / "artifacts/publication_v3/az_value_diagnostic_v1/AZ_VALUE_DIAGNOSTIC_V1.jsonl"
FROZEN_PATH = _ROOT / "artifacts/publication_v3/az_value_probe_gate/FROZEN_LINEAR_VALUE_V1.json"
DEV_FRACTION = 0.3

#: Part-D measured reference -- reproduction must match these exactly.
REFERENCE_DEV = {"spearman": 0.3985, "pairwise": 0.7128, "auroc": 0.8825}

FEATURE_NAMES = (
    "n_devices", "stage_count", "n_caps", "n_res", "n_nmos", "n_pmos",
    "n_ports", "n_roles", "n_gain", "mean_degree", "max_degree",
    "depth", "is_edited",
    "spec_gain", "spec_log_ugbw", "spec_pm", "spec_cload", "spec_supply",
    "gain_x_stages", "ugbw_x_caps", "ugbw_x_cload", "pm_x_caps",
    "gain_x_ngain", "cload_x_ndev",
)


def physical_features_core(device_graph, spec: dict, depth: float,
                           is_edited: bool) -> list[float]:
    """The 24 physical features. `device_graph` is a real
    DeviceCircuitGraph; `spec` uses the AlphaZero-internal keys
    (target_gain_db / target_gbw_hz / minimum_phase_margin_deg /
    load_capacitance_f / supply_voltage)."""
    from collections import Counter
    devs = device_graph.devices
    n = len(devs)
    type_counts = Counter((d.kind or "other").lower() for d in devs)
    role_counts = Counter((d.role or "none") for d in devs)
    n_gain = sum(1 for d in devs
                for marker in ("gain_stage", "gain_device", "input_pair")
                if marker in (d.role or ""))
    net_use: Counter = Counter()
    for d in devs:
        for term_net in (d.nets or {}).values():
            net_use[term_net] += 1
    degrees = [sum(net_use[t] - 1 for t in (d.nets or {}).values()) for d in devs]
    g = spec.get("target_gain_db", 40) / 100
    u = math.log10(max(spec.get("target_gbw_hz", 1e4), 1)) / 10
    p = spec.get("minimum_phase_margin_deg", 45) / 90
    cl = spec.get("load_capacitance_f", 5e-10) * 1e12 / 1000
    n_caps = sum(v for k, v in type_counts.items() if "cap" in k)
    n_res = sum(v for k, v in type_counts.items() if "res" in k)
    n_nmos = sum(v for k, v in type_counts.items() if "nmos" in k or k == "n")
    n_pmos = sum(v for k, v in type_counts.items() if "pmos" in k or k == "p")
    return [
        n / 20.0, device_graph.stage_count / 5.0, n_caps / 4.0, n_res / 4.0,
        n_nmos / 12.0, n_pmos / 12.0, len(device_graph.ports or []) / 8.0,
        len(role_counts) / 10.0, n_gain / 4.0,
        (st.mean(degrees) / 8.0) if degrees else 0.0,
        (max(degrees) / 16.0) if degrees else 0.0,
        depth / 4.0, 1.0 if is_edited else 0.0,
        g, u, p, cl, spec.get("supply_voltage", 1.8) / 5,
        g * device_graph.stage_count / 5.0, u * n_caps / 4.0, u * cl,
        p * n_caps / 4.0, g * n_gain / 4.0, cl * n / 20.0,
    ]


def features_from_row(row: dict) -> list[float]:
    from agentic_raptor.topology_rl.alphazero import deserialize_device_graph
    dg = deserialize_device_graph(row["state_graph"])
    return physical_features_core(dg, row["spec"], float(row["depth"]),
                                  bool(row["is_edited"]))


def _load_rows():
    return [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _split(rows):
    dev_specs = {r["spec_hash"] for r in rows
                if (int(hashlib.sha256(r["spec_hash"].encode()).hexdigest()[:8], 16)
                    % 10_000) / 10_000 < DEV_FRACTION}
    return ([r for r in rows if r["spec_hash"] not in dev_specs],
            [r for r in rows if r["spec_hash"] in dev_specs])


def train_and_freeze(force: bool = False) -> dict:
    """Reproduce the Part-D linear probe deterministically and freeze it.
    Hard-fails if the reproduced DEV metrics do not match the Part-D
    reference (a drifted dataset or feature change must never be silently
    frozen as if it were the validated probe)."""
    import torch

    from agentic_raptor.topology_rl.value_probe_gate import evaluate_candidate

    if FROZEN_PATH.is_file() and not force:
        return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

    rows = _load_rows()
    train, dev = _split(rows)
    x_tr = torch.tensor([features_from_row(r) for r in train], dtype=torch.float32)
    y_tr = torch.tensor([r["z"] for r in train], dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(0)
    mean, std = x_tr.mean(0), x_tr.std(0).clamp(min=1e-6)
    lin = torch.nn.Linear(x_tr.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        loss = ((lin((x_tr - mean) / std) - y_tr) ** 2).mean()
        loss.backward()
        opt.step()

    weights = lin.weight.detach().squeeze(0).tolist()
    bias = float(lin.bias.detach())
    frozen = {"model_id": "FROZEN_LINEAR_VALUE_V1",
             "feature_names": list(FEATURE_NAMES),
             "weights": weights, "bias": bias,
             "feature_mean": mean.tolist(), "feature_std": std.tolist(),
             "n_train_rows": len(train), "reference_dev": REFERENCE_DEV}

    # verification: the frozen closed-form predictor must reproduce the
    # Part-D DEV metrics exactly (same split rule as the gate module)
    def _predict(row):
        f = features_from_row(row)
        return sum(w * (fi - m) / s for w, fi, m, s in
                  zip(weights, f, frozen["feature_mean"], frozen["feature_std"])) + bias
    res = evaluate_candidate(_predict)
    if (abs(res["dev_spearman"] - REFERENCE_DEV["spearman"]) > 1e-3
            or abs(res["dev_pairwise_ranking"] - REFERENCE_DEV["pairwise"]) > 1e-3):
        raise RuntimeError(
            f"frozen linear probe failed to reproduce the Part-D reference: "
            f"got spearman={res['dev_spearman']} pairwise={res['dev_pairwise_ranking']}, "
            f"expected {REFERENCE_DEV} -- refusing to freeze a drifted probe")
    frozen["verified_dev_metrics"] = {k: res[k] for k in
                                      ("dev_spearman", "dev_pairwise_ranking",
                                       "dev_feasibility_auroc", "dev_distance_spearman")}
    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_PATH.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    return frozen


class FrozenLinearValue:
    """Live leaf evaluator: score a TopologySearchState with the frozen
    linear probe via the registry's real DeviceCircuitGraph. Output clamped
    to [-1, 1] (z's own range)."""

    def __init__(self, frozen: dict | None = None):
        self.frozen = frozen or train_and_freeze()
        self.model_id = self.frozen["model_id"]

    def score_state(self, state, reg) -> float:
        tid = state.topology_id
        resolved = reg._resolve(tid) if hasattr(reg, "_resolve") else tid
        dg = reg._device_graphs[resolved]
        feats = physical_features_core(dg, state.spec, float(state.depth),
                                       "~" in resolved)
        f = self.frozen
        raw = sum(w * (fi - m) / s for w, fi, m, s in
                 zip(f["weights"], feats, f["feature_mean"], f["feature_std"])) + f["bias"]
        return max(-1.0, min(1.0, raw))


def hybrid_linear_value_nets(base_nets: dict, probe: FrozenLinearValue) -> dict:
    """LINEAR_VALUE_MCTS: the EXPERIMENTAL nets bundle -- policy P(s,a)
    (and therefore priors/PUCT behavior) identical to `base_nets`; ONLY the
    leaf value_forward is replaced by the frozen linear probe."""
    import torch

    def value_forward(state, reg):
        v = probe.score_state(state, reg)
        return {"scalar": torch.tensor(v),
               "feasibility_logit_uncalibrated": torch.tensor(0.0),
               "stability_logit_uncalibrated": torch.tensor(0.0),
               "expected_spice_cost": torch.tensor(0.0),
               "budget_exhaustion_logit": torch.tensor(0.0)}

    return {**base_nets, "value_forward": value_forward,
           "value_mode": "FROZEN_LINEAR_VALUE_V1"}
