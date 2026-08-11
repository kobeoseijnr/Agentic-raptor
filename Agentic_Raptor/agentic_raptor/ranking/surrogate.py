"""Part 5: genuine post-SAC surrogate inference.

`sac_size` already trains an internal surrogate on every measurement it takes
(`spec_sizing.py`, the `surrogate` MLP over normalised knobs -> pm/gain). The
repair is to EXPOSE that trained model's frozen predictive interface, rather
than copying the most recent SPICE result and calling it a prediction.

Hard rule: `predict_post_sac` never touches the ngspice executable. It takes
a sizing vector and returns estimates. If the surrogate is unavailable the
prediction is returned with explicit UNKNOWNs -- unknown must never be
silently upgraded to good.
"""

from __future__ import annotations

import math
from pathlib import Path

from agentic_raptor.ranking.types import (SurrogatePrediction,
                                          checkpoint_sha256)

_ROOT = Path(__file__).resolve().parents[2]


def _load_family_surrogate(family: str, path=None):
    """Frozen surrogate persisted by sac_size.

    `path` is THIS run's surrogate, written beside the sizing run. Prefer it:
    the per-family checkpoint only exists when persist=True, and the canonical
    pipeline sizes with persist=False, so without it every prediction came
    back UNKNOWN and the ranker had nothing to compare.

    Identity is the SHA-256 of the checkpoint file, not a tensor-sum digest:
    sums collide under permutation, so they cannot distinguish two different
    models and would give false provenance.
    """
    from agentic_raptor.mb_sac.spec_sizing import N_KNOBS, _mem_paths
    try:
        import torch
        src = Path(path) if path else _mem_paths(family)["surrogate"]
        if not src.is_file():
            return None, None
        mp = {"surrogate": src}
        net = torch.nn.Sequential(torch.nn.Linear(N_KNOBS, 32),
                                  torch.nn.ReLU(), torch.nn.Linear(32, 3))
        # A stale 2-output checkpoint (pre-UGBW) raises a shape mismatch here,
        # which the caller's try/except turns into "surrogate unavailable" --
        # not a crash, and not a silent wrong-shaped load.
        net.load_state_dict(torch.load(mp["surrogate"], weights_only=True))
        net.eval()                                  # frozen for inference
        for p in net.parameters():
            p.requires_grad_(False)
        return net, checkpoint_sha256(mp["surrogate"])
    except Exception:
        return None, None


def _normalized_margins(spec: dict, gain_db, pm_deg, ugbw_hz) -> dict:
    """Margins normalised so >= 0 means satisfied.

    Same scales the outcome classifier uses, so predicted and measured
    margins are directly comparable.
    """
    m = {}
    if gain_db is not None and spec.get("gain_target_db") is not None:
        m["gain"] = (gain_db - spec["gain_target_db"]) / 20.0
    if pm_deg is not None and spec.get("phase_margin_target_deg") is not None:
        m["pm"] = (pm_deg - spec["phase_margin_target_deg"]) / 45.0
    ut = spec.get("ugbw_target_hz")
    if ugbw_hz is not None and ut:
        m["ugbw"] = math.log10(max(ugbw_hz, 1.0) / max(ut, 1.0)) / 2.0
    return m


#: Monte-Carlo dropout passes used to estimate predictive uncertainty. The
#: surrogate is a plain MLP, so dropout is injected at inference only.
MC_PASSES = 16
MC_DROPOUT_P = 0.1


def _mc_dropout_predict(net, x, passes: int = MC_PASSES):
    """Predictive mean and spread from MC dropout.

    Part 2 forbids a constant such as `predictive_uncertainty = 0.25`. This
    derives it: the spread of `passes` stochastic forward passes, normalised
    to the output scale. A design far from the surrogate's training
    distribution produces a wide spread, which is exactly the signal the
    ranker needs to distrust it.
    """
    import torch
    outs = []
    with torch.no_grad():
        for _ in range(passes):
            h = x
            for layer in net:
                h = layer(h)
                if isinstance(layer, torch.nn.ReLU):
                    mask = (torch.rand_like(h) > MC_DROPOUT_P).float()
                    h = h * mask / (1.0 - MC_DROPOUT_P)
            outs.append(h)
    stack = torch.stack(outs)
    mean = stack.mean(0)
    std = stack.std(0)
    return mean, std


def predict_post_sac(spec: dict, topology_hash: str, topology_family: str,
                     sizing_vector: dict, sizing_manifest_hash: str,
                     checkpoint_family: str | None = None,
                     uncertainty_method: str = "mc_dropout",
                     surrogate_path=None) -> SurrogatePrediction:
    """Predict the electrical outcome of ONE sized design. Never calls ngspice.

    Anything the surrogate cannot predict is returned as None. It is a
    THREE-output model (PM, gain, UGBW): UGBW was added because the ranker
    choosing which design gets verified was previously blind to the
    constraint responsible for most measured failures. Power and area are
    still UNKNOWN and must stay so -- `predicted_feasible_for(spec)` treats a
    missing required constraint as "cannot say", which is the honest answer.
    """
    from agentic_raptor.mb_sac.spec_sizing import (KNOB_HI, KNOB_NAMES,
                                                   UGBW_LOG_HI, UGBW_LOG_LO)
    fam = checkpoint_family or topology_family
    net, ckpt_hash = _load_family_surrogate(fam, surrogate_path)
    gain = pm = ugbw = None
    unc = None
    op_p = stab_p = None
    if net is not None and sizing_vector:
        try:
            import torch
            hi = torch.tensor(KNOB_HI)
            knobs = torch.tensor([float(sizing_vector.get(k, 1.0))
                                  for k in KNOB_NAMES])
            x = knobs / hi

            def _decode_ugbw(v: float) -> float | None:
                # inverse of _ugbw_target: v<=0 means "at/below the band
                # floor" -- report unknown rather than a specific tiny Hz
                # value the model was never actually trained to mean
                if v <= 1e-6:
                    return None
                log = UGBW_LOG_LO + max(0.0, min(1.0, v)) * (
                    UGBW_LOG_HI - UGBW_LOG_LO)
                return float(10 ** log)

            if uncertainty_method == "mc_dropout":
                mean, std = _mc_dropout_predict(net, x)
                pm = float(mean[0]) * 90.0
                gain = float(mean[1]) * 100.0
                ugbw = _decode_ugbw(float(mean[2]))
                # normalised spread across all three heads, clipped to [0, 1]
                unc = float(min(1.0, (float(std[0]) + float(std[1])
                                      + float(std[2])) / 3.0))
            else:                      # explicitly named baseline arm
                with torch.no_grad():
                    out = net(x)
                pm, gain = float(out[0]) * 90.0, float(out[1]) * 100.0
                ugbw = _decode_ugbw(float(out[2]))
                unc = None
            # stability probability derives from the predicted margin and its
            # spread, not from a hard-coded constant
            if uncertainty_method == "mc_dropout":
                pm_std_deg = max(float(std[0]) * 90.0, 1e-6)
                z = pm / pm_std_deg
                stab_p = float(1.0 / (1.0 + math.exp(-z)))
            else:
                stab_p = 1.0 if pm > 0 else 0.0
        except Exception:
            gain = pm = ugbw = unc = stab_p = None
    # the surrogate models no operating-point head: UNKNOWN, not 0.9
    op_p = None
    margins = _normalized_margins(spec, gain, pm, ugbw)
    return SurrogatePrediction(
        topology_hash=topology_hash,
        sizing_manifest_hash=sizing_manifest_hash,
        gain_db=gain, pm_deg=pm,
        ugbw_hz=ugbw,
        power_w=None,        # not modelled -> unknown
        area_um2=None,       # not modelled -> unknown
        operating_point_probability=op_p,
        stability_probability=stab_p,
        normalized_margins=margins,
        predictive_uncertainty=unc,
        surrogate_checkpoint_hash=ckpt_hash,
        feature_schema_version=f"post_sac_surrogate.v2_ugbw+{uncertainty_method}")
