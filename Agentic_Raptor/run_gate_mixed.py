"""POOL-COMPOSITION GATE for sft_adapter_tier2_mixed (2026-08-18).
Prompts the retrained proposer on 6 tier-2 heldout specs and checks the
pool it ACTUALLY emits: must contain BOTH forecast-winning classes AND
near-miss losers on >= 4/6 specs. Winners-only => the ablation
re-saturates (the mistake of the first tier-2 corpus); no losers => FAIL."""
import json
from pathlib import Path
from run_qwen_ablation import _load
from run_raptor_v2 import propose_and_validate, ig
from agentic_raptor.publication.tier2 import load_tier2_specs
ADAPTER = "artifacts/publication_v2/proposer_repair/sft_adapter_tier2_mixed"
WINNERS = {"3s_rc_ab", "4s_rc", "4s_rc_ab", "4s_rc_cas"}
tok, model = _load(ADAPTER)
ok = 0; detail = []
for s in load_tier2_specs("tier2_heldout")[:6]:
    spec = ig.parse_spec(s["prompt"]); spec["spec_id"] = s["spec_id"]
    out = propose_and_validate(model, tok, s["prompt"], target_k=5, conditioning="exclusion",
                               seed0=0, spec=spec, stall_stop=None)
    fams = sorted({c["canonical_family"] for c in out["candidates"]})
    w = [f for f in fams if f in WINNERS]; l = [f for f in fams if f not in WINNERS]
    mixed = bool(w) and bool(l)
    ok += mixed
    detail.append({"spec": s["spec_id"], "families": fams, "winners": w, "losers": l, "mixed": mixed})
    print(f"{s['spec_id']:26s} pool={fams}  winners={len(w)} losers={len(l)} -> {'MIXED' if mixed else 'NOT MIXED'}", flush=True)
verdict = "PASS" if ok >= 4 else "FAIL"
Path("artifacts/publication_v3/tier2/SFT_MIXED_GATE.json").write_text(
    json.dumps({"adapter": ADAPTER, "mixed_specs": ok, "of": 6, "detail": detail, "verdict": verdict}, indent=1), encoding="utf-8")
print(f"\nGATE: {ok}/6 specs emit a MIXED pool -> {verdict}", flush=True)
