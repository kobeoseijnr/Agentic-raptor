"""Topology-baseline experiment: CktGen + AnalogGenie vs RAPTOR.

Parts (run in order):
  map          deterministic HELDOUT29 -> OCB-101 constraint mapping (CPU, seconds)
  cktgen-gen   CktGen generation via their MAB, child process   (GPU, ~30-60s/spec)
  genie-gen    AnalogGenie generation via their GPT, seeded     (GPU, ~40s/sample)
  evaluate     realization + common 65-call sizing + ngspice + VT corners (CPU)
  summary      aggregate 8-column comparison table

Protocol (metric_definitions.md): per spec per seed the method's first-5
candidates (its OWN ranking order, fixed before any SPICE) each receive a
13-call common tune (5 x 13 = 65 = the Track-A budget); the returned design is
the best (pass, fom) among them -- the same use-online-SPICE-to-pick privilege
AG's own pipeline has. P@5 = any of the 5 passing. VT corners (4) run on the
returned design only, tracked separately. All ngspice calls go through the
central counter fields. No baseline code is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation.external_baselines.run_stage6_v2_tuner import (  # noqa: E402
    CONTROL, CORNERS, run_deck, score, perturb)

ART = ROOT / "artifacts" / "topology_baselines"
GEN = ART / "generation"
import os as _os
RESULTS = ART / _os.environ.get("TOPO_RESULTS", "results.jsonl")
PY = sys.executable
CKTGEN_REPO = ROOT / "external_baselines" / "CktGen"
GENIE_REPO = ROOT / "external_baselines" / "AnalogGenie"
TT_SPICE = (r"C:\Users\kobeo\OneDrive\Desktop\raptor1\RAPTOR_Legacy\AnalogGym"
            r"\RGNN_RL\mosfet_model\sky130_pdk\sky130_pdk\libs.tech\ngspice"
            r"\corners\tt.spice")
BUDGET_TOTAL = 65
PER_CAND = 13
TOP_K = 5

# CktBench-101 node-type table (verified: START_TYPE=0/END_TYPE=1 set in their
# get_datasets.py; their is_valid_Circuit comment marks types 8,9 = R,C;
# 2..7 are the six gm-block variants -- only gm-membership is used here).
GM_TYPES = {2, 3, 4, 5, 6, 7}
R_TYPE, C_TYPE = 8, 9
MAIN_PATH = {2, 3, 4}   # their 'path' positions forming the main signal path

AMP_EXCLUDE = re.compile(
    r"^(VCLK|VCONT|VTRACK|VHOLD|VLO|VRF|VIF|LOGIC|XOR|PFD|INVERTER|"
    r"TRANSMISSION_GATE|VLATCH|VCM|VREF)" )
GENIE_UNREALIZABLE = re.compile(r"^(NPN|PNP|L\d|DIO)")


def specs29():
    d = json.loads((ROOT / "data/external_baseline_eval/specs_validation.json"
                    ).read_text(encoding="utf-8"))
    return {s["spec_index"]: s["parsed_spec"] for s in d["specs"]}


# ---------------------------------------------------------------- part: map
def part_map():
    import statistics
    sp = specs29()
    rows = [(i, p["gain_target_db"], p["ugbw_target_hz"],
             p["phase_margin_target_deg"]) for i, p in sorted(sp.items())]
    perf = []
    with open(CKTGEN_REPO / "dataset/OCB/CktBench101/perform101.csv",
              encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for k, r in enumerate(rd):
            if k >= 9000:      # their test split
                try:
                    perf.append((float(r["gain"]), float(r["bw"]),
                                 float(r["pm"])))
                except ValueError:
                    pass
    def quant(vals, q):
        s = sorted(vals)
        return s[min(len(s) - 1, max(0, int(q * len(s))))]
    out = []
    for axis in range(3):
        tgt = [r[1 + axis] for r in rows]
        ocb = [p[axis] for p in perf]
        ranks = {v: i for i, v in enumerate(sorted(tgt))}
        out.append({rows[j][0]: math.floor(quant(ocb, (ranks[tgt[j]] + 0.5)
                                                 / len(tgt)))
                    for j in range(len(rows))})
    GEN.mkdir(parents=True, exist_ok=True)
    mapping = [{"spec_index": i,
                "target": {"gain_db": sp[i]["gain_target_db"],
                           "ugbw_hz": sp[i]["ugbw_target_hz"],
                           "pm_deg": sp[i]["phase_margin_target_deg"]},
                "cons": [out[0][i], out[1][i], out[2][i]]}
               for i in sorted(sp)]
    (ART / "spec_mapping.json").write_text(json.dumps(mapping, indent=1),
                                           encoding="utf-8")
    with open(ART / "spec_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["spec_index", "gain_db", "ugbw_hz", "pm_deg",
                    "ocb_gain", "ocb_bw", "ocb_pm"])
        for m in mapping:
            w.writerow([m["spec_index"], m["target"]["gain_db"],
                        m["target"]["ugbw_hz"], m["target"]["pm_deg"], *m["cons"]])
    print(f"mapped {len(mapping)} specs -> spec_mapping.csv/json")


# ---------------------------------------------------------- part: cktgen-gen
def part_cktgen_gen(seed: int):
    mapping = json.loads((ART / "spec_mapping.json").read_text(encoding="utf-8"))
    cons_file = GEN / f"cktgen_cons_s{seed}.json"
    cons_file.write_text(json.dumps(mapping), encoding="utf-8")
    out = GEN / f"cktgen_s{seed}.json"
    env = dict(__import__("os").environ, PYTHONUTF8="1",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    r = subprocess.run(
        [PY, str(ROOT / "src/evaluation/external_baselines/_cktgen_gen_runner.py"),
         str(CKTGEN_REPO), str(seed), str(cons_file), str(out)],
        env=env)
    print("cktgen-gen rc:", r.returncode)


# ----------------------------------------------------------- part: genie-gen
def part_genie_gen(seed: int, need: int = TOP_K, cap: int = 120):
    import torch
    sys.path.insert(0, str(GENIE_REPO))
    src = (GENIE_REPO / "Inference.py").read_text(encoding="utf-8")
    ns = {}
    exec(src[:src.index("model = GPTLanguageModel")], ns)   # their vocab, verbatim
    from Models.GPT import GPTLanguageModel
    model = GPTLanguageModel(ns["vocab_size"], ns["n_embd"], ns["block_size"],
                             ns["n_head"], ns["n_layer"], ns["dropout"])
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(str(GENIE_REPO / "Pretrain.pth"),
                                     map_location=dev), strict=False)
    m = model.to(dev).eval()
    torch.manual_seed(seed)
    GEN.mkdir(parents=True, exist_ok=True)
    kept, tried, t0 = [], 0, time.time()
    while len(kept) < need and tried < cap:
        tried += 1
        ctx = torch.full((1, 1), 1003, dtype=torch.long, device=dev)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts = time.time()
        seq = m.generate(ctx, max_new_tokens=1024)[0].tolist()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        walk = [ns["itos"][i] for i in seq]
        if "TRUNCATE" in walk:
            walk = walk[:walk.index("TRUNCATE")]
        toks = set(walk)
        amp = (any(t.startswith("VIN") and "_" not in t for t in toks)
               and any(t.startswith("VOUT") and "_" not in t for t in toks)
               and not any(AMP_EXCLUDE.match(t) for t in toks))
        dec = decode_walk(walk) if amp else None
        ok = bool(dec and dec["devices"] and not dec["unrealizable"])
        kept.append({"rank": len(kept) + 1, "walk": walk, "sample_no": tried,
                     "gen_s": round(time.time() - ts, 2)}) if ok else None
        print(f"[genie s{seed}] sample {tried}: amp={amp} realizable={ok} "
              f"kept={len(kept)}", flush=True)
    (GEN / f"genie_s{seed}.json").write_text(json.dumps(
        {"seed": seed, "tried": tried, "kept": kept,
         "total_gen_s": round(time.time() - t0, 1)}), encoding="utf-8")
    print(f"genie-gen s{seed}: kept {len(kept)}/{tried} in "
          f"{time.time() - t0:.0f}s")


# ------------------------------------------------------ decoding/realization
def decode_walk(walk):
    pin_re = re.compile(r"^([A-Z_]+\d+)_[A-Z]+\d*$")
    edges = set()
    for a, b in zip(walk, walk[1:]):
        if a != b:
            edges.add((a, b) if a < b else (b, a))
    devices, net_edges = {}, []
    for a, b in edges:
        ma, mb = pin_re.match(a), pin_re.match(b)
        if ma and ma.group(1) == b:
            devices.setdefault(b, set()).add(a)
        elif mb and mb.group(1) == a:
            devices.setdefault(a, set()).add(b)
        else:
            net_edges.append((a, b))
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in net_edges:
        parent[find(a)] = find(b)
    def net_of(pin):
        r = find(pin)
        members = [n for n in parent if find(n) == r]
        for t in members:            # prefer a terminal name for the net
            if "_" not in t or t.split("_")[0] not in devices:
                if not pin_re.match(t):
                    return t.lower()
        return "n_" + re.sub(r"[^A-Za-z0-9]", "", sorted(members)[0]).lower()
    unreal = [d for d in devices if GENIE_UNREALIZABLE.match(d)]
    return {"devices": devices, "net_of": net_of, "unrealizable": unreal}


def genie_netlist(walk, cl_pf):
    """Decoded walk -> flat sky130 deck body + source lines. None if invalid."""
    dec = decode_walk(walk)
    if not dec or dec["unrealizable"] or not dec["devices"]:
        return None
    net = dec["net_of"]
    body, bias = [], []
    for d, pins in sorted(dec["devices"].items()):
        pin = {p.split("_", 1)[1]: net(p) for p in pins}
        if d.startswith(("NM", "PM")):
            if not {"D", "G", "S"} <= set(pin):
                return None
            b = pin.get("B", "vss" if d.startswith("NM") else "vdd")
            mdl = ("sky130_fd_pr__nfet_01v8" if d.startswith("NM")
                   else "sky130_fd_pr__pfet_01v8")
            w = 5.0 if d.startswith("NM") else 10.0
            body.append(f"x{d} {pin['D']} {pin['G']} {pin['S']} {b} {mdl} "
                        f"l=0.5 w={w} m=1")
        elif d.startswith("R"):
            body.append(f"r{d} {pin.get('P','0')} {pin.get('N','0')} 10k")
        elif d.startswith("C"):
            body.append(f"c{d} {pin.get('P','0')} {pin.get('N','0')} 1p")
        else:
            return None
    for t in sorted({t for t in set(walk)
                     if re.match(r"^(VB|IB)\d+$", t)}):
        if t.startswith("VB"):
            bias.append(f"V{t} {t.lower()} 0 0.9")
        else:
            bias.append(f"I{t} vdd {t.lower()} 10u")
    return "\n".join(body + bias)


def genie_builder(walk_tokens, cl_pf):
    has_vin2 = any(t == "VIN2" for t in walk_tokens)
    def build(body, vdd_scale=1.0, temp_c=None):
        vdd = 1.8 * vdd_scale
        deck = ["* topo-baseline genie deck",
                f".include {TT_SPICE}",
                ".param mc_mm_switch=0", ".param mc_pr_switch=0",
                f"V1 vdd 0 {vdd}", "V2 vss 0 0",
                f"Vinp vin1 0 dc {0.5 * vdd:.4g} ac 1"]
        if has_vin2:
            deck.append(f"Vinn vin2 0 dc {0.5 * vdd:.4g}")
        deck += [body, f"CLOAD_TB vout1 0 {cl_pf}p"]
        if temp_c is not None:
            deck.append(f".temp {temp_c}")
        deck += [c.format(out="vout1") for c in CONTROL] + [".end"]
        return "\n".join(deck)
    return build


def cktgen_realize(graph):
    from agentic_raptor.mapping import (map_family, emit_netlist,
                                        static_validate)
    types = [int(v.get("type", -1)) for v in graph["vs"]]
    paths = [int(v.get("path", -1)) for v in graph["vs"]]
    gm_main = sum(1 for t, p in zip(types, paths)
                  if t in GM_TYPES and p in MAIN_PATH)
    gm_any = sum(1 for t in types if t in GM_TYPES)
    n = gm_main or gm_any
    if n == 0:
        return None, "no_gm_stage"
    blocks = []
    if any(t == C_TYPE for t in types):
        blocks.append("C")
    if any(t == R_TYPE for t in types) and "C" in blocks:
        blocks.append("RC_series")
    audit = {"gain_stages": min(n, 4), "mapping_readiness": "external_cktgen",
             "functional_blocks": blocks, "unresolved_blocks": []}
    entry = SimpleNamespace(topology_id="cktgen_ext")
    g, status = map_family(entry, audit)
    if g is None:
        return None, status
    net = emit_netlist(g, "extckt")
    sv = static_validate(g, net)
    if sv["status"] != "mapped_static_valid":
        return None, "static_invalid:" + ";".join(sv["problems"])
    return net, "ok"


def subckt_builder(cl_pf):
    def build(body, vdd_scale=1.0, temp_c=None):
        vdd = 1.8 * vdd_scale
        deck = ["* topo-baseline subckt deck", body,
                f".include {TT_SPICE}",
                ".param mc_mm_switch=0", ".param mc_pr_switch=0",
                f"V1 vdd 0 {vdd}", "V2 vss 0 0",
                f"Vindc opin 0 {0.5 * vdd:.4g}",
                f"Vin signal_in 0 dc {0.5 * vdd:.4g} ac 1",
                "Lfb opout opout_dc 1T", "Cin opout_dc signal_in 1T",
                "Xop1 vss vdd opout_dc opin opout extckt",
                f"Cload1 opout 0 {cl_pf}p"]
        if temp_c is not None:
            deck.append(f".temp {temp_c}")
        deck += [c.format(out="opout") for c in CONTROL] + [".end"]
        return "\n".join(deck)
    return build


# ----------------------------------------------- AnalogToBi (added 2026-08-30)
TOBI_REPO = ROOT / "external_baselines" / "AnalogToBi"
TOBI_PIN = {"M": {"B": "b", "D": "d", "G": "g", "S": "s"}}
TOBI_UNREALIZABLE = re.compile(r"^(NPN|PNP|L\d|DIO)")


def tobi_decode(walk):
    """Typed-edge bipartite walk -> {device: {pin: net}}. None on parse error."""
    devices: dict = {}
    dev_re = re.compile(r"^(NM|PM|NPN|PNP|R|C|L|DIO)\d+$")
    toks = [t for t in walk if t not in ("TRUNCATE",)
            and not t.startswith("CIRCUIT_")]
    for j in range(1, len(toks) - 1):
        t = toks[j]
        if "_" not in t or dev_re.match(t):
            continue
        fam, pins = t.split("_", 1)
        a, b = toks[j - 1], toks[j + 1]
        dev, net = (a, b) if dev_re.match(a) else (b, a)
        if not dev_re.match(dev) or dev_re.match(net):
            return None
        d = devices.setdefault(dev, {})
        if fam == "M":
            for ch in pins:
                d[TOBI_PIN["M"][ch]] = net
        elif fam == "B":
            for ch in pins:
                d[{"B": "base", "C": "col", "E": "emit"}[ch]] = net
        elif fam in ("R", "C", "L"):
            d.setdefault("terms", []).append(net)
        elif fam == "D":
            for ch in pins:
                d["anode" if ch == "P" else "cathode"] = net
    return devices


def tobi_netlist(walk, cl_pf):
    dec = tobi_decode(walk)
    if not dec:
        return None
    if any(TOBI_UNREALIZABLE.match(d) for d in dec):
        return None
    def net(n):
        return {"VSS": "vss", "VDD": "vdd"}.get(n, n.lower())
    body, bias = [], []
    for d, p in sorted(dec.items()):
        if d.startswith(("NM", "PM")):
            if not {"d", "g", "s"} <= set(p):
                return None
            b = p.get("b", "vss" if d.startswith("NM") else "vdd")
            mdl = ("sky130_fd_pr__nfet_01v8" if d.startswith("NM")
                   else "sky130_fd_pr__pfet_01v8")
            w = 5.0 if d.startswith("NM") else 10.0
            body.append(f"x{d} {net(p['d'])} {net(p['g'])} {net(p['s'])} "
                        f"{net(b)} {mdl} l=0.5 w={w} m=1")
        elif d.startswith("R") or d.startswith("C"):
            t = p.get("terms", [])
            if len(t) != 2:
                return None
            val = "10k" if d.startswith("R") else "1p"
            pre = "r" if d.startswith("R") else "c"
            body.append(f"{pre}{d} {net(t[0])} {net(t[1])} {val}")
        else:
            return None
    for t in sorted({t for t in set(walk) if re.match(r"^(VB|IB)\d+$", t)}):
        if t.startswith("VB"):
            bias.append(f"V{t} {t.lower()} 0 0.9")
        else:
            bias.append(f"I{t} vdd {t.lower()} 10u")
    return "\n".join(body + bias)


def tobi_builder(walk_tokens, cl_pf):
    outs = [t for t in walk_tokens if re.match(r"^VOUT\d*$", t)]
    out = outs[0].lower() if outs else "vout1"
    has_vin2 = any(t == "VIN2" for t in walk_tokens)
    def build(body, vdd_scale=1.0, temp_c=None):
        vdd = 1.8 * vdd_scale
        deck = ["* topo-baseline analogtobi deck",
                f".include {TT_SPICE}",
                ".param mc_mm_switch=0", ".param mc_pr_switch=0",
                f"V1 vdd 0 {vdd}", "V2 vss 0 0",
                f"Vinp vin1 0 dc {0.5 * vdd:.4g} ac 1"]
        if has_vin2:
            deck.append(f"Vinn vin2 0 dc {0.5 * vdd:.4g}")
        deck += [body, f"CLOAD_TB {out} 0 {cl_pf}p"]
        if temp_c is not None:
            deck.append(f".temp {temp_c}")
        deck += [c.format(out=out) for c in CONTROL] + [".end"]
        return build_dedup(deck)
    return build, out


def build_dedup(deck):
    return "\n".join(deck)


def part_tobi_gen(seed: int, need: int = TOP_K, cap: int = 200):
    import os
    import torch
    os.chdir(TOBI_REPO)
    sys.path.insert(0, str(TOBI_REPO))
    src = (TOBI_REPO / "GPT_Inference_Grammar.py").read_text(encoding="utf-8")
    ns = {"__name__": "analogtobi_verbatim"}
    exec(src[:src.index("run = 1000")], ns)      # their vocab+model+grammar, verbatim
    torch.manual_seed(seed)
    m, stoi, itos = ns["m"], ns["stoi"], ns["itos"]
    gen = ns["generate_with_masking_batch"]
    vss = stoi["VSS"]
    GEN.mkdir(parents=True, exist_ok=True)
    kept, tried, t0 = [], 0, time.time()
    while len(kept) < need and tried < cap:
        bsz = 8
        ctx = __import__("torch").tensor(
            [[stoi["CIRCUIT_Opamp"], vss]] * bsz, dtype=__import__("torch").long,
            device=next(m.parameters()).device)
        ts = time.time()
        seqs, _ = gen(m, ctx, max_new_tokens=1024, max_length=1020,
                      temperature=0.7)
        dt = time.time() - ts
        for seq in seqs:
            tried += 1
            walk = [itos[i] for i in (seq.tolist()
                                      if hasattr(seq, "tolist") else seq)]
            if "TRUNCATE" in walk:
                walk = walk[:walk.index("TRUNCATE")]
            toks = set(walk)
            amp = (any(re.match(r"^VIN\d+$", t) for t in toks)
                   and any(re.match(r"^VOUT\d*$", t) for t in toks))
            ok = bool(amp and tobi_netlist(walk, 10.0))
            if ok and len(kept) < need:
                kept.append({"rank": len(kept) + 1, "walk": walk,
                             "sample_no": tried,
                             "gen_s": round(dt / max(1, len(seqs)), 2)})
            print(f"[tobi s{seed}] sample {tried}: amp={amp} realizable={ok} "
                  f"kept={len(kept)}", flush=True)
        tried += 0 if seqs else bsz   # count fully-discarded batches
    (GEN / f"tobi_s{seed}.json").write_text(json.dumps(
        {"seed": seed, "tried": tried, "kept": kept,
         "total_gen_s": round(time.time() - t0, 1)}), encoding="utf-8")
    print(f"tobi-gen s{seed}: kept {len(kept)}/{tried} in "
          f"{time.time() - t0:.0f}s")


# --------------------------------------- AnalogCoder-Pro (reinstated 2026-09-05)
# Spec-aligned generation via the UNMODIFIED baseline (analogcoderpro_adapter,
# frozen commit 05542af) -> per-spec netlist pools; realization maps their
# level-1 PySpice netlists into the SAME sky130 1.8 V judge environment every
# other method uses. Documented uniform adaptations (model-parity rule):
#   * their generic nmos/pmos level-1 devices -> sky130 fd_pr fets (nfet/pfet
#     01v8), W preserved via w<=7um + integer m, L clamped >= 0.15 um;
#   * their 5 V testbench sources (supply + AC input) are DROPPED -- the
#     common builder supplies vdd=1.8 V and the standard 0.5*vdd AC drive,
#     exactly as for genie/tobi;
#   * internal DC bias sources are mapped RAIL-REFERENCED into the 1.8 V frame
#     (lower-half biases kept, upper-half biases keep their distance to the
#     top rail; clamped [0,1.8]) -- preserves overdrive since level-1 vto=0.5
#     ~ sky130 vth; they remain tunable through the vsrc knob (+/-0.2 V);
#   * R/C/I values kept verbatim (their chosen sizes = tuner starting point);
#   * L / diodes / controlled sources / .subckt blocks -> invalid attempt
#     (uniform realizability rule, same as GENIE_UNREALIZABLE).
ACP_REPO = ROOT / "external_baselines" / "AnalogCoderPro"
_ACP_NUM = r"[0-9.eE+-]+[a-zA-Z]*"


def _acp_val(tok: str) -> float:
    """SPICE number (optional scale suffix / trailing unit text) -> float."""
    m = re.match(r"([0-9.eE+-]+)([a-zA-Z]*)", tok)
    if not m:
        return 0.0
    v = float(m.group(1))
    suf = m.group(2).lower()
    for s, f in (("meg", 1e6), ("f", 1e-15), ("p", 1e-12), ("n", 1e-9),
                 ("u", 1e-6), ("m", 1e-3), ("k", 1e3), ("g", 1e9), ("t", 1e12)):
        if suf.startswith(s):
            return v * f
    return v


def acp_realize(sp_text: str, cl_pf: float):
    """AnalogCoder exported .sp -> (body, out_node, in_nodes) sky130 deck body.
    Returns (None, why, None, None) when the uniform rule rejects it."""
    models: dict[str, str] = {}
    for m in re.finditer(r"(?im)^\.model\s+(\S+)\s+(nmos|pmos)", sp_text):
        models[m.group(1).lower()] = m.group(2).lower()
    body, keep_v = [], []
    in_nodes: list[str] = []
    nets: set[str] = set()
    supply_nets: set[str] = set()

    def net(tok: str) -> str:
        t = tok.lower()
        if t in ("0", "gnd"):
            return "0"
        if t in supply_nets:
            return "vdd"
        nets.add(t)
        return t

    lines = [ln.strip() for ln in sp_text.splitlines()]
    # pass 1: find supply + input sources among V-lines
    vlines = []
    for ln in lines:
        if not ln or ln.startswith(("*", ".", "+")):
            continue
        tok = ln.split()
        k = tok[0][0].upper()
        if k == "V":
            vlines.append(tok)
    for tok in vlines:
        n1 = tok[1].lower()
        rest = " ".join(tok[3:]).lower()
        dcv = _acp_val(tok[3]) if len(tok) > 3 and not tok[3].lower().startswith(
            ("dc", "ac")) else (_acp_val(tok[4]) if len(tok) > 4 else 0.0)
        # NAME-ONLY supply detection (2026-09-06 fix, found via zero-pass
        # audit): the prior "or dcv >= 3.0" fallback misclassified ordinary
        # PMOS gate biases (typically ~Vdd-Vov, e.g. 3.7 V on their 5 V rail
        # -- completely normal) as extra supply rails, shorting them to vdd
        # and disabling the gain stage they drive. Audited all 434 generated
        # netlists: every genuine supply is literally named vdd/vcc/vpwr (434
        # matches); zero cases need a value-based fallback. Name-only is safe.
        if n1 in ("vdd", "vcc", "vpwr"):
            supply_nets.add(n1)
        elif " ac " in f" {rest} " or "ac" in rest.split():
            in_nodes.insert(0, n1)          # AC-driven input first
        elif any(s in n1 for s in ("vin", "inp", "inn", "vip", "vim")):
            in_nodes.append(n1)
    if not supply_nets:
        supply_nets.add("vdd")              # net literally named vdd, if any
    # pass 2: emit
    for ln in lines:
        if not ln or ln.startswith(("*", ".", "+")):
            if ln.lower().startswith(".subckt"):
                return None, "subckt_unsupported", None, None
            continue
        tok = ln.split()
        name, k = tok[0], tok[0][0].upper()
        if k == "M":
            if len(tok) < 6:
                return None, "bad_mosline", None, None
            mdl = models.get(tok[5].lower())
            if mdl is None:
                return None, f"unknown_model:{tok[5]}", None, None
            w_m = l_m = None
            for t in tok[6:]:
                if t.lower().startswith("w="):
                    w_m = _acp_val(t[2:])
                elif t.lower().startswith("l="):
                    l_m = _acp_val(t[2:])
            w_um = (w_m if w_m and w_m > 0.01 else (w_m or 5e-6) * 1e6)
            l_um = (l_m if l_m and l_m > 0.01 else (l_m or 1e-6) * 1e6)
            w = min(7.0, max(0.42, w_um))
            mm = max(1, round(w_um / w))
            l = max(0.15, l_um)
            sky = ("sky130_fd_pr__nfet_01v8" if mdl == "nmos"
                   else "sky130_fd_pr__pfet_01v8")
            d, g, s = net(tok[1]), net(tok[2]), net(tok[3])
            b = net(tok[4]) if len(tok) > 4 else ("0" if mdl == "nmos" else "vdd")
            body.append(f"x{name} {d} {g} {s} {b} {sky} "
                        f"l={l:.4g} w={w:.4g} m={mm}")
        elif k == "R":
            body.append(f"{name} {net(tok[1])} {net(tok[2])} "
                        f"{_acp_val(tok[3]):.6g}")
        elif k == "C":
            body.append(f"{name} {net(tok[1])} {net(tok[2])} "
                        f"{_acp_val(tok[3]):.6g}")
        elif k == "I":
            body.append(f"{name} {net(tok[1])} {net(tok[2])} "
                        f"{_acp_val(tok[3]):.6g}")
        elif k == "V":
            n1 = tok[1].lower()
            if n1 in supply_nets or n1 in in_nodes:
                continue                    # replaced by the common builder
            dcv = _acp_val(tok[3]) if len(tok) > 3 else 0.0
            # RAIL-REFERENCED bias mapping (not proportional): their level-1
            # vto=+/-0.5 matches sky130 vth magnitudes, so preserving the
            # bias's distance to its nearer rail preserves device overdrive.
            # Proportional 1.8/5 scaling was measured to push NMOS gates
            # subthreshold (the legacy VCM_ratio bug pattern). Clamped [0,1.8];
            # the vsrc tuner knob (+/-0.2 V) retains adjustment room.
            v18 = (min(dcv, 1.8) if dcv <= 2.5
                   else max(0.0, 1.8 - (5.0 - dcv)))
            keep_v.append(f"{name} {net(tok[1])} {net(tok[2])} {v18:.4g}")
        elif k in ("L", "D", "E", "G", "F", "H", "B", "X"):
            return None, f"unrealizable_{k}", None, None
        else:
            return None, f"unknown_card_{k}", None, None
    if not any(b.startswith("x") for b in body):
        return None, "no_mos_device", None, None
    out = next((n for n in sorted(nets) if "vout" in n), None) or \
        next((n for n in sorted(nets) if n.endswith("out") or "out" in n), None)
    if out is None:
        return None, "no_output_node", None, None
    # STRIP THEIR OWN OUTPUT LOAD (2026-09-06 fix, found via zero-pass audit):
    # their exported netlists near-universally (432/434 checked) carry an
    # output-to-ground cap for THEIR OWN dc-sweep/gain check (e.g. "CL Vout 0
    # 100pF"), independent of the spec's actual load target. Left in place it
    # sits in parallel with the harness's own CLOAD_TB, silently altering the
    # load every candidate sees (fairness-parity violation vs every other
    # baseline, which sees only the harness's spec-matched load). Signature is
    # unambiguous and safe to distinguish from a real Miller/compensation cap:
    # one terminal is exactly the output node AND the other is ground -- a
    # functional cap (e.g. "CCC n1 Vout 5pF") never has a grounded terminal.
    body = [ln for ln in body if not (
        ln[:1].upper() == "C" and
        {ln.split()[1], ln.split()[2]} == {out, "0"})]
    return "\n".join(body + keep_v), "ok", out, in_nodes[:2]


def acp_builder(cl_pf, out_node, in_nodes):
    def build(body, vdd_scale=1.0, temp_c=None):
        vdd = 1.8 * vdd_scale
        deck = ["* topo-baseline analogcoder deck",
                f".include {TT_SPICE}",
                ".param mc_mm_switch=0", ".param mc_pr_switch=0",
                f"V1 vdd 0 {vdd}", "V2 vss 0 0"]
        if in_nodes:
            deck.append(f"Vinp {in_nodes[0]} 0 dc {0.5 * vdd:.4g} ac 1")
        if len(in_nodes) > 1:
            deck.append(f"Vinn {in_nodes[1]} 0 dc {0.5 * vdd:.4g}")
        deck += [body, f"CLOAD_TB {out_node} 0 {cl_pf}p"]
        if temp_c is not None:
            deck.append(f".temp {temp_c}")
        deck += [c.format(out=out_node) for c in CONTROL] + [".end"]
        return "\n".join(deck)
    return build


def part_acp_gen(seed: int, model: str = "gpt-5-mini", n_samples: int = TOP_K):
    """One generation BATCH (seed = batch id; LLM output is not seedable --
    batches are independent draws, disclosed). Resumable per spec."""
    from src.evaluation.external_baselines.analogcoderpro_adapter import (
        run_frozen_spec)
    sp = specs29()
    out_path = GEN / f"acp_s{seed}.json"
    data = (json.loads(out_path.read_text(encoding="utf-8"))
            if out_path.exists() else {"batch": seed, "model": model,
                                       "results": []})
    done = {r["spec_index"] for r in data["results"]}
    for i in sorted(sp):
        if i in done:
            continue
        p = sp[i]
        entry = {"context_id": f"spec{i}_b{seed}", "parsed_spec": p}
        t0 = time.time()
        rows = run_frozen_spec(entry, n_samples, model=model)
        nets, calls, toks = [], 0, 0
        for r in rows:
            np_ = getattr(r, "netlist_path", "") or ""
            if np_ and Path(np_).exists():
                nets.append(Path(np_).read_text(encoding="utf-8",
                                                errors="replace"))
            calls += getattr(r, "llm_calls", 0) or 0
            toks += getattr(r, "llm_tokens", 0) or 0
        data["results"].append({"spec_index": i, "netlists": nets,
                                "gen_time_s": round(time.time() - t0, 1),
                                "llm_calls": calls, "llm_tokens": toks})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data), encoding="utf-8")
        print(f"[acp-gen b{seed} spec{i}] {len(nets)} netlists, "
              f"{calls} llm calls, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------ part: evaluate
# SIZING-BOUNDS PARITY (2026-08-31): the original tune_budget compounded
# unbounded multiplicative steps (W and Ibias could drift 10-100x), while
# RAPTOR's frozen action space clamps W to 4-8x, Ibias to 0.25-4x,
# caps to 0.5-2048x, Rz to 0.1-40x -- the brief requires SAME sizing bounds.
# bounded_tune keeps the same dumb random hill-climb but tracks the TOTAL
# multiplier of every device value relative to the method's ORIGINAL netlist
# and clamps it to the AG knob ranges. Bias-voltage sources keep a +/-0.2 V
# absolute window (operating-point enablement, not an AG knob).
BOUNDS = {"w": (0.5, 8.0), "cap": (0.5, 2048.0), "res": (0.1, 40.0),
          "isrc": (0.25, 4.0)}
VSRC_WIN = 0.2


def bounded_tune(build, body, kinds, p, out_node, seed, budget):
    import re as _re
    from src.evaluation.external_baselines.run_stage6_v2_tuner import (
        KNOB_RES, _val, _SUFFIX)
    rng = random.Random(seed)
    kinds = [k for k in kinds if k != "mmul"]     # no m-multiplication
    # locate every tunable value in the ORIGINAL body
    slots = []          # (kind, match_start, match_end, orig_value, suffix)
    for kind in kinds:
        for m in KNOB_RES[kind].finditer(body):
            g = 1 if kind == "w" else 2
            v, suf = _val(m.group(g))
            slots.append([kind, m.start(g), m.end(g), v, suf])
    def rebuild(mults):
        out, pos = [], 0
        for (kind, a, b, v, suf), mu in sorted(zip(slots, mults),
                                               key=lambda x: x[0][1]):
            out.append(body[pos:a])
            nv = v + mu if kind == "vsrc" else v * mu
            out.append(f"{nv:.6g}{suf}")
            pos = b
        out.append(body[pos:])
        return "".join(out)
    best_m = [0.0 if s[0] == "vsrc" else 1.0 for s in slots]
    calls = 0
    best_body, best = body, None
    best_ok, best_fom = False, -99.0
    first = None
    t0 = time.time()
    cand_m = list(best_m)
    while calls < budget:
        cand_body = rebuild(cand_m)
        meas = run_deck(build(cand_body), out_node)
        calls += 1
        ok, fom = score(meas, p)
        if ok and first is None:
            first = calls
        if (ok, fom) > (best_ok, best_fom):
            best_ok, best_fom, best_m, best = ok, fom, list(cand_m), meas
            best_body = cand_body
        cand_m = []
        for s, mu in zip(slots, best_m):
            if rng.random() < 0.5:
                cand_m.append(mu)
            elif s[0] == "vsrc":
                cand_m.append(max(-VSRC_WIN, min(VSRC_WIN,
                                                 mu + rng.uniform(-0.1, 0.1))))
            else:
                lo_, hi_ = BOUNDS[s[0]]
                cand_m.append(max(lo_, min(hi_,
                                           mu * math.exp(rng.gauss(0, 0.3)))))
    # (2026-08-31: a metric-aligned min-current delivery selection was
    # implemented, smoke-tested, and REVERTED by user decision -- protocol B:
    # the margin-selected delivery below stands for all baselines, matching
    # the published results_bounded.jsonl rows. See session log.)
    return {"pass": best_ok, "fom": round(best_fom, 4), "calls": calls,
            "first_pass_call": first, "meas": best, "body": best_body,
            "tune_s": round(time.time() - t0, 1)}


def tune_budget(build, body, kinds, p, out_node, seed, budget):
    rng = random.Random(seed)
    calls, best_body, best = 0, body, None
    best_ok, best_fom, first = False, -99.0, None
    cand = body
    t0 = time.time()
    while calls < budget:
        meas = run_deck(build(cand), out_node)
        calls += 1
        ok, fom = score(meas, p)
        if ok and first is None:
            first = calls
        if (ok, fom) > (best_ok, best_fom):
            best_ok, best_fom, best_body, best = ok, fom, cand, meas
        cand = perturb(best_body, kinds, rng)
    return {"pass": best_ok, "fom": round(best_fom, 4), "calls": calls,
            "first_pass_call": first, "meas": best, "body": best_body,
            "tune_s": round(time.time() - t0, 1)}


def _append(row):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _done():
    if not RESULTS.exists():
        return set()
    return {(json.loads(l)["method"], json.loads(l)["seed"],
             json.loads(l)["spec_index"])
            for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()}


_GEN_CACHE: dict = {}


def _load_gen(method: str, seed: int):
    key = (method, seed)
    if key not in _GEN_CACHE:
        if method == "cktgen":
            g = json.loads((GEN / f"cktgen_s{seed}.json").read_text(
                encoding="utf-8"))["results"]
            _GEN_CACHE[key] = ("per_spec", {r["spec_index"]: r for r in g},
                               None)
        elif method == "analogtobi":
            g = json.loads((GEN / f"tobi_s{seed}.json").read_text(
                encoding="utf-8"))
            _GEN_CACHE[key] = ("pool", g["kept"], g["total_gen_s"])
        elif method == "analogcoder":
            g = json.loads((GEN / f"acp_s{seed}.json").read_text(
                encoding="utf-8"))
            _GEN_CACHE[key] = ("per_spec",
                               {r["spec_index"]: r for r in g["results"]},
                               None)
        else:
            g = json.loads((GEN / f"genie_s{seed}.json").read_text(
                encoding="utf-8"))
            _GEN_CACHE[key] = ("pool", g["kept"], g["total_gen_s"])
    return _GEN_CACHE[key]


def eval_one(method: str, seed: int, i: int) -> dict:
    """Evaluate ONE (method, seed, spec): the parallel unit of work."""
    sp = specs29()
    kind, data, total_gen_s = _load_gen(method, seed)
    per_spec = data if kind == "per_spec" else None
    pool = data if kind == "pool" else None
    if True:
        p = sp[i]
        cl = p["load_capacitance_pf"]
        cands = []
        gen_time = 0.0
        if method == "cktgen":
            r = per_spec.get(i, {})
            gen_time = r.get("generation_time_s", 0.0)
            for k, c in enumerate(r.get("candidates", [])[:TOP_K]):
                net, st = cktgen_realize(c["graph"])
                cands.append({"rank": k + 1, "body": net, "status": st,
                              "builder": (subckt_builder(cl) if net else None),
                              "out": "opout",
                              "kinds": ["w", "cap", "res", "isrc", "mmul"]})
        elif method == "analogcoder":
            r = per_spec.get(i, {})
            gen_time = r.get("gen_time_s", 0.0)
            for k, net_text in enumerate(r.get("netlists", [])[:TOP_K]):
                body, st, out, vins = acp_realize(net_text, cl)
                cands.append({"rank": k + 1, "body": body,
                              "status": st,
                              "builder": (acp_builder(cl, out, vins)
                                          if body else None),
                              "out": out or "vout",
                              "kinds": ["w", "cap", "res", "isrc", "vsrc"]})
        elif method == "analogtobi":
            gen_time = total_gen_s / max(1, len(sp))
            for c in pool[:TOP_K]:
                body = tobi_netlist(c["walk"], cl)
                bo = tobi_builder(c["walk"], cl) if body else (None, "vout1")
                cands.append({"rank": c["rank"], "body": body,
                              "status": "ok" if body else "decode_invalid",
                              "builder": bo[0], "out": bo[1],
                              "kinds": ["w", "cap", "res", "isrc", "vsrc",
                                        "mmul"]})
        else:
            gen_time = total_gen_s / max(1, len(sp))
            for c in pool[:TOP_K]:
                body = genie_netlist(c["walk"], cl)
                cands.append({"rank": c["rank"], "body": body,
                              "status": "ok" if body else "decode_invalid",
                              "builder": (genie_builder(c["walk"], cl)
                                          if body else None),
                              "out": "vout1",
                              "kinds": ["w", "cap", "res", "isrc", "vsrc",
                                        "mmul"]})
        results, spice = [], 0
        t0 = time.time()
        for c in cands:
            if not c["body"]:
                results.append({"rank": c["rank"], "valid": False,
                                "why": c["status"], "pass": False,
                                "fom": None, "calls": 0})
                continue
            r = bounded_tune(c["builder"], c["body"], c["kinds"], p,
                             c["out"], seed * 1000 + i * 10 + c["rank"],
                             PER_CAND)
            spice += r["calls"]
            results.append({"rank": c["rank"], "valid": True, "pass": r["pass"],
                            "fom": r["fom"], "calls": r["calls"],
                            "first_pass_call": r["first_pass_call"],
                            "meas": r["meas"], "_body": r["body"],
                            "_builder": c["builder"], "_out": c["out"]})
        finals = [r for r in results if r.get("valid")]
        final = max(finals, key=lambda r: (r["pass"], r["fom"] or -99),
                    default=None)
        fom_power = None
        if final and final.get("meas") and final["meas"].get("idd_a"):
            from agentic_raptor.electrical.fom import compute_fom
            fom_power = compute_fom(final["meas"].get("ugbw_hz"),
                                    cl * 1e-12,
                                    final["meas"]["idd_a"]).get("fom_value")
        corners = pvt_calls = None
        if final and final["pass"]:
            corners, pvt_calls = 0, 0
            for vs, tc in CORNERS.values():
                m2 = run_deck(final["_builder"](final["_body"], vdd_scale=vs,
                                                temp_c=tc), final["_out"])
                pvt_calls += 1
                o2, _ = score(m2, p)
                corners += int(o2)
        for r in results:
            r.pop("_body", None)
            r.pop("_builder", None)
            r.pop("_out", None)
        row = {"method": method, "seed": seed, "spec_index": i,
               "n_candidates": len(cands),
               "final_pass": bool(final and final["pass"]),
               "pass_at_5": any(r.get("pass") for r in results),
               "final_fom": final["fom"] if final else None,
               "final_fom_power": fom_power,
               "final_idd_a": (final["meas"].get("idd_a")
                               if final and final.get("meas") else None),
               "final_rank": final["rank"] if final else None,
               "optimization_calls": spice, "pvt_calls": pvt_calls or 0,
               "vt_corners_pass": corners,
               "generation_time_s": gen_time,
               "eval_time_s": round(time.time() - t0, 1),
               "candidates": results}
        return row


def part_evaluate(method: str, seed: int, limit: int | None):
    sp = specs29()
    done = _done()
    n_run = 0
    for i in sorted(sp):
        if (method, seed, i) in done:
            continue
        if limit and n_run >= limit:
            break
        n_run += 1
        row = eval_one(method, seed, i)
        _append(row)
        print(f"[{method} s{seed} spec{i}] pass={row['final_pass']} "
              f"p@5={row['pass_at_5']} fom={row['final_fom']} "
              f"calls={row['optimization_calls']} "
              f"corners={row['vt_corners_pass']}", flush=True)


def _job(args):
    return eval_one(*args)


def part_evaluate_all(workers: int):
    """All methods x seeds x specs through a process pool (each job's
    65-call tune is internally sequential; jobs are independent)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    sp = specs29()
    done = _done()
    jobs = [(m, s, i) for m in ("cktgen", "analogtobi")
            for s in (0, 1, 2) for i in sorted(sp)
            if (m, s, i) not in done]
    print(f"{len(jobs)} jobs on {workers} workers -> {RESULTS.name}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            m, s, i = futs[f]
            try:
                row = f.result()
            except Exception as e:
                print(f"[{m} s{s} spec{i}] FAILED: {e}", flush=True)
                continue
            _append(row)
            print(f"[{k}/{len(jobs)}] {m} s{s} spec{i} "
                  f"pass={row['final_pass']} fom={row['final_fom']} "
                  f"fomP={row.get('final_fom_power')}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------- part: summary
def part_summary():
    rows = [json.loads(l) for l in RESULTS.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    lines = ["# Topology-baseline results (working table)", "",
             "| Method | FinalPass | P@5 | Calls | Runtime (s) | FoM (uniform rel.) | PVT (%) | n |",
             "|---|---|---|---|---|---|---|---|"]
    for m, rs in sorted(by.items()):
        n = len(rs)
        fp = sum(r["final_pass"] for r in rs) / n
        p5 = sum(r["pass_at_5"] for r in rs) / n
        calls = sum(r["optimization_calls"] for r in rs) / n
        rt = sum(r["generation_time_s"] + r["eval_time_s"] for r in rs) / n
        foms = [r["final_fom"] for r in rs if r["final_pass"]
                and r["final_fom"] is not None]
        fom = sum(foms) / len(foms) if foms else None
        vt = [r["vt_corners_pass"] for r in rs
              if r["vt_corners_pass"] is not None]
        pvt = 100 * sum(vt) / (4 * len(vt)) if vt else None
        lines.append(f"| {m} | {fp:.3f} | {p5:.3f} | {calls:.1f} | {rt:.1f} | "
                     f"{(f'{fom:.2f}' if fom is not None else 'n/a')} | "
                     f"{(f'{pvt:.1f}' if pvt is not None else 'n/a')} | {n} |")
    out = ART / "working_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["map", "cktgen-gen", "genie-gen",
                                     "tobi-gen", "acp-gen", "evaluate",
                                     "evaluate-all", "summary"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["cktgen", "analoggenie",
                                         "analogtobi", "analogcoder"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--samples", type=int, default=TOP_K)
    a = ap.parse_args()
    if a.part == "map":
        part_map()
    elif a.part == "acp-gen":
        part_acp_gen(a.seed, a.model, a.samples)
    elif a.part == "cktgen-gen":
        part_cktgen_gen(a.seed)
    elif a.part == "genie-gen":
        part_genie_gen(a.seed)
    elif a.part == "tobi-gen":
        part_tobi_gen(a.seed)
    elif a.part == "evaluate":
        part_evaluate(a.method, a.seed, a.limit)
    elif a.part == "evaluate-all":
        part_evaluate_all(a.workers)
    else:
        part_summary()
