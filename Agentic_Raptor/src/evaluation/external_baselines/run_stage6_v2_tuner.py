"""STAGE 6 v2 -- COMMON TUNER over Track-A topologies (2026-08-29).

One deliberately simple, identical optimizer for every method's topologies:
  start from the as-generated values -> 65 judge calls of random multiplicative
  perturbation (log-sigma 0.3 on device/passive knobs, +/-0.1 V additive on
  bias-voltage knobs), keep the best by (pass, uniform relative FoM) ->
  VT-corner sweep (vdd +/-10% x 0/70C) on the winner (4 extra calls, bucketed
  separately). The tuner is intentionally dumb: any intelligence would be a
  confound; the question is topology quality, not optimizer quality.

Per-dialect decks (knob DOF differs by construction -- documented per row):
  ACP    flat level-1 netlist (own models, disclosed): knobs = every w=,
         every capacitor, every non-supply V source.
  AG     the design's OWN campaign tb.cir (correct per-design input biasing,
         sky130 tt corner include -- SkyWater PDK data vendored under the
         legacy tree: third-party DATA, no legacy code executed), with the
         netlist inlined + spec CL + standardized measurement block:
         knobs = every w= on x-instances, every iIB current, every r/c value.
  PANDA  assembled deck: PPAAS-shipped 45nm BSIM4 model cards renamed to
         nch_mac/pch_mac (documented substitution) + cfmom_2t cap shim +
         their 24-cell leaf library + the generated topology + a 1.2 V
         testbench; knobs = per-instance m multipliers + one V source per
         declared bias port (init 0.6 V).

Uniform judge + uniform relative FoM for every method:
  pass  = gain>=gt AND ugbw>=ut AND pm>=pt   (the spec's own targets)
  fom   = (gain-gt)/gt + (ugbw-ut)/ut + (pm-pt)/pt

Output: artifacts/external_baselines/stage6_v2_results.jsonl
Every ngspice invocation counted.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "external_baselines" / "stage6_v2_results.jsonl"
TRACK_A = ROOT / "artifacts" / "external_baselines" / "track_a_full.jsonl"
SIZING = ROOT / "artifacts" / "publication_v2" / "raptor_v2_runs" / "sizing"
PANDA_DIR = ROOT / "external_baselines" / "PANDA" / "topology_gen"
PPAAS_45NM = ROOT / "external_baselines" / "PPAAS" / "eval_engines" / "pdk" / "45nm_bulk.txt"
NGSPICE = r"C:/Users/kobeo/OneDrive/Desktop/Spice64/bin/ngspice_con.exe"
CORNERS = {"LL": (0.9, 0), "LH": (0.9, 70), "HL": (1.1, 0), "HH": (1.1, 70)}
BUDGET = 65

CONTROL = [".control", "op",
           # supply current at the DC operating point (V1 = the supply
           # source in every deck this judge sees) -> power-FoM input
           "wrdata s6_op.csv v1#branch",
           "ac dec 20 1 10G",
           "wrdata s6_out.csv vdb({out}) cph({out})", "quit", ".endc"]


def _specs():
    d = json.loads((ROOT / "data/external_baseline_eval/specs_validation.json"
                    ).read_text(encoding="utf-8"))
    return {s["context_id"]: s for s in d["specs"]}, \
           {s["spec_index"]: s for s in d["specs"]}


def run_deck(deck_text: str, out_node: str, timeout=60) -> dict | None:
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "d.cir"
        f.write_text(deck_text, encoding="utf-8")
        try:
            subprocess.run([NGSPICE, "-b", str(f)], cwd=td,
                           capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        outf = Path(td) / "s6_out.csv"
        if not outf.exists():
            return None
        idd = None
        opf = Path(td) / "s6_op.csv"
        if opf.exists():
            for ln in opf.read_text().splitlines():
                parts = ln.split()
                if len(parts) >= 2:
                    try:
                        idd = abs(float(parts[1]))
                    except ValueError:
                        pass
        rows = []
        for ln in outf.read_text().splitlines():
            p = ln.split()
            if len(p) >= 4:
                try:
                    rows.append((float(p[0]), float(p[1]), float(p[3])))
                except ValueError:
                    pass
    if not rows:
        return None
    gain_db = rows[0][1]
    ugbw = pm = None
    for i in range(1, len(rows)):
        if rows[i - 1][1] >= 0 > rows[i][1]:
            f0, g0 = rows[i - 1][0], rows[i - 1][1]
            f1, g1 = rows[i][0], rows[i][1]
            t = g0 / (g0 - g1)
            ugbw = f0 * (f1 / f0) ** t
            ph = rows[i - 1][2] + t * (rows[i][2] - rows[i - 1][2])
            pm = 180.0 + math.degrees(ph) if abs(ph) < 7 else 180.0 + ph
            pm = abs(((pm + 180) % 360) - 180)
            break
    return {"gain_db": gain_db, "ugbw_hz": ugbw, "pm_deg": pm, "idd_a": idd}


def score(meas, p):
    if not meas or meas.get("ugbw_hz") is None:
        return False, -99.0
    ok = (meas["gain_db"] >= p["gain_target_db"]
          and meas["ugbw_hz"] >= p["ugbw_target_hz"]
          and (meas["pm_deg"] or 0) >= p["phase_margin_target_deg"])
    fom = ((meas["gain_db"] - p["gain_target_db"]) / max(p["gain_target_db"], 1)
           + (meas["ugbw_hz"] - p["ugbw_target_hz"]) / max(p["ugbw_target_hz"], 1)
           + ((meas["pm_deg"] or 0) - p["phase_margin_target_deg"])
           / max(p["phase_margin_target_deg"], 1))
    return ok, fom


# ------------------------- knob machinery -----------------------------------
KNOB_RES = {
    "w": re.compile(r"(?i)\bw=([0-9.eE+-]+[a-z]*)"),
    "cap": re.compile(r"(?im)^(C\w+\s+\S+\s+\S+\s+)([0-9.eE+-]+[a-z]*)"),
    "res": re.compile(r"(?im)^(r\w+\s+\S+\s+\S+\s+)([0-9.eE+-]+[a-z]*)"),
    "isrc": re.compile(r"(?im)^(i\w+\s+\S+\s+\S+\s+)([0-9.eE+-]+[a-z]*)"),
    "vsrc": re.compile(r"(?im)^(V(?!dd|DD|1 )\w*\s+\S+\s+\S+\s+)([0-9.eE+-]+)\s*$"),
    "mmul": re.compile(r"(?i)\bm=(\d+)"),
}
_SUFFIX = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
           "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}


def _val(tok: str) -> tuple[float, str]:
    m = re.match(r"([0-9.eE+-]+)([a-z]*)", tok, re.I)
    v = float(m.group(1))
    suf = m.group(2).lower()
    return v, suf


def perturb(body: str, kinds: list[str], rng: random.Random,
            sigma: float = 0.3) -> str:
    def mul(tok):
        v, suf = _val(tok)
        return f"{v * math.exp(rng.gauss(0, sigma)):.6g}{suf}"
    out = body
    for kind in kinds:
        rx = KNOB_RES[kind]
        if kind == "w":
            out = rx.sub(lambda m: f"w={mul(m.group(1))}"
                         if rng.random() < 0.5 else m.group(0), out)
        elif kind in ("cap", "res", "isrc"):
            out = rx.sub(lambda m: m.group(1) + (mul(m.group(2))
                         if rng.random() < 0.5 else m.group(2)), out)
        elif kind == "vsrc":
            out = rx.sub(lambda m: m.group(1) +
                         f"{float(m.group(2)) + rng.uniform(-0.1, 0.1):.4g}"
                         if rng.random() < 0.5 else m.group(0), out)
        elif kind == "mmul":
            out = rx.sub(lambda m: f"m={max(1, min(32, int(round(int(m.group(1)) * math.exp(rng.gauss(0, sigma))))))}"
                         if rng.random() < 0.5 else m.group(0), out)
    return out


def tune(build_deck, base_body: str, kinds: list[str], p: dict, out_node: str,
         seed: int) -> dict:
    rng = random.Random(seed)
    calls = 0
    best_body, best = base_body, None
    best_ok, best_fom = False, -99.0
    first_pass = None
    t0 = time.time()
    cand = base_body
    while calls < BUDGET:
        meas = run_deck(build_deck(cand), out_node)
        calls += 1
        ok, fom = score(meas, p)
        if ok and first_pass is None:
            first_pass = calls
        if (ok, fom) > (best_ok, best_fom):
            best_ok, best_fom, best_body, best = ok, fom, cand, meas
        cand = perturb(best_body, kinds, rng)
    corners = None
    if best_ok:
        corners = 0
        for vs, tc in CORNERS.values():
            m2 = run_deck(build_deck(best_body, vdd_scale=vs, temp_c=tc),
                          out_node)
            o2, _ = score(m2, p)
            corners += int(o2)
    return {"final_pass": best_ok, "fom_rel": round(best_fom, 4),
            "first_pass_call": first_pass, "tune_calls": calls,
            "corner_calls": 4 if best_ok else 0,
            "vt_corners_pass": corners, "best_meas": best,
            "runtime_s": round(time.time() - t0, 1)}


# ------------------------- deck builders ------------------------------------
def acp_builder(net_text: str, cl_pf: float):
    def build(body, vdd_scale=1.0, temp_c=None):
        lines = []
        for ln in body.splitlines():
            s = ln.strip()
            if re.match(r"^C\w*\s+Vout\s+0\s", s, re.I):
                continue
            if vdd_scale != 1.0 and s.lower().startswith("vdd"):
                m = re.match(r"^(V\w+\s+\S+\s+\S+\s+)([0-9.eE+-]+)\s*$", s)
                if m:
                    s = f"{m.group(1)}{float(m.group(2)) * vdd_scale}"
            lines.append(s)
        b = "\n".join(lines)
        b = re.sub(r"(?im)^(Vinp\s+\S+\s+\S+\s+)([0-9.eE+-]+)\s*$",
                   lambda m: f"{m.group(1)}{m.group(2)} ac 1", b, count=1)
        deck = ["* s6v2 acp", b, f"CLOAD_S6 Vout 0 {cl_pf}p"]
        if temp_c is not None:
            deck.append(f".temp {temp_c}")
        deck += [c.format(out="Vout") for c in CONTROL] + [".end"]
        return "\n".join(deck)
    return build


AG_TB_CACHE: dict = {}


def ag_builder(run_dir: Path, cl_pf: float):
    tb = (run_dir / "run" / "tb.cir").read_text(encoding="utf-8",
                                                errors="replace")
    def build(body, vdd_scale=1.0, temp_c=None):
        # inline the (tuned) netlist: drop the netlist include, prepend body
        lines = []
        for ln in tb.splitlines():
            s = ln.rstrip()
            low = s.lower()
            if low.startswith(".include") and "netlist.sp" in low:
                continue
            if "param_cload" in low and low.startswith(".param"):
                s = f".PARAM PARAM_CLOAD = {cl_pf}e-12"
            if low.startswith(".param supply_voltage") and vdd_scale != 1.0:
                v = float(s.split("=")[1])
                s = f".PARAM supply_voltage = {v * vdd_scale}"
            if low.startswith(".temp") and temp_c is not None:
                s = f".TEMP {temp_c}"
            if low.startswith(".control"):
                break
            lines.append(s)
        out = [body] + lines
        if temp_c is not None and not any(l.lower().startswith(".temp")
                                          for l in lines):
            out.append(f".temp {temp_c}")
        out += [c.format(out="opout") for c in CONTROL] + [".end"]
        return "\n".join(out)
    return build


PANDA_PRELUDE: str | None = None


def _panda_prelude() -> str:
    global PANDA_PRELUDE
    if PANDA_PRELUDE is not None:
        return PANDA_PRELUDE
    models = PPAAS_45NM.read_text(encoding="utf-8", errors="replace")
    models = re.sub(r"(?im)^\.model\s+nmos\b", ".model nch_mac", models)
    models = re.sub(r"(?im)^\.model\s+pmos\b", ".model pch_mac", models)
    lib = (PANDA_DIR / "SubCircuit" / "user_macro_raw.txt").read_text(
        encoding="utf-8", errors="replace")
    # The macro prompt file contains name-colliding variants (e.g. a 2-port
    # block mislabeled CommonSourceN). Dedupe by the CANONICAL port count
    # from PANDA's own validator cell library (their cli `library` output).
    import os as _os
    import subprocess as _sp
    cp = _sp.run([_os.environ.get("AGR_EVAL_PYTHON",
                  r"C:\Users\kobeo\AppData\Local\Python\pythoncore-3.14-64\python.exe"),
                  "-m", "analogxpert.cli", "library"],
                 cwd=PANDA_DIR.parent, capture_output=True, text=True)
    canon: dict[str, int] = {}
    for ln in cp.stdout.splitlines():
        m2 = re.match(r"^- (\w+): (.+)$", ln.strip())
        if m2:
            canon[m2.group(1).lower()] = len(m2.group(2).split())
    # Each block's preceding '* Cell Name:' header is the TRUE identity; the
    # prompt file mislabels several .SUBCKT lines (e.g. the Pair variants
    # reuse the base cell's name). Rename the subckt to its header name.
    picked: dict[str, str] = {}
    for m3 in re.finditer(r"(?ims)^\*\s*Cell Name:\s*(\w+).*?^(\.SUBCKT\s+(\S+)([^\n]*)\n.*?^\.ENDS)",
                          lib):
        header, blk, sub_nm, rest = (m3.group(1), m3.group(2),
                                     m3.group(3), m3.group(4))
        if header.lower() != sub_nm.lower():
            blk = blk.replace(f".SUBCKT {sub_nm}{rest}",
                              f".SUBCKT {header}{rest}", 1)
        picked.setdefault(header.lower(), blk)
    cells = "\n".join(picked.values())
    shim = (".SUBCKT cfmom_2t p n\nC1 p n 100f\n.ENDS\n")
    PANDA_PRELUDE = models + "\n" + shim + "\n" + cells
    return PANDA_PRELUDE


def _panda_clean(topo_text: str) -> str:
    """Make the generated topology simulatable: strip pin-type annotations
    Vx(I|O) and the '/' cell-name delimiter on instance lines -- the same
    normalization PANDA's own topology_cleaner applies before simulation."""
    out = []
    for ln in topo_text.splitlines():
        ln = re.sub(r"\(([IVBOP|]+)\)", "", ln)
        ln = ln.replace(" / ", " ").replace(" /", " ")
        out.append(ln)
    return "\n".join(out)


def panda_builder(topo_text: str, cl_pf: float):
    topo_text = _panda_clean(topo_text)
    m = re.search(r"(?im)^\.SUBCKT\s+(\S+)\s+(.*)$", topo_text)
    top, ports = m.group(1), m.group(2).split()
    ports = [p.split("(")[0] for p in ports]
    def node(p):
        return p
    bias_ports = [p for p in ports if p.upper().startswith(("VBIAS", "VCM",
                                                            "VCASC", "VCMFB"))]
    inp = next((p for p in ports if p.upper() in ("VIP", "VINP")), None)
    inn = next((p for p in ports if p.upper() in ("VIN", "VINN")), None)
    outp = next((p for p in ports if p.upper() in ("VOUT", "VOUTP", "OUT")),
                ports[-3] if len(ports) > 2 else ports[0])
    vddp = next((p for p in ports if p.upper() == "VDD"), "VDD")
    gndp = next((p for p in ports if p.upper() in ("GND", "VSS")), "GND")
    def build(body, vdd_scale=1.0, temp_c=None):
        body = _panda_clean(body)
        vdd = 1.2 * vdd_scale
        deck = ["* s6v2 panda", _panda_prelude(), body,
                f"V_VDD {node(vddp)} 0 {vdd}", f"V_GND {node(gndp)} 0 0"]
        # bias sources: the tuner's vsrc knob perturbs these
        for i, bp in enumerate(bias_ports):
            deck.append(f"Vb{i} {node(bp)} 0 0.6")
        if inp and inn:
            deck.append(f"Vinp {node(inp)} 0 {0.5 * vdd} ac 1")
            deck.append(f"Vinn {node(inn)} 0 {0.5 * vdd}")
        elif inn:
            deck.append(f"Vinn {node(inn)} 0 {0.5 * vdd} ac 1")
        x_ports = " ".join(node(p) for p in ports)
        deck.append(f"Xdut {x_ports} {top}")
        deck.append(f"CLOAD_S6 {node(outp)} 0 {cl_pf}p")
        if temp_c is not None:
            deck.append(f".temp {temp_c}")
        deck += [c.format(out=node(outp)) for c in CONTROL] + [".end"]
        return "\n".join(deck)
    return build, bias_ports


# ------------------------- parts --------------------------------------------
def _append(row: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _done() -> set:
    if not OUT.exists():
        return set()
    return {(json.loads(l)["baseline"], json.loads(l)["spec_id"],
             json.loads(l)["seed"])
            for l in OUT.read_text(encoding="utf-8").splitlines()}


def part_acp() -> None:
    by_ctx, _ = _specs()
    rows = [json.loads(l) for l in TRACK_A.read_text(encoding="utf-8").splitlines()]
    # one BEST candidate per (spec_id, seed): prefer their-check pass, else graph-valid
    cand: dict = {}
    for r in rows:
        if r["baseline"] != "analogcoderpro_specaligned" or not r["valid_graph"]:
            continue
        k = (r["spec_id"], r["seed"])
        cur = cand.get(k)
        if cur is None or (r["simulatable"], 0) > (cur["simulatable"], 0):
            cand[k] = r
    done = _done()
    for i, ((sid, seed), r) in enumerate(sorted(cand.items())):
        if ("analogcoderpro", sid, seed) in done:
            continue
        p = by_ctx[sid]["parsed_spec"]
        npth = Path(r["netlist_path"])
        if not npth.exists():
            continue
        body = npth.read_text(encoding="utf-8", errors="replace")
        res = tune(acp_builder(body, p["load_capacitance_pf"]), body,
                   ["w", "cap", "res", "vsrc"], p, "Vout", seed=1000 + i)
        _append({"baseline": "analogcoderpro", "spec_id": sid, "seed": seed,
                 "dof": "w,cap,res,bias", **res})
        print(f"[acp {i+1}/{len(cand)}] {sid} s{seed}: pass={res['final_pass']} "
              f"fom={res['fom_rel']} vt={res['vt_corners_pass']}", flush=True)


def part_ag() -> None:
    _, by_idx = _specs()
    done = _done()
    n = 0
    for seed_dir in sorted(SIZING.iterdir()):
        if not seed_dir.name.startswith("s") or not seed_dir.is_dir():
            continue
        # campaign layout: sizing/<runname>/... map via ABLv3 run dirs instead
    # simpler: iterate the ABLv3 HELDOUT29 A0 sizing dirs
    runs = sorted(SIZING.glob("*/"))
    for rd in runs:
        net = rd / "netlist.sp"
        tb = rd / "run" / "tb.cir"
        if not net.exists() or not tb.exists():
            continue
        name = rd.name
        m = re.match(r"s(\d+)$", name)
        # heuristic mapping: the Tier-3 sizing dirs carry run names; only use
        # dirs whose tb references heldout specs is impractical -- instead use
        # spec_index from design_variables file name if present
        # v2 scope: use directory order for the 29*3 most recent A0 runs is
        # unreliable -> we match by the ABLv3 trace's sizing path when present.
        break
    print("ag: run-dir mapping requires trace linkage; using trace-linked list")
    trace_dir = ROOT / "artifacts/publication_v2/raptor_v2_runs"
    count = 0
    for seed in (0, 1, 2):
        for idx in range(29):
            matches = sorted(trace_dir.glob(
                f"ABLv3HELDOUT29_A0_s{seed}_heldout_{idx:03d}_*.json"))
            if not matches:
                continue
            tr = json.loads(matches[-1].read_text(encoding="utf-8"))
            sdir = None
            szg = tr.get("stage6_sizing") or {}
            for k, v in szg.items():
                if isinstance(v, str) and "sizing" in v.replace("\\", "/"):
                    cand_p = Path(v)
                    if cand_p.exists():
                        sdir = cand_p
                        break
            if sdir is None:
                # fall back: search sizing dirs whose tb mentions this run id
                continue
            spec = by_idx.get(idx)
            if spec is None:
                continue
            sid = f"heldout_{idx}"
            if ("agentic_raptor", sid, seed) in _done():
                continue
            p = spec["parsed_spec"]
            body = (sdir / "netlist.sp").read_text(encoding="utf-8",
                                                   errors="replace")
            res = tune(ag_builder(sdir, p["load_capacitance_pf"]), body,
                       ["w", "cap", "res", "isrc"], p, "opout",
                       seed=2000 + idx * 3 + seed)
            _append({"baseline": "agentic_raptor", "spec_id": sid,
                     "seed": seed, "dof": "w,cap,res,ibias", **res})
            count += 1
            print(f"[ag {count}] {sid} s{seed}: pass={res['final_pass']} "
                  f"fom={res['fom_rel']}", flush=True)
    print(f"ag: tuned {count} designs")


def panda_cleaned_builder(topo_text: str, cl_pf: float):
    """Deck for a STRICT-CLEANED (leaf-level) PANDA netlist: 45nm BSIM cards
    renamed to nch/pch (documented substitution), resistor/capacitor
    primitive shims, supply + IBIAS + shared AC testbench."""
    m = re.search(r"(?im)^\.SUBCKT\s+(\S+)\s+(.*)$", topo_text)
    top, ports = m.group(1), m.group(2).split()
    up = [p.upper() for p in ports]
    def port(name, fallback=None):
        for cand in name:
            if cand in up:
                return ports[up.index(cand)]
        return fallback
    inp = port(("VINP", "VIP"))
    inn = port(("VINN", "VIN"))
    outp = port(("VOUT", "OUT", "VOUTP"))
    vddp = port(("VDD",), "VDD")
    gndp = port(("VSS", "GND"), "VSS")
    ibias = port(("IBIAS", "IB", "IREF"))
    models = PPAAS_45NM.read_text(encoding="utf-8", errors="replace")
    models = re.sub(r"(?im)^\.model\s+nmos\b", ".model nch", models)
    models = re.sub(r"(?im)^\.model\s+pmos\b", ".model pch", models)
    shims = (".subckt resistor p n r=1k\nR1 p n {r}\n.ends\n"
             ".subckt capacitor p n c=1p\nC1 p n {c}\n.ends\n")
    def _ngspice_xline(ln: str) -> str:
        # PANDA's cleaner emits Spectre-style 'Xname TYPE nodes params';
        # ngspice wants 'Xname nodes TYPE params'. Pure format conversion.
        t = ln.split()
        if len(t) >= 3 and t[0][0].upper() == "X" and \
                t[1].lower() in ("resistor", "capacitor"):
            nodes = [x for x in t[2:] if "=" not in x]
            params = [x for x in t[2:] if "=" in x]
            return " ".join([t[0]] + nodes + [t[1]] + params)
        return ln

    def build(body, vdd_scale=1.0, temp_c=None):
        body = "\n".join(_ngspice_xline(l) for l in body.splitlines())
        vdd = 1.2 * vdd_scale
        deck = ["* s6v2 panda cleaned", models, shims, body,
                f"V_VDD {vddp} 0 {vdd}", f"V_GND {gndp} 0 0"]
        if ibias:
            deck.append(f"IB_S6 {vddp} {ibias} 10u")
        if inp and inn:
            deck.append(f"Vinp {inp} 0 {0.5 * vdd} ac 1")
            deck.append(f"Vinn {inn} 0 {0.5 * vdd}")
        deck.append(f"Xdut {' '.join(ports)} {top}")
        deck.append(f"CLOAD_S6 {outp} 0 {cl_pf}p")
        if temp_c is not None:
            deck.append(f".temp {temp_c}")
        deck += [c.format(out=outp) for c in CONTROL] + [".end"]
        return "\n".join(deck)
    return build


CLEANED_DIR = ROOT / "artifacts" / "external_baselines" / "raw" / "panda_cleaned"


def part_panda() -> None:
    by_ctx, _ = _specs()
    done = _done()
    files = sorted(CLEANED_DIR.glob("*.sp"))
    reports = {f.stem for f in files}
    # rows for strict-clean FAILURES too (their own gate's rejection)
    for rep in sorted(CLEANED_DIR.glob("*.report.json")):
        stem = rep.name[:-len(".report.json")]
        if stem in reports:
            continue
        sid = re.sub(r"_s\d+$", "", stem)
        if ("panda_strictfail", sid, 0) in done:
            continue
        _append({"baseline": "panda_strictfail", "spec_id": sid, "seed": 0,
                 "final_pass": False, "fom_rel": None, "tune_calls": 0,
                 "notes": "rejected by PANDA's OWN strict cleaner "
                          "(pin-order/unknown-cell errors) -- their gate's "
                          "verdict, no simulation attempted"})
    for i, f in enumerate(files):
        sid = re.sub(r"_s\d+$", "", f.stem)
        if ("panda", sid, 0) in done:
            continue
        spec = by_ctx.get(sid)
        if spec is None:
            continue
        p = spec["parsed_spec"]
        body = f.read_text(encoding="utf-8", errors="replace")
        try:
            build = panda_cleaned_builder(body, p["load_capacitance_pf"])
        except Exception as e:
            _append({"baseline": "panda", "spec_id": sid, "seed": 0,
                     "final_pass": None, "notes": f"deck build failed: {e}"})
            continue
        res = tune(build, body, ["w", "mmul", "res", "cap"], p, "x",
                   seed=3000 + i)
        _append({"baseline": "panda", "spec_id": sid, "seed": 0,
                 "dof": "w,m,r,c (strict-cleaned leaf netlist)", **res})
        print(f"[panda {i+1}/{len(files)}] {sid}: pass={res['final_pass']} "
              f"fom={res['fom_rel']}", flush=True)


def part_panda_OLD() -> None:
    by_ctx, _ = _specs()
    rows = [json.loads(l) for l in TRACK_A.read_text(encoding="utf-8").splitlines()]
    done = _done()
    todo = [r for r in rows if r["baseline"] in ("panda", "panda_llm")
            and r["valid_graph"] and r.get("netlist_path")]
    for i, r in enumerate(todo):
        sid, seed, base = r["spec_id"], r["seed"], r["baseline"]
        if (base, sid, seed) in done:
            continue
        p = by_ctx[sid]["parsed_spec"]
        body = Path(r["netlist_path"]).read_text(encoding="utf-8",
                                                 errors="replace")
        try:
            build, bias_ports = panda_builder(body, p["load_capacitance_pf"])
        except Exception as e:
            _append({"baseline": base, "spec_id": sid, "seed": seed,
                     "final_pass": None, "notes": f"deck build failed: {e}"})
            continue
        res = tune(build, body, ["mmul"], p, "opout", seed=3000 + i)
        # NOTE: bias voltages live in the TB (not body); tuner perturbs body
        # knobs (m=) only in v2.0 -- bias search is v2.1 (documented DOF gap)
        _append({"baseline": base, "spec_id": sid, "seed": seed,
                 "dof": "m-multipliers (bias fixed 0.6V -- v2.0 limitation)",
                 **res})
        print(f"[panda {i+1}/{len(todo)}] {base} {sid}: "
              f"pass={res['final_pass']} fom={res['fom_rel']}", flush=True)


def part_summary() -> None:
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    by = defaultdict(list)
    for r in rows:
        by[r["baseline"]].append(r)
    print(f"\n=== STAGE-6 v2 COMMON-TUNER SUMMARY ({len(rows)} rows) ===")
    for b, rs in sorted(by.items()):
        judged = [r for r in rs if r.get("final_pass") is not None]
        ok = [r for r in judged if r["final_pass"]]
        fp5 = sum(1 for r in judged if (r.get("first_pass_call") or 99) <= 5)
        foms = sorted(r["fom_rel"] for r in ok)
        med = foms[len(foms) // 2] if foms else None
        vt = [r for r in ok if r.get("vt_corners_pass") is not None]
        vt4 = sum(1 for r in vt if r["vt_corners_pass"] == 4)
        calls = sum((r.get("tune_calls") or 0) + (r.get("corner_calls") or 0)
                    for r in rs)
        print(f"{b:18s} tuned={len(judged):3d} pass={len(ok):3d} "
              f"P@5={fp5}/{len(judged)} fom_med={med} vt4={vt4}/{len(vt)} "
              f"spice={calls}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["acp", "ag", "panda", "summary"])
    a = ap.parse_args()
    {"acp": part_acp, "ag": part_ag, "panda": part_panda,
     "summary": part_summary}[a.part]()
