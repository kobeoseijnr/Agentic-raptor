"""Task 1-3: diagnose pm_deg=None from RAW AC data already on disk.

No new SPICE needed - parses the sweep's stored acdata, classifies every
0dB-crossing situation, validates the PM pipeline against known-stable and
known-unstable references, and audits the emitted nested netlist.
"""
import json
import numpy as np
from pathlib import Path

from agentic_raptor.electrical.measurements import load_wrdata_complex

NM = Path("artifacts/nested_miller")


def diagnose(acpath: Path) -> dict:
    try:
        f, h = load_wrdata_complex(acpath)
    except Exception as exc:
        return {"pm_status": "invalid_loop_measurement", "error": str(exc)[:80]}
    mag = 20 * np.log10(np.abs(h) + 1e-30)
    ph = np.unwrap(np.angle(h)) * 180 / np.pi
    ph -= ph[0]                       # reference to low-frequency phase
    cross = []
    for i in range(len(f) - 1):
        if (mag[i] - 0) * (mag[i + 1] - 0) < 0:
            t = mag[i] / (mag[i] - mag[i + 1])
            fc = f[i] * (f[i + 1] / f[i]) ** t
            pc = ph[i] + t * (ph[i + 1] - ph[i])
            slope = (mag[i + 1] - mag[i]) / np.log10(f[i + 1] / f[i])
            cross.append({"f_hz": round(float(fc), 1),
                          "phase_deg": round(float(pc), 1),
                          "slope_db_dec": round(float(slope), 1),
                          "pm_est_deg": round(float(180 + pc), 1)})
    status = ("no_unity_crossing" if not cross else
              "multiple_crossings" if len(cross) > 2 else
              "ascending_crossing_only" if all(c["slope_db_dec"] > 0 for c in cross)
              else "measurable")
    if cross and f[-1] < 10 * cross[-1]["f_hz"]:
        status = "out_of_frequency_range"
    return {"pm_status": status, "crossing_count": len(cross),
            "crossings": cross[:6],
            "f_range_hz": [float(f[0]), float(f[-1])],
            "dc_gain_db": round(float(mag[0]), 1),
            "estimated_pm_candidates_deg": [c["pm_est_deg"] for c in cross
                                            if c["slope_db_dec"] < 0][:3],
            "verified_pm_deg": (cross[0]["pm_est_deg"]
                                if len(cross) == 1 and cross[0]["slope_db_dec"] < 0
                                else None)}


# Task 1: three representative nested configs
picks = {"highest_ugbw": "c2_1_s1.0_b1.6", "nominal": "c4_1_s1.0_b1.0",
         "smallest_caps": "c2_0.5_s0.7_b1.0"}
print("=== NESTED-MILLER DIAGNOSIS ===")
for name, tag in picks.items():
    ac = NM / tag / "run" / "acdata.txt"
    d = diagnose(ac) if ac.is_file() else {"pm_status": "file_missing", "tag": tag}
    print(name, json.dumps(d))

# Task 2: pipeline validation on known references
print("=== PIPELINE VALIDATION ===")
refs = {"stable_2stage": Path("artifacts/full_raptor_run/keep/run/acdata.txt"),
        "unstable_3stage_simple":
            Path("artifacts/variant_verification/vv_3m00/run/acdata.txt")}
for name, p in refs.items():
    print(name, json.dumps(diagnose(p)) if p.is_file() else f"missing:{p}")

# Task 3: structural audit of the emitted nested netlist
print("=== NETLIST AUDIT (comp lines) ===")
net = (NM / "c4_1_s1.0_b1.0" / "netlist.sp")
if net.is_file():
    for line in net.read_text().splitlines():
        ll = line.lower()
        if ll.startswith(("ccc", "crz", "rrz", "ccc1", "ccc2")) or "cc1" in ll \
                or "cc2" in ll or "rz1" in ll:
            print(" ", line)
