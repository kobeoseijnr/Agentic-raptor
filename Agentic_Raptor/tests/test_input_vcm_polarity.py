"""The input common mode must follow the input-pair polarity.

The testbench was inherited from legacy AnalogGym, which is entirely
PMOS-input and biases the inputs at 0.25*1.8 = 0.45 V. mapping/ builds
NMOS-input amplifiers, for which 0.45 V is BELOW the sky130 nfet threshold
(0.564 V): measured branch current 121 nA and gm 3.0 uS at 0.25, versus
9.5 uA and 191 uS at 0.5. These tests pin the polarity split so the legacy
value can never be re-applied to an NMOS pair, and so legacy circuits can
never be silently re-biased.
"""
from pathlib import Path

import pytest

from agentic_raptor.electrical import (_LEGACY_AMP, input_pair_polarity,
                                       input_vcm_ratio)

NMOS_PAIR = """\
.subckt e2 gnda vdda vinn vinp vout
xM1 nmir vinn ntail gnda sky130_fd_pr__nfet_01v8 l=0.5 w=10.0 m=1
xM2 n1 vinp ntail gnda sky130_fd_pr__nfet_01v8 l=0.5 w=10.0 m=1
xM3 nmir nmir vdda vdda sky130_fd_pr__pfet_01v8 l=0.5 w=20.0 m=1
.ends
"""

PMOS_PAIR = """\
.subckt leg gnda vdda vinn vinp vout
xm9 net063 VINP net31 net31 sky130_fd_pr__pfet_01v8 l='L_gm1_PMOS'
xm8 DM_2 VINN net31 net31 sky130_fd_pr__pfet_01v8 l='L_gm1_PMOS'
xm5 net31 nb gnda gnda sky130_fd_pr__nfet_01v8 l=1.0
.ends
"""


def test_nmos_input_pair_detected():
    assert input_pair_polarity(NMOS_PAIR) == "nmos"


def test_pmos_input_pair_detected_case_insensitively():
    # legacy netlists spell the nets VINP/VINN in upper case
    assert input_pair_polarity(PMOS_PAIR) == "pmos"


def test_polarity_ignores_non_input_devices():
    """A pfet load must not be mistaken for the input pair.

    NMOS_PAIR's xM3 is a pfet; only the gate-on-vinp/vinn device counts.
    """
    assert input_pair_polarity(NMOS_PAIR) == "nmos"
    assert input_pair_polarity(PMOS_PAIR) == "pmos"


def test_nmos_gets_common_mode_above_threshold():
    """0.5*1.8 = 0.9 V clears vth(0.564) + vov + vdsat_tail."""
    assert input_vcm_ratio("nmos") == 0.5
    assert 1.8 * input_vcm_ratio("nmos") > 0.564 + 0.15 + 0.15


def test_pmos_keeps_the_legacy_value():
    assert input_vcm_ratio("pmos") == 0.25


def test_unknown_polarity_falls_back_to_legacy():
    """An unparsable netlist must not silently change anyone's bias."""
    assert input_pair_polarity("* no devices here") == "unknown"
    assert input_vcm_ratio("unknown") == 0.25


@pytest.mark.skipif(not (_LEGACY_AMP / "spice_netlist").is_dir(),
                    reason="legacy AnalogGym netlists not present")
def test_every_legacy_netlist_is_unchanged():
    """Legacy behaviour must be bit-identical: all of them are PMOS-input."""
    checked = 0
    for f in sorted((_LEGACY_AMP / "spice_netlist").rglob("*")):
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if ".subckt" not in text.lower():
            continue
        assert input_pair_polarity(text) == "pmos", f.name
        assert input_vcm_ratio(input_pair_polarity(text)) == 0.25, f.name
        checked += 1
    assert checked >= 16, f"expected the full legacy set, saw {checked}"


def test_testbench_emits_the_polarity_aware_value(tmp_path: Path):
    from agentic_raptor.electrical import build_testbench

    (tmp_path / "netlist.sp").write_text(NMOS_PAIR, encoding="utf-8")

    class _Entry:
        path = tmp_path
        topology_id = "t1"
        metadata: dict = {}

    tb = build_testbench(_Entry(), {"subckt_name": "e2"})
    assert ".PARAM VCM_ratio = 0.5" in tb
    assert "input pair = nmos" in tb
