"""MOS model library management.

Default: a built-in generic LEVEL=1 library (technology-neutral, converges
easily) clearly labelled as such. Real technologies are supplied via
``spice.model_library_path`` (a file .include'd verbatim) plus model names in
the file; the library version participates in the cache key via
``technology_label``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_raptor.utils.exceptions import SimulationError

#: Built-in generic 1um-class LEVEL=1 models. NOT a real PDK — smoke/benchmark only.
GENERIC_LIBRARY = """\
* Built-in generic LEVEL=1 models (technology-neutral; benchmark use only)
.model nmos_generic NMOS (LEVEL=1 VTO=0.5 KP=120u LAMBDA=0.05 GAMMA=0.4 PHI=0.7 CGSO=0.2n CGDO=0.2n)
.model pmos_generic PMOS (LEVEL=1 VTO=-0.5 KP=40u LAMBDA=0.06 GAMMA=0.5 PHI=0.7 CGSO=0.2n CGDO=0.2n)
"""


@dataclass(frozen=True)
class ModelLibrary:
    label: str                    # participates in the SPICE cache key
    nmos_model: str
    pmos_model: str
    include_path: str | None      # external file to .include, or None → inline generic
    inline_cards: str | None      # inline model cards when no include file

    def netlist_section(self) -> str:
        if self.include_path:
            return f".include {self.include_path}"
        return self.inline_cards or ""


def load_model_library(
    model_library_path: str | None,
    technology_label: str,
    nmos_model: str | None = None,
    pmos_model: str | None = None,
) -> ModelLibrary:
    """Resolve the model library from config.

    External libraries must also name the models to use (config attributes or
    defaults ``nmos``/``pmos``); the built-in generic library names its own.
    """
    if model_library_path:
        path = Path(model_library_path)
        if not path.is_file():
            raise SimulationError(f"model library not found: {path}")
        return ModelLibrary(
            label=technology_label,
            nmos_model=nmos_model or "nmos",
            pmos_model=pmos_model or "pmos",
            include_path=str(path),
            inline_cards=None,
        )
    return ModelLibrary(
        label=technology_label or "generic_1u_level1",
        nmos_model="nmos_generic",
        pmos_model="pmos_generic",
        include_path=None,
        inline_cards=GENERIC_LIBRARY,
    )
