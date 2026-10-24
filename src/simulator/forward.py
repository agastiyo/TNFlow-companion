"""The Shkuratov forward model: composition + grain sizes + porosity -> reflectance spectrum.

Wraps :class:`cana.composition.IntimateMixture` (Shkuratov 1999 intimate mixing) with optical
constants from SDOC (Small Database of Optical Constants). Used by the corpus generator
(``generate.py``), the single-spectrum plot (``plot_one.py``) and ``pipeline/inference.py``.

    ocs = load_optical_constants(["H2O", "CO2"])
    R = simulate(["H2O", "CO2"], [0.7, 0.3], [20.0, 50.0], porosity=0.1, ocs=ocs)   # [len(WAVELENGTHS)]

NOTE ON DEPENDENCIES
--------------------
CANA declares ``numba`` as a dependency but uses it only as a no-op ``@jit`` decorator on a
pure-numpy static method. numba/llvmlite does not build on this machine, so cana and sdoc are
installed with ``--no-deps`` and a tiny no-op ``numba`` shim (below) is injected before importing
cana. The Shkuratov math is unaffected.
"""

from __future__ import annotations

import sys
import types

# --- no-op numba shim (must be installed before importing cana) ----------------------------
if "numba" not in sys.modules:  # pragma: no cover - environment shim
    _numba = types.ModuleType("numba")

    def _jit(*args, **kwargs):
        # Support both @jit and @jit(nopython=True) decorator forms.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    _numba.jit = _jit
    _numba.njit = _jit
    sys.modules["numba"] = _numba
# -------------------------------------------------------------------------------------------

import numpy as np
import sdoc
from cana.composition import IntimateMixture, read_constant

# Wavelength grid (um): 0.35 -> 5.0 at 0.01 step. SDOC constants end at 4.99 um; values beyond
# that are clamped by the interpolation (edge-held), which is fine at the very red end.
WAVELENGTHS = np.round(np.arange(0.35, 5.0 + 1e-9, 0.01), 2)

# Component library: name -> SDOC optical-constant id. Every name is a Component enum member.
# Where a material has several SDOC entries (temperature/phase variants) one representative is
# chosen; swap the id here to use a different variant.
COMPONENT_LIBRARY: dict[str, str] = {
    "N2": "nit_0",       # Nitrogen, 36.5 K
    "CO2": "cod_1",      # Carbon dioxide, Hansen et al. 1997
    "H2O": "wat_2",      # Water, amorphous 40 K (Mastrapa 2008/2009)
    "CH4": "met_0",      # Methane, crystalline 39 K (Grundy 2002)
    "CH3OH": "mol_0",    # Methanol, 100 K (Hudgins 1993 / Brown 1995)
    "AC": "car_0",       # Amorphous carbon (Rouleau & Martin 1991)
    "TTH": "tit_0",      # Titan tholin (Khare et al. 1984)
    "TrT": "trt_0",      # Triton tholin (Khare)
    "Oliv": "oli_0",     # Olivine Mg_2y Fe_2-2y SiO4, y=0.5 (Dorschner 1995)
    "C2H6": "eta_0",     # Ethane, crystalline 30 K (Hudson 2014)
    "C2H4": "eth_0",     # Ethylene, amorphous 30 K (Hudson 2014)
    "NH3": "amo_3",      # Ammonia, amorphous 40 K (Roser 2021)
    "CO": "com_0",       # Carbon monoxide, 10 K (PubChem)
    "Pyroxene": "pyr_1",  # Pyroxene Mg_x Fe_1-x SiO3, x=0.50 (Dorschner 1995)
}


def load_optical_constants(names=None, wavelengths=WAVELENGTHS) -> dict[str, object]:
    """Read optical constants from SDOC once, rebased onto ``wavelengths``.

    Args:    names: iterable of COMPONENT_LIBRARY names (default: all 14).
    Returns: {name: OpticalConstant}, so repeated mixture builds don't re-read SDOC.
    """
    sdb = sdoc.SDOC(mode="r")
    ocs: dict[str, object] = {}
    for name in (COMPONENT_LIBRARY if names is None else names):
        _label, data = sdb.get_constant(COMPONENT_LIBRARY[name])
        ocs[name] = read_constant(data, label=name).rebase(baseaxis=wavelengths)
    sdb.close()
    return ocs


def simulate(names, proportions, grains_um, porosity, ocs=None, wavelengths=WAVELENGTHS) -> np.ndarray:
    """One Shkuratov intimate-mixture spectrum.

    Args:    names: component names; proportions: same length, summing to ~1; grains_um: grain size
             per component (µm); porosity: float in [0, 1); ocs: preloaded optical constants (from
             ``load_optical_constants``; loaded here if None).
    Returns: reflectance ndarray [len(wavelengths)]. NaN where the Shkuratov albedo has no real
             solution (some highly absorbing / porous mixtures); callers decide whether to redraw.
    """
    names = list(names)
    ocs = ocs or load_optical_constants(names, wavelengths)
    mix = IntimateMixture([ocs[n] for n in names], grainsizes=list(grains_um),
                          proportions=list(proportions), porosity=porosity)
    with np.errstate(invalid="ignore"):
        spec, _albedo = mix.make(wavelengths=wavelengths)
    return np.asarray(spec.r, dtype=float)
