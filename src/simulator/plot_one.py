"""Generate a single Shkuratov intimate-mixture spectrum and plot it.

Edit the MIXTURE block below to change components, proportions, grain sizes and
porosity, then run::

    python3.10 -m src.simulator.plot_one
"""

import matplotlib.pyplot as plt

from src.simulator.forward import WAVELENGTHS, simulate

# --- MIXTURE: edit this block for different mixtures ----------------------------------------
# name -> (proportion, grain_size_um). Names from forward.COMPONENT_LIBRARY; proportions sum to ~1.
MIXTURE = {
    "CH3OH": (0.50, 15.3),
    "H2O" : (0.24, 5.0),
    "CO2" : (0.10, 15.3),
    "TrT" : (0.08, 20.8),
    "AC" : (0.05, 28.2),
    "CO" : (0.02, 74.2)
}
POROSITY = 0.1
# -------------------------------------------------------------------------------------------


def main() -> None:
    names = list(MIXTURE.keys())
    proportions = [MIXTURE[n][0] for n in names]
    grains = [MIXTURE[n][1] for n in names]
    reflectance = simulate(names, proportions, grains, POROSITY)

    label = ", ".join(f"{n} {p:.2f}" for n, p in zip(names, proportions))
    plt.figure(figsize=(8, 5))
    plt.plot(WAVELENGTHS, reflectance, lw=1.2)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Reflectance")
    plt.title(f"Shkuratov mixture: {label} (porosity {POROSITY})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
