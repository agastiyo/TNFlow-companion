"""Every matplotlib figure: PIT histograms, posterior marginals, JWST exemplars, attention.

Figures written to disk switch to the non-interactive Agg backend; the interactive ones (used by
``pipeline/inference.py``) return the figure and leave the backend alone.
"""

import numpy as np

from src.data_utils.components import Component
from src.evaluation.posterior import grain_draws, order_by_detection

PIT_COLOR, TRUTH_COLOR, ATTN_COLOR = "#4C72B0", "red", "#C44E52"


def _pyplot(agg=False):
    import matplotlib
    if agg:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path, plt):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path}")


def _axes(axes):
    """matplotlib returns a bare Axes for n=1; always give back something iterable."""
    return axes if hasattr(axes, "__len__") else [axes]


# --------------------------------------------------------------------------- PIT

def _pit_step(ax, pit_vals, title, bins, ylabel=True, xlabel=True):
    """Draw one step-style PIT histogram on an axes (frequency = proportion)."""
    u = np.asarray(pit_vals)
    counts, edges = np.histogram(u, bins=bins, range=(0.0, 1.0))
    freq = counts / len(u) if len(u) else counts.astype(float)
    ax.stairs(freq, edges, color=PIT_COLOR, linewidth=1.5)
    ax.axhline(1.0 / bins, color=TRUTH_COLOR, ls="--", lw=1.0)
    ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel("Frequency")
    if xlabel:
        ax.set_xlabel("PIT value")


def pit_grid(columns, row_titles, path, bins=10):
    """2-row PIT histogram grid, one column per split.

    Args: columns: list of (split label, row-0 PIT values, row-1 PIT values);
          row_titles: (row-0 title, row-1 title), each formatted with ``{label}``.
    """
    plt = _pyplot(agg=True)
    fig, axes = plt.subplots(2, len(columns), figsize=(3.5 * len(columns), 5.5),
                             sharex=True, sharey="row", squeeze=False)
    for col, (label, top, bottom) in enumerate(columns):
        _pit_step(axes[0, col], top, row_titles[0].format(label=label), bins,
                  ylabel=(col == 0), xlabel=False)
        _pit_step(axes[1, col], bottom, row_titles[1].format(label=label), bins,
                  ylabel=(col == 0), xlabel=True)
    _save(fig, path, plt)


# --------------------------------------------------------------------------- one posterior

def hist_marginals(comp, true=None, n=3):
    """comp [N, K] -> composition histograms for the top n components. Returns: fig."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    for ax, c in zip(_axes(axes), order_by_detection(comp, n).tolist()):
        ax.hist(comp[:, c].cpu().numpy(), bins=50, range=(0.0, 1.0), density=True, alpha=0.6)
        if true is not None:
            ax.axvline(true[c], color="r", lw=2, label=f"true = {true[c]:.3f}")
            ax.legend()
        ax.set_title(f"{Component.decode(c)} marginal")
        ax.set_xlabel("proportion"); ax.set_ylabel("density")
    fig.tight_layout()
    return fig


def hist_grains(comp, grain, n=3):
    """comp/grain [N, K] -> grain histograms for the top n components. Returns: fig."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    for ax, c in zip(_axes(axes), order_by_detection(comp, n).tolist()):
        g = grain_draws(grain, c)
        if g.numel():
            ax.hist(g.cpu().numpy(), bins=40)
            ax.set_xscale("log")
        ax.set_title(f"{Component.decode(c)} grain")
        ax.set_xlabel("grain size (µm)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- spectra

def spectrum(sample, name):
    """Plot a spectrum. Returns: fig."""
    plt = _pyplot()
    fig, ax = plt.subplots()
    ax.plot(sample.lams, sample.values, lw=1)
    ax.set_title(name)
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("arbitrary reflectance")
    return fig


def attention(sample, lam, weights, name):
    """The spectrum with its per-wavelength query-pooling attention weight below it. Returns: fig."""
    plt = _pyplot()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(sample.lams, sample.values, lw=1, color=PIT_COLOR)
    ax1.set_ylabel("reflectance")
    ax1.set_title(f"{name}")

    ax2.plot(lam, weights, lw=1, color=ATTN_COLOR)
    ax2.fill_between(lam, weights, alpha=0.3, color=ATTN_COLOR)
    ax2.set_xlabel("wavelength (µm)")
    ax2.set_ylabel("attention weight")
    ax2.set_title("Query-pooling attention (share of the context vector from each wavelength)")
    fig.tight_layout()
    return fig


JWST_BANDS = [(1.5, "H$_2$O"), (2.0, "H$_2$O"), (2.27, "CH$_3$OH"),
              (3.0, "H$_2$O"), (4.27, "CO$_2$"), (4.67, "CO")]


def normalize_at(values, lams, lam0, half_window):
    """Reflectance / its mean within ±half_window µm of lam0 (unchanged if the window is empty)."""
    w = np.abs(lams - lam0) < half_window
    return values / values[w].mean() if w.any() else values


def jwst_exemplars(entries, path, norm_lam=0.9, norm_win=0.05):
    """Three-panel figure: one exemplar spectrum per DiSCo group, normalized at ``norm_lam``.

    Args: entries: list of (designation, group, SpectrumSample).
    """
    plt = _pyplot(agg=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharex=True)
    for ax, (desig, group, sample) in zip(axes, entries):
        lam, R = sample.lams, normalize_at(sample.values, sample.lams, norm_lam, norm_win)
        ax.plot(lam, R, color=PIT_COLOR, lw=0.8)
        for x, lbl in JWST_BANDS:
            ax.axvline(x, color="0.75", ls=":", lw=0.7, zorder=0)
            ax.text(x, ax.get_ylim()[1], lbl, fontsize=5.5, rotation=90,
                    va="top", ha="right", color="0.45")
        ax.set_title(f"{group} — {desig}", fontsize=9)
        ax.set_xlabel("Wavelength (µm)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(f"Reflectance (norm. at {norm_lam} µm)", fontsize=8)
    _save(fig, path, plt)
