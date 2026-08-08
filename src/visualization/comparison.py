"""
Charts for the reserving comparison.

Each figure answers one question. The previous versions plotted every model
against every accident year as grouped bars — eight models over twenty years is
160 bars in one panel, which encodes the data faithfully and communicates
nothing. Where more than a few series have to be shown together, they are
faceted into small multiples instead of stacked into the same axes.

Colour follows the job, not the taste: a single categorical pair (nominal vs.
constant currency) for the headline comparison, one hue per facet elsewhere, and
a neutral reference line for the realised value. The pair is validated for
colour-vision deficiency separation on the chart surface.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette — validated categorical pair on the light chart surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8985"
SERIES_NOMINAL = "#2a78d6"
SERIES_REAL = "#eb6834"
REFERENCE = "#0b0b0b"
GRID = "#e3e2de"

_MILLION = 1e6


def _style(ax: plt.Axes) -> None:
    """Recessive chrome: no top/right spine, soft grid behind the marks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def _finish(fig: plt.Figure, save_path: str | None) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_error_by_model(
    nominal: pd.DataFrame,
    real: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Headline chart: error on the total reserve, nominal vs. constant currency.

    Horizontal bars because the category labels are model names, which read
    badly rotated. Sorted by the constant-currency error so the ranking is the
    reading order, with a zero line as the target.
    """
    if nominal.empty or "pct_error_ibnr" not in nominal.columns:
        return

    merged = nominal[["model", "pct_error_ibnr"]].merge(
        real[["model", "pct_error_ibnr"]], on="model", suffixes=("_nom", "_real")
    )
    merged = merged.reindex(
        merged.pct_error_ibnr_real.abs().sort_values(ascending=False).index
    )

    y = np.arange(len(merged))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(merged) + 1.6))
    _style(ax)

    ax.barh(y + height / 2 + 0.01, merged.pct_error_ibnr_nom, height,
            color=SERIES_NOMINAL, label="Nominal")
    ax.barh(y - height / 2 - 0.01, merged.pct_error_ibnr_real, height,
            color=SERIES_REAL, label="Moneda constante")

    for yi, (nom, rl) in enumerate(
        zip(merged.pct_error_ibnr_nom, merged.pct_error_ibnr_real)
    ):
        for value, offset in ((nom, height / 2 + 0.01), (rl, -height / 2 - 0.01)):
            ax.text(
                value + (6 if value >= 0 else -6),
                yi + offset,
                f"{value:+.0f}%",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
                color=INK_SOFT,
            )

    ax.axvline(0, color=REFERENCE, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(merged.model, fontsize=9, color=INK)
    ax.set_xlabel("Error sobre la reserva total (%)", fontsize=9, color=INK_SOFT)
    ax.set_title(
        "Error de la reserva por modelo",
        fontsize=12, color=INK, fontweight="bold", loc="left", pad=26,
    )
    ax.text(
        0, 1.015, "Cero es la reserva efectivamente observada. Positivo sobre-reserva.",
        transform=ax.transAxes, fontsize=8.5, color=INK_MUTED, va="bottom",
    )
    ax.margins(x=0.18)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_SOFT)
    _finish(fig, save_path)


def plot_reserve_by_accident_year(
    ultimates: dict[str, pd.Series],
    latest: pd.Series,
    true_ultimates: pd.Series | None = None,
    save_path: str | None = None,
    n_years: int = 10,
) -> None:
    """
    Small multiples: reserve by accident year, one panel per model.

    Faceting rather than overlaying — eight lines over ten points in shared axes
    is unreadable, and the comparison that matters is each model against the
    realised reserve, not against the other models. Only the immature accident
    years are shown; the fully developed ones carry no reserve and would flatten
    the scale.
    """
    if not ultimates:
        return

    years = sorted(latest.dropna().index.astype(int))[-n_years:]
    names = list(ultimates)
    cols = 4
    rows = int(np.ceil(len(names) / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(3.3 * cols, 2.5 * rows), sharex=True, sharey=True, squeeze=False
    )

    real = (
        (true_ultimates.reindex(years) - latest.reindex(years)) / _MILLION
        if true_ultimates is not None
        else None
    )
    x = np.arange(len(years))

    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        _style(ax)
        reserve = (ultimates[name].reindex(years) - latest.reindex(years)) / _MILLION

        if real is not None:
            ax.plot(x, real.values, color=REFERENCE, linewidth=1.6,
                    label="Reserva real", zorder=3)
        ax.plot(x, reserve.values, color=SERIES_NOMINAL, linewidth=2,
                marker="o", markersize=4, label=name, zorder=4)

        ax.set_title(name, fontsize=9.5, color=INK, loc="left")
        ax.margins(y=0.12)
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(y) for y in years[::2]], rotation=45, fontsize=7.5)
        if idx % cols == 0:
            ax.set_ylabel("Reserva (M ARS)", fontsize=8, color=INK_SOFT)

    for idx in range(len(names), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    handles = [
        plt.Line2D([], [], color=REFERENCE, linewidth=1.6, label="Reserva real"),
        plt.Line2D([], [], color=SERIES_NOMINAL, linewidth=2, marker="o",
                   markersize=4, label="Modelo"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2,
               loc="lower center", labelcolor=INK_SOFT)
    fig.suptitle(
        "Reserva por año de ocurrencia frente a la observada",
        fontsize=12, color=INK, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.patch.set_facecolor(SURFACE)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_uncertainty_bands(
    uq_results: dict[str, tuple[pd.Series, pd.Series]],
    latest: pd.Series,
    true_ultimates: pd.Series | None = None,
    save_path: str | None = None,
    n_years: int = 8,
) -> None:
    """
    Small multiples of the 90% predictive interval, one panel per model that
    quantifies uncertainty.

    Restricted to the immature accident years: on the developed ones the
    interval collapses to a point and only compresses the scale.
    """
    if not uq_results:
        return

    years = sorted(latest.dropna().index.astype(int))[-n_years:]
    names = list(uq_results)
    cols = min(len(names), 2)
    rows = int(np.ceil(len(names) / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(5.2 * cols, 3.1 * rows), squeeze=False, sharex=True
    )
    x = np.arange(len(years))

    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        _style(ax)
        mean, std = uq_results[name]
        mu = mean.reindex(years).values / _MILLION
        sd = std.reindex(years).values / _MILLION
        paid = latest.reindex(years).values / _MILLION

        # The predictive distribution of an ultimate is right-skewed, so a
        # symmetric mean ± z·sd band understates the upper tail and can fall
        # below the amount already paid — an impossible ultimate. The lower
        # bound is floored at the paid-to-date, which is the constraint the
        # quantity actually has.
        lower = np.maximum(mu - 1.645 * sd, paid)
        ax.fill_between(x, lower, mu + 1.645 * sd,
                        color=SERIES_NOMINAL, alpha=0.18, linewidth=0)
        ax.plot(x, paid, color=INK_MUTED, linewidth=1, linestyle=":")
        ax.plot(x, mu, color=SERIES_NOMINAL, linewidth=2, marker="o", markersize=4)
        if true_ultimates is not None:
            ax.plot(x, true_ultimates.reindex(years).values / _MILLION,
                    color=REFERENCE, linewidth=1.6)

        ax.set_title(f"{name} — IC 90%", fontsize=9.5, color=INK, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=7.5)
        ax.set_ylabel("Ultimate (M ARS)", fontsize=8, color=INK_SOFT)

    for idx in range(len(names), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    handles = [
        plt.Line2D([], [], color=REFERENCE, linewidth=1.6, label="Ultimate real"),
        plt.Line2D([], [], color=SERIES_NOMINAL, linewidth=2, label="Media del modelo"),
        plt.Line2D([], [], color=INK_MUTED, linewidth=1, linestyle=":", label="Pagado a la fecha"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=3,
               loc="lower center", labelcolor=INK_SOFT)
    fig.suptitle(
        "Distribuciones predictivas por modelo",
        fontsize=12, color=INK, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.patch.set_facecolor(SURFACE)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_sensitivity(
    anchor_sweep: pd.DataFrame,
    inflation_sweep: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Two panels: the hybrid's anchor weight and the future-inflation assumption."""
    panels = [
        (anchor_sweep, "anchor_weight", "Peso del ancla Chain Ladder",
         "Sensibilidad del híbrido al peso del ancla"),
        (inflation_sweep, "inflacion_futura_pct", "Inflación futura anual supuesta (%)",
         "Sensibilidad de la reserva al supuesto de inflación"),
    ]
    panels = [p for p in panels if not p[0].empty]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 3.4), squeeze=False)

    for idx, (data, xcol, xlabel, title) in enumerate(panels):
        ax = axes[0][idx]
        _style(ax)
        ax.plot(data[xcol], data["error_pct"], color=SERIES_NOMINAL,
                linewidth=2, marker="o", markersize=4)
        ax.axhline(0, color=REFERENCE, linewidth=1.2)

        best = data.loc[data.error_pct.abs().idxmin()]
        ax.annotate(
            f"{best[xcol]:g}",
            xy=(best[xcol], best.error_pct),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_SOFT,
        )
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK_SOFT)
        ax.set_ylabel("Error sobre la reserva (%)", fontsize=8.5, color=INK_SOFT)
        ax.set_title(title, fontsize=10, color=INK, loc="left")

    _finish(fig, save_path)
