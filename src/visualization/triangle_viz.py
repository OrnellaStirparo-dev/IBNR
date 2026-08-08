"""
Development pattern of the triangle, nominal and in constant currency.

A heatmap of the cumulative amounts adds nothing the triangle table does not
already show, and on twenty accident years of Argentine pesos the cell labels
collide into an unreadable block. What the table cannot show is the shape of the
development pattern and how it drifts across accident years, which is the
observation the inflation treatment rests on.

Accident year is an ordered variable, so the cohorts are coloured on a single
sequential ramp from light to dark rather than with categorical hues.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from src.data.inflation import deflate_triangle
from src.traditional.chain_ladder_manual import cumulative_pivot_from_long
from src.visualization.comparison import GRID, INK, INK_MUTED, INK_SOFT, SURFACE, _style


def _paid_share(pivot: pd.DataFrame) -> pd.DataFrame:
    """Cumulative share of the latest observed amount paid by each period."""
    out = pivot.div(pivot.ffill(axis=1).iloc[:, -1], axis=0)
    return out.where(pivot.notna())


def plot_triangle(
    triangle_df: pd.DataFrame,
    base_year: int = 2005,
    save_path: str | None = None,
) -> None:
    """
    Two panels: share of the run-off paid by development period, one line per
    accident year, in nominal pesos and in constant currency.

    Only the cohorts with a full ten periods of development are drawn, so every
    line covers the same span and the comparison between them is meaningful.
    """
    nominal = cumulative_pivot_from_long(triangle_df)
    real = cumulative_pivot_from_long(deflate_triangle(triangle_df, base_year, "calendar"))

    complete = [ay for ay in nominal.index if nominal.loc[ay].notna().all()]
    if len(complete) < 2:
        complete = list(nominal.index)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    # Truncated ramp: the palest steps of Blues fall below 3:1 contrast on the
    # chart surface and the earliest cohorts would be invisible.
    ramp = colors.LinearSegmentedColormap.from_list(
        "cohortes", cm.get_cmap("Blues")(np.linspace(0.35, 0.95, 256))
    )
    norm = colors.Normalize(vmin=min(complete), vmax=max(complete))

    for ax, pivot, title in (
        (axes[0], nominal, "Pesos nominales"),
        (axes[1], real, f"Pesos constantes de {base_year}"),
    ):
        _style(ax)
        share = _paid_share(pivot)
        for ay in complete:
            row = share.loc[ay].dropna()
            ax.plot(
                [int(c) for c in row.index],
                100 * row.values,
                color=ramp(norm(ay)),
                linewidth=1.8,
            )
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
        ax.set_xlabel("Período de desarrollo", fontsize=9, color=INK_SOFT)
        ax.set_xticks(range(1, int(pivot.columns.max()) + 1))
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("% del run-off pagado", fontsize=9, color=INK_SOFT)

    bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=ramp), ax=axes, pad=0.015, fraction=0.03)
    bar.set_label("Año de ocurrencia", fontsize=8.5, color=INK_SOFT)
    bar.set_ticks([min(complete), max(complete)])
    bar.ax.tick_params(labelsize=8, colors=INK_SOFT)
    bar.outline.set_edgecolor(GRID)

    fig.suptitle(
        "Patrón de desarrollo por año de ocurrencia",
        fontsize=12, color=INK, fontweight="bold", x=0.01, y=1.06, ha="left",
    )
    fig.text(
        0.01, 1.00,
        "En pesos nominales el patrón se desplaza entre cohortes; deflactado, las curvas se superponen.",
        fontsize=8.5, color=INK_MUTED,
    )
    fig.patch.set_facecolor(SURFACE)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
