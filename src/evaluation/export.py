"""
Write the numbered tables the thesis needs, in Markdown, with their sources.

Every table carries the cuadro or module it comes from, so the reader can trace
a number back to the SSN Anuario, the CPI series, or the assumption that
produced it. For a triangle that is simulated rather than observed, that trail
is what makes the exercise auditable — the alternative is a table the reader has
no way to check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _md(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        else:
            out[col] = out[col].astype(str).replace("nan", "")
    header = "| " + " | ".join(str(c) for c in out.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in out.astype(str).values]
    return "\n".join([header, rule, *rows])


def _section(number: str, title: str, source: str, body: str) -> str:
    return f"\n### Tabla {number}. {title}\n\n*Fuente: {source}.*\n\n{body}\n"


def write_thesis_tables(
    out_path: Path,
    *,
    triangle_pivot: pd.DataFrame,
    real_pivot: pd.DataFrame,
    ldfs_nominal: np.ndarray,
    ldfs_real: np.ndarray,
    calibration: pd.DataFrame,
    cpi_sources: pd.DataFrame,
    mack: pd.DataFrame,
    metrics_nominal: pd.DataFrame,
    metrics_real: pd.DataFrame,
    ultimates: pd.DataFrame,
    anchor_sweep: pd.DataFrame,
    inflation_sweep: pd.DataFrame,
    valuation_date: str,
    base_year: int,
) -> None:
    """Write every table to a single Markdown file ready to paste into the thesis."""
    parts: list[str] = [
        "# Tablas para la tesis\n",
        f"Valuación al {valuation_date}. Importes en millones de ARS salvo indicación.\n",
        "\nEl triángulo es sintético: el Anuario de la SSN publica exposición, "
        "siniestros y frecuencia, pero no montos ni desarrollo por año de "
        "ocurrencia, y los datos individuales son confidenciales. Lo que sí es "
        "verificable es la calibración, que es lo que documentan las Tablas 5.1 "
        "y 5.2.\n",
    ]

    cal = calibration.copy()
    cal["vehiculos_expuestos"] = cal["vehiculos_expuestos"] / 1e6
    cal["siniestros_mercado"] = cal["siniestros_mercado"] / 1e6
    parts.append(
        _section(
            "5.1",
            "Calibración del triángulo: parámetros por año de ocurrencia",
            "SSN, *Anuario Estadístico — Automotores, Coberturas de Casco* (Anexo II), "
            "Cuadros 1, 2 y 4. Vehículos expuestos = TOTAL − categoría A (RC sin casco). "
            "Los años previos a 2015 se extrapolan de la tendencia y están marcados",
            _md(
                cal.reset_index().rename(
                    columns={
                        "anio": "Año",
                        "vehiculos_expuestos": "Vehículos expuestos (M)",
                        "frecuencia": "Frecuencia",
                        "indice_precios": "Índice de precios",
                        "siniestros_mercado": "Siniestros mercado (M)",
                        "fuente": "Fuente",
                    }
                ),
                "{:,.4f}",
            ),
        )
    )

    parts.append(
        _section(
            "5.2",
            "Serie de inflación empleada",
            "Variación diciembre-diciembre. INDEC IPC Nacional desde 2017; para "
            "2007–2015 se empalman índices provinciales y alternativos, dado que "
            "el IPC del INDEC estuvo intervenido en ese período",
            _md(
                cpi_sources.reset_index().rename(
                    columns={
                        "index": "Año",
                        "variacion_pct": "Variación anual (%)",
                        "fuente": "Fuente",
                    }
                ),
                "{:,.1f}",
            ),
        )
    )

    parts.append(
        _section(
            "5.3",
            "Triángulo de pagos acumulados, pesos nominales (M ARS)",
            "elaboración propia sobre la calibración de la Tabla 5.1",
            _md((triangle_pivot / 1e6).reset_index(), "{:,.1f}"),
        )
    )

    parts.append(
        _section(
            "5.4",
            f"Triángulo de pagos acumulados, pesos constantes de {base_year} (M ARS)",
            "elaboración propia. Cada pago incremental se deflacta por el índice "
            "de su año calendario de pago y luego se re-acumula",
            _md((real_pivot / 1e6).reset_index(), "{:,.3f}"),
        )
    )

    factors = pd.DataFrame(
        {
            "Factor": [f"f{j + 1}" for j in range(len(ldfs_nominal))],
            "Nominal": ldfs_nominal,
            "Moneda constante": ldfs_real,
        }
    )
    parts.append(
        _section(
            "5.5",
            "Factores de desarrollo ponderados",
            "elaboración propia. La diferencia entre ambas columnas es el "
            "componente inflacionario que el triángulo nominal incorpora al "
            "patrón de desarrollo",
            _md(factors, "{:,.3f}"),
        )
    )

    mack_out = mack.copy()
    for col in ("latest", "ultimate", "ibnr", "se"):
        if col in mack_out.columns:
            mack_out[col] = mack_out[col] / 1e6
    parts.append(
        _section(
            "6.1",
            "Mack (1993): reserva y error de predicción por año de ocurrencia",
            "elaboración propia. El MSEP cubre error de proceso y de estimación "
            "bajo los supuestos del Chain Ladder; no cubre error de modelo",
            _md(
                mack_out.drop(columns=["mse"], errors="ignore").reset_index().rename(
                    columns={
                        "index": "Año",
                        "latest": "Pagado",
                        "ultimate": "Ultimate",
                        "ibnr": "IBNR",
                        "se": "Error estándar",
                        "cv": "CV",
                    }
                ),
                "{:,.2f}",
            ),
        )
    )

    parts.append(
        _section(
            "6.2",
            "Ultimates proyectados por modelo y año de ocurrencia (M ARS nominales)",
            "elaboración propia",
            _md(
                ultimates.set_index("accident_year")
                .filter(regex="^(?!.*_IBNR).*$")
                .div(1e6)
                .reset_index()
                .rename(columns={"accident_year": "Año"}),
                "{:,.1f}",
            ),
        )
    )

    def _metrics_md(df: pd.DataFrame) -> str:
        out = df.copy()
        cols = {
            "model": "Modelo",
            "mape": "MAPE (%)",
            "r2": "R²",
            "total_ibnr": "IBNR total",
            "pct_error_ibnr": "Error s/ IBNR (%)",
        }
        keep = [c for c in cols if c in out.columns]
        out = out[keep].rename(columns=cols)
        if "IBNR total" in out.columns:
            out["IBNR total"] = out["IBNR total"] / 1e6
        return _md(out, "{:,.2f}")

    parts.append(
        _section(
            "6.3",
            "Métricas por modelo — ajuste sobre el triángulo nominal",
            "elaboración propia. El error se mide contra el ultimate realmente "
            "observado en la simulación, no contra la última diagonal: esta "
            "última es el pagado a la fecha, de modo que compararse con ella "
            "premia al modelo que menos reserva constituye",
            _metrics_md(metrics_nominal),
        )
    )

    parts.append(
        _section(
            "6.4",
            f"Métricas por modelo — ajuste en pesos constantes de {base_year}, "
            "reserva re-inflada a nominal",
            "elaboración propia",
            _metrics_md(metrics_real),
        )
    )

    if not metrics_nominal.empty and "pct_error_ibnr" in metrics_nominal.columns:
        comp = metrics_nominal[["model", "pct_error_ibnr"]].merge(
            metrics_real[["model", "pct_error_ibnr"]], on="model", suffixes=("_nom", "_real")
        )
        comp.columns = ["Modelo", "Error nominal (%)", "Error moneda constante (%)"]
        comp["Mejora (p.p.)"] = comp["Error nominal (%)"].abs() - comp[
            "Error moneda constante (%)"
        ].abs()
        parts.append(
            _section(
                "6.5",
                "Efecto de trabajar en moneda constante sobre el error de reserva",
                "elaboración propia",
                _md(comp, "{:,.1f}"),
            )
        )

    if not anchor_sweep.empty:
        parts.append(
            _section(
                "6.6",
                "Sensibilidad del híbrido al peso del ancla Chain Ladder",
                "elaboración propia",
                _md(
                    anchor_sweep.assign(ibnr_total=anchor_sweep.ibnr_total / 1e6).rename(
                        columns={
                            "anchor_weight": "Peso del ancla",
                            "ibnr_total": "IBNR total",
                            "error_pct": "Error (%)",
                        }
                    ),
                    "{:,.2f}",
                ),
            )
        )

    if not inflation_sweep.empty:
        parts.append(
            _section(
                "6.7",
                "Sensibilidad de la reserva al supuesto de inflación futura",
                "elaboración propia, Chain Ladder ajustado en moneda constante. "
                "El patrón de desarrollo es idéntico en todas las filas: lo único "
                "que cambia es la conversión de la reserva proyectada a nominal",
                _md(
                    inflation_sweep.assign(
                        ibnr_total=inflation_sweep.ibnr_total / 1e6
                    ).rename(
                        columns={
                            "inflacion_futura_pct": "Inflación futura anual (%)",
                            "ibnr_total": "IBNR total",
                            "error_pct": "Error (%)",
                        }
                    ),
                    "{:,.1f}",
                ),
            )
        )

    out_path.write_text("\n".join(parts), encoding="utf-8")
