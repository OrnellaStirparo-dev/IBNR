"""Tablas resumen de la validación empírica (lee resultados/empirico/*.csv)."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

OUT = Path("resultados/empirico")
C2 = "MLP agrupado, mismo ramo (C2)"
C3 = "MLP agrupado, multirramo (C3)"
ORDER = [
    "Chain Ladder", "Bornhuetter-Ferguson", "Bayesian Chain Ladder", "MLP", "GRU",
    "CANN", "Híbrido de credibilidad", "CL factores de mercado (C2)",
    "MLP agrupado, mismo ramo (C2)", "MLP agrupado, multirramo (C3)",
]


def summary(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["absE"] = r["E"].abs()
    best = r.loc[r.groupby(["lob", "group_code"])["absE"].idxmin(), "model"].value_counts(normalize=True) * 100
    agg = r.groupby("model").agg(
        n=("E", "size"),
        E_mediana=("E", "median"),
        absE_mediana=("absE", "median"),
        absE_media=("absE", "mean"),
        WMAPE_mediana=("WMAPE", "median"),
    )
    tot = r.groupby("model")[["R_hat", "R_true"]].sum()
    agg["E_cartera_total"] = (tot["R_hat"] / tot["R_true"] - 1) * 100
    agg["pct_mejor"] = best.reindex(agg.index).fillna(0)
    return agg.reindex([m for m in ORDER if m in agg.index]).round(1)


def abs_e(r: pd.DataFrame) -> pd.DataFrame:
    """|E| por cartera (filas) y modelo (columnas)."""
    return r.pivot_table(index=["lob", "group_code"], columns="model", values="E").abs()


def paired(r: pd.DataFrame, a: str, b: str) -> dict:
    """
    Comparación pareada de *b* contra *a* sobre las carteras con ambos ajustes.

    Devuelve el porcentaje de carteras en que *b* tiene menor |E| y dos
    contrastes de la hipótesis de que ninguno de los dos domina:

    ``p_signos``
        Test de signos: binomial sobre la cantidad de carteras en que *b*
        mejora a *a*, con n = total de carteras. Usa sólo el orden, de modo
        que una mejora de un punto pesa igual que una de cien.
    ``p_wilcoxon``
        Wilcoxon de rangos signados sobre los |E| pareados, que además
        pondera por el tamaño de la diferencia.
    """
    d = abs_e(r)[[a, b]].dropna()
    n = len(d)
    if n == 0:
        return {"n": 0, "pct": np.nan, "p_signos": np.nan, "p_wilcoxon": np.nan}
    wins = int((d[b] < d[a]).sum())
    return {
        "n": n,
        "pct": 100.0 * wins / n,
        "p_signos": float(binomtest(wins, n, 0.5).pvalue),
        "p_wilcoxon": float(wilcoxon(d[a], d[b]).pvalue),
    }


def c3_vs_c2(r: pd.DataFrame) -> pd.DataFrame:
    """
    C3 (multirramo) contra C2 (mismo ramo), desagregado por ramo.

    Separa el efecto de agrupar compañías del de agrupar ramos: C2 y C3
    comparten arquitectura y features salvo el indicador de ramo, así que la
    diferencia entre ambos es lo que aporta sumar carteras de otros ramos.
    """
    rows = {}
    for lob, g in [("TODOS", r)] + list(r.groupby("lob")):
        d = abs_e(g)[[C2, C3]].dropna()
        if d.empty:
            continue
        better = d[C3] < d[C2]
        rows[lob] = {
            "n": len(d),
            "absE_mediana_C2": d[C2].median(),
            "absE_mediana_C3": d[C3].median(),
            "C3_mejora_pct": 100.0 * better.mean(),
            "p_signos": float(binomtest(int(better.sum()), len(d), 0.5).pvalue),
        }
    return pd.DataFrame(rows).T


def main() -> None:
    res = pd.read_csv(OUT / "resultados_por_compania.csv", dtype={"group_code": str})
    diag = pd.read_csv(OUT / "homogeneidad.csv", dtype={"group_code": str})
    cov = res[res.model == "_mack_cobertura"]
    r = res[res.model != "_mack_cobertura"].merge(
        diag[["lob", "group_code", "homogeneous"]], on=["lob", "group_code"])

    with pd.ExcelWriter(OUT / "resumen_empirico.xlsx") as xw:
        for lob, g in [("TODOS", r)] + list(r.groupby("lob")):
            s = summary(g)
            s.to_excel(xw, sheet_name=lob[:31])
            print(f"\n=== {lob}  ({g.group_code.nunique() if lob != 'TODOS' else len(g[['lob','group_code']].drop_duplicates())} carteras)")
            print(s.to_string())
        for flag, g in r.groupby("homogeneous"):
            s = summary(g)
            s.to_excel(xw, sheet_name=f"homogenea_{flag}")
            print(f"\n=== Homogénea = {flag} ({len(g[['lob','group_code']].drop_duplicates())} carteras)")
            print(s.to_string())
        pp = r[r.lob == "private_passenger_auto"]
        for flag, g in pp.groupby("homogeneous"):
            print(f"\n=== Autos particulares, homogénea = {flag}")
            print(summary(g)[["n", "E_mediana", "absE_mediana", "pct_mejor"]].to_string())

    comp = []
    for a, b in [("MLP", C2),
                 (C2, C3),
                 ("Chain Ladder", "CL factores de mercado (C2)"),
                 ("Chain Ladder", "MLP"),
                 ("Chain Ladder", C2),
                 ("CL factores de mercado (C2)", C2),
                 ("Chain Ladder", "Bornhuetter-Ferguson"),
                 ("Chain Ladder", "CANN")]:
        todos, autos = paired(r, a, b), paired(pp, a, b)
        comp.append({
            "comparación": f"{a} → {b}",
            "n": todos["n"],
            "mejora_pct": todos["pct"],
            "p_signos": todos["p_signos"],
            "p_wilcoxon": todos["p_wilcoxon"],
            "auto_pct": autos["pct"],
            "auto_p_signos": autos["p_signos"],
            "auto_p_wilcoxon": autos["p_wilcoxon"],
        })
    comp = pd.DataFrame(comp).set_index("comparación")
    print("\nComparaciones pareadas (% de carteras en que el segundo mejora al primero,")
    print("con test de signos y Wilcoxon sobre los |E| pareados):")
    with pd.option_context("display.float_format", lambda v: f"{v:,.3g}"):
        print(comp.to_string())

    print("\nC3 (multirramo) contra C2 (mismo ramo), por ramo:")
    print(c3_vs_c2(r).to_string(float_format=lambda v: f"{v:,.3g}"))

    print(f"\nCobertura del intervalo Mack ±1,96·se sobre la reserva real: {cov['WMAPE'].mean()*100:.1f} % "
          f"(autos part.: {cov[cov.lob=='private_passenger_auto']['WMAPE'].mean()*100:.1f} %)")


if __name__ == "__main__":
    main()
