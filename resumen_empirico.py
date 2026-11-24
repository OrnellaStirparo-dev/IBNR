"""Tablas resumen de la validación empírica (lee resultados/empirico/*.csv)."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("resultados/empirico")
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


def paired(r: pd.DataFrame, a: str, b: str) -> float:
    """% de compañías en que *b* tiene menor |E| que *a*."""
    p = r.pivot_table(index=["lob", "group_code"], columns="model", values="E").abs()
    return float((p[b] < p[a]).mean() * 100)


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

    print("\nComparaciones pareadas (% de carteras en que el segundo mejora al primero):")
    for a, b in [("MLP", "MLP agrupado, mismo ramo (C2)"),
                 ("MLP agrupado, mismo ramo (C2)", "MLP agrupado, multirramo (C3)"),
                 ("Chain Ladder", "CL factores de mercado (C2)"),
                 ("Chain Ladder", "MLP"),
                 ("Chain Ladder", "MLP agrupado, mismo ramo (C2)"),
                 ("CL factores de mercado (C2)", "MLP agrupado, mismo ramo (C2)"),
                 ("Chain Ladder", "Bornhuetter-Ferguson"),
                 ("Chain Ladder", "CANN")]:
        print(f"  {a:32s} → {b:32s} {paired(r, a, b):5.1f} %   (autos part.: {paired(r[r.lob=='private_passenger_auto'], a, b):5.1f} %)")
    print(f"\nCobertura del intervalo Mack ±1,96·se sobre la reserva real: {cov['WMAPE'].mean()*100:.1f} % "
          f"(autos part.: {cov[cov.lob=='private_passenger_auto']['WMAPE'].mean()*100:.1f} %)")


if __name__ == "__main__":
    main()
