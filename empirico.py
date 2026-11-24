"""
Validación sobre cartera empírica — NAIC Schedule P (CAS Loss Reserving Database).

Responde a los comentarios 2 y 5 de la revisión:

* Comentario 2: los estimadores se miden contra el *ultimate* efectivamente
  pagado por cada compañía (triángulo inferior observado hasta 1997+9 = 2006),
  no contra un generador sintético.
* Comentario 5: antes de modelar se diagnostica la homogeneidad de cada cartera
  en el tiempo (``src.evaluation.homogeneity``), y la vinculación entre
  triángulos se hace explícita en tres configuraciones:

  C1  un único triángulo (lo que tiene una aseguradora sola): los 8 estimadores
      del caso argentino;
  C2  varias compañías del mismo ramo: Chain Ladder con factores de mercado y
      MLP agrupado con indicador de compañía;
  C3  varias compañías y varios ramos: MLP agrupado con indicadores de compañía
      y de ramo.

Objeto estimado: triángulo de pagos acumulados → reserva de siniestros
pendientes R_i = U_i − C_{i,n+1−i} (comentario 3).

Uso:  python empirico.py [--lines ppauto|all] [--fast]
Salidas en resultados/empirico/.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai.cann import run_cann
from src.ai.gru_triangle import run_gru_triangle
from src.ai.hybrid_models import run_credibility_hybrid
from src.ai.pooled_mlp import run_pooled_mlp
from src.ai.pytorch_mlp import run_pytorch_mlp
from src.data.schedule_p import (
    LINES, full_square, load_schedule_p, premium, to_long, upper_triangle, usable,
)
from src.evaluation.homogeneity import diagnose, pattern_distance
from src.traditional.bayesian_chainladder import run_bayesian_chainladder
from src.traditional.bf_model import run_bornhuetter_ferguson
from src.traditional.chain_ladder_manual import _weighted_ldfs, ultimate_from_pivot
from src.traditional.mack_model import mack_chainladder

warnings.filterwarnings("ignore")
OUT = Path("resultados/empirico")


def latest(piv: pd.DataFrame) -> np.ndarray:
    return np.array([piv.iloc[i].dropna().iloc[-1] for i in range(len(piv))])


def cl_with_factors(piv: pd.DataFrame, f: np.ndarray) -> pd.Series:
    C = piv.values.astype(float)
    n = C.shape[1]
    u = []
    for i in range(len(C)):
        j0 = max(j for j in range(n) if np.isfinite(C[i, j]))
        u.append(C[i, j0] * np.prod(f[j0:]))
    return pd.Series(u, index=piv.index)


def market_pivot(uppers: dict, keys: list) -> pd.DataFrame:
    return sum(uppers[k].fillna(0) for k in keys).where(next(iter(uppers.values())).notna())


def main(lines: list[str], fast: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_schedule_p()
    mc = 30 if fast else 100

    # ------------------------------------------------------------------ datos
    squares, uppers, prems = {}, {}, {}
    for lob in LINES:
        for code in sorted(d.loc[d.lob == lob, "group_code"].unique()):
            sq, pr = full_square(d, lob, code), premium(d, lob, code)
            if usable(sq, pr):
                squares[(lob, code)], uppers[(lob, code)], prems[(lob, code)] = sq, upper_triangle(sq), pr
    print("Triángulos utilizables por ramo:",
          pd.Series([k[0] for k in squares]).value_counts().to_dict())

    # ---------------------------------------------------- homogeneidad (C5)
    diag = pd.DataFrame({k: diagnose(uppers[k], prems[k]) for k in squares}).T
    diag.index.names = ["lob", "group_code"]
    diag.to_csv(OUT / "homogeneidad.csv")
    print("\nCarteras homogéneas por ramo:")
    print(diag.groupby(level=0)["homogeneous"].agg(["sum", "count"]))

    mkt_f = {lob: _weighted_ldfs(market_pivot(uppers, [k for k in uppers if k[0] == lob]).values)
             for lob in LINES}
    dist = pd.DataFrame({a: {b: pattern_distance(mkt_f[a], mkt_f[b]) for b in LINES} for a in LINES})
    dist.to_csv(OUT / "distancia_patrones_ramos.csv")
    pd.DataFrame(mkt_f).T.to_csv(OUT / "factores_mercado_por_ramo.csv")
    print("\nDistancia máxima entre curvas de % desarrollado (mercado):\n", dist.round(3))

    # ------------------------------------------------------- C2/C3 agrupados
    t0 = time.time()
    pooled_line = {}
    for lob in LINES:
        keys = [k for k in uppers if k[0] == lob]
        pooled_line.update(run_pooled_mlp({k: uppers[k] for k in keys}, use_line=False, mc_samples=mc))
    pooled_all = run_pooled_mlp(uppers, use_line=True, mc_samples=mc)
    print(f"\nMLP agrupados ajustados ({time.time() - t0:.0f}s)")

    # --------------------------------------------------------- evaluación
    targets = [k for k in squares if k[0] in lines]
    rows, per_ay = [], []
    t0 = time.time()
    for n, k in enumerate(targets, 1):
        lob, code = k
        sq, up, pr = squares[k], uppers[k], prems[k]
        long = to_long(up)
        paid = latest(up)
        true_R = sq[10].values - paid

        others = [o for o in uppers if o[0] == lob and o != k]
        mkt = market_pivot(uppers, others)
        f_mkt = _weighted_ldfs(mkt.values)
        # A priori BF externo: siniestralidad de mercado (resto de las
        # compañías del ramo, con su propio Chain Ladder) × prima de la compañía.
        elr_mkt = ultimate_from_pivot(mkt).values / sum(prems[o] for o in others).values
        apriori = pd.Series(elr_mkt * pr.values, index=up.index)

        est = {}
        est["Chain Ladder"] = ultimate_from_pivot(up)
        est["Bornhuetter-Ferguson"] = run_bornhuetter_ferguson(long, apriori=apriori)
        mack = mack_chainladder(up)
        est["Bayesian Chain Ladder"] = run_bayesian_chainladder(long)[0]
        uq = {}
        for name, fn in [("MLP", lambda: run_pytorch_mlp(long, epochs=800 if fast else 1500, mc_samples=mc)[-2:]),
                         ("GRU", lambda: run_gru_triangle(long, epochs=200 if fast else 400, mc_samples=mc)[-2:])]:
            try:
                m, s = fn()
                est[name], uq[name] = m, (m, s)
            except Exception as exc:
                print(f"   {k} {name}: {exc}")
        try:
            est["CANN"] = run_cann(long, mc_samples=mc)[0]
        except Exception as exc:
            print(f"   {k} CANN: {exc}")
        if uq:
            est["Híbrido de credibilidad"] = run_credibility_hybrid(uq, anchor=est["Chain Ladder"], anchor_weight=0.20)
        est["CL factores de mercado (C2)"] = cl_with_factors(up, f_mkt)
        est["MLP agrupado, mismo ramo (C2)"] = pooled_line[k]
        est["MLP agrupado, multirramo (C3)"] = pooled_all[k]

        R_true = float(np.nansum(true_R))
        for name, u in est.items():
            R_hat = u.reindex(up.index).values - paid
            rows.append({
                "lob": lob, "group_code": code, "model": name,
                "R_true": R_true, "R_hat": float(np.nansum(R_hat)),
                "E": (float(np.nansum(R_hat)) - R_true) / R_true * 100,
                "WMAPE": float(np.nansum(np.abs(R_hat - true_R)) / R_true * 100),
            })
            for ay, rh, rt in zip(up.index, R_hat, true_R):
                per_ay.append({"lob": lob, "group_code": code, "model": name,
                               "accident_year": ay, "R_hat": rh, "R_true": rt})
        se = float(mack.loc["TOTAL", "se"])
        rows.append({"lob": lob, "group_code": code, "model": "_mack_cobertura",
                     "R_true": R_true, "R_hat": float(mack.loc["TOTAL", "ibnr"]), "E": np.nan,
                     "WMAPE": float(abs(mack.loc["TOTAL", "ibnr"] - R_true) <= 1.96 * se)})
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(targets)}  ({el:.0f}s, ~{el / n * (len(targets) - n):.0f}s restantes)")
            pd.DataFrame(rows).to_csv(OUT / "resultados_por_compania.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "resultados_por_compania.csv", index=False)
    pd.DataFrame(per_ay).to_csv(OUT / "resultados_por_anio.csv", index=False)
    print("Listo →", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="ppauto")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    main(["private_passenger_auto"] if a.lines == "ppauto" else list(LINES), a.fast)
