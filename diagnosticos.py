"""
Diagnósticos complementarios que cruzan el caso argentino con el empírico.

Ninguno recalcula estimadores ya reportados: los tres leen las salidas que
``main.py``, ``simulaciones.py`` y ``empirico.py`` dejaron en ``resultados/``,
salvo el barrido de semillas del MLP agrupado, que necesita reajustar esa red.

1. ``homogeneidad_argentina``
   Aplica al triángulo argentino el mismo diagnóstico que ``empirico.py``
   corre sobre las 187 carteras de Schedule P. Sin esto la comparación entre
   ambos casos queda incompleta: se exige homogeneidad a las carteras reales y
   no al triángulo propio. Se mide sobre el triángulo en moneda constante,
   porque en pesos corrientes el diagnóstico sólo detectaría inflación.

2. ``semillas_mlp_agrupado``
   El resultado central del capítulo empírico es que agrupar compañías mejora
   al MLP de un triángulo único. Esa comparación se apoya en una sola
   inicialización de la red, de modo que conviene saber cuánto se mueve el
   resultado al cambiarla.

3. ``comparacion_casos``
   Si los dos casos ordenan los estimadores de forma parecida, el caso
   argentino —simulado y con una unidad de cuenta inestable— informa sobre el
   comportamiento relativo de los métodos más allá de su propia calibración.
   La correlación de órdenes es la forma mínima de verificarlo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from src.ai.pooled_mlp import run_pooled_mlp
from src.data.sample_policies import calibrated_exposure
from src.data.schedule_p import (
    LINES, full_square, load_schedule_p, premium, upper_triangle, usable,
)
from src.evaluation.homogeneity import (
    MAX_F1_CV, MAX_LR_CV, MAX_PREMIUM_JUMP, TREND_P, diagnose,
)

OUT = Path(__file__).parent / "resultados"
OUT_EMP = OUT / "empirico"
SEEDS = (1, 7, 42, 123)

#: Los siete estimadores presentes en los dos casos. El híbrido cambia de
#: nombre entre una salida y la otra.
MODELOS = {
    "Bornhuetter-Ferguson": "Bornhuetter-Ferguson",
    "CANN": "CANN",
    "Credibility Hybrid": "Híbrido de credibilidad",
    "Chain Ladder": "Chain Ladder",
    "Bayesian Chain Ladder": "Bayesian Chain Ladder",
    "GRU": "GRU",
    "MLP": "MLP",
}


# --------------------------------------------------------------------- 1
def homogeneidad_argentina() -> pd.DataFrame:
    """
    Diagnóstico de homogeneidad del triángulo argentino en moneda constante.

    Se reusa ``homogeneity.diagnose`` sin modificarlo, con la medida de
    exposición en el lugar de la prima: el Anuario de la SSN no publica primas,
    y los vehículos expuestos son la magnitud con la que el propio organismo
    normaliza la frecuencia.
    """
    piv = pd.read_csv(OUT / "triangulo_real_MARS.csv", index_col=0)
    piv.columns = [int(c) for c in piv.columns]
    piv.index = [int(i) for i in piv.index]
    exposure = calibrated_exposure().reindex(piv.index)

    d = diagnose(piv, exposure)
    P = exposure.values.astype(float)
    anio_salto = int(piv.index[1:][int(np.argmax(np.abs(np.diff(P) / P[:-1])))])

    filas = [
        ("volumen", "salto máximo de exposición entre años", d["max_premium_jump"],
         MAX_PREMIUM_JUMP, d["volume_flag"], f"máximo en {anio_salto}"),
        ("siniestralidad", "CV de C[i,1] / vehículos expuestos", d["lr1_cv"],
         MAX_LR_CV, d["lr_flag"], ""),
        ("desarrollo", "CV de f[i,1] = C[i,2]/C[i,1]", d["f1_cv"],
         MAX_F1_CV, d["f1_cv"] > MAX_F1_CV, ""),
        ("desarrollo", "tendencia de Spearman de f[i,1]", d["f1_trend_rho"],
         np.nan, d["f1_trend_p"] < TREND_P, f"p = {d['f1_trend_p']:.4f}"),
    ]
    out = pd.DataFrame(filas, columns=["dimension", "indicador", "valor", "umbral",
                                       "supera_umbral", "nota"])
    out.loc[len(out)] = ["global", "homogéneo", float(d["homogeneous"]), np.nan,
                         not d["homogeneous"],
                         "marcado por desarrollo" if d["dev_flag"] else ""]
    return out


# --------------------------------------------------------------------- 2
def _schedule_p_triangulos() -> tuple[dict, dict]:
    """Cuadrados completos y triángulos superiores utilizables, como en empirico.py."""
    d = load_schedule_p()
    squares, uppers = {}, {}
    for lob in LINES:
        for code in sorted(d.loc[d.lob == lob, "group_code"].unique()):
            sq, pr = full_square(d, lob, code), premium(d, lob, code)
            if usable(sq, pr):
                squares[(lob, code)] = sq
                uppers[(lob, code)] = upper_triangle(sq)
    return squares, uppers


def semillas_mlp_agrupado(seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    """
    Reajusta el MLP agrupado por ramo (C2) con distintas semillas.

    Se compara contra el Chain Ladder de ``resultados_por_compania.csv``, que
    es determinístico y por lo tanto el mismo en todas las corridas.
    """
    squares, uppers = _schedule_p_triangulos()
    res = pd.read_csv(OUT_EMP / "resultados_por_compania.csv", dtype={"group_code": str})
    cl = (res[res.model == "Chain Ladder"]
          .set_index(["lob", "group_code"])["E"].abs())

    verdad = {}
    for k, sq in squares.items():
        up = uppers[k]
        paid = np.array([r[np.isfinite(r)][-1] for r in up.values])
        verdad[k] = (paid, float(np.nansum(sq[10].values - paid)))

    filas = []
    for seed in seeds:
        pooled = {}
        for lob in LINES:
            keys = [k for k in uppers if k[0] == lob]
            pooled.update(run_pooled_mlp({k: uppers[k] for k in keys},
                                         use_line=False, mc_samples=100, seed=seed))
        err = {}
        for k, ult in pooled.items():
            paid, r_true = verdad[k]
            err[k] = abs(float(np.nansum(ult.reindex(uppers[k].index).values - paid))
                         - r_true) / r_true * 100
        e = pd.Series(err)
        e.index = pd.MultiIndex.from_tuples(e.index, names=["lob", "group_code"])
        d = pd.concat([e.rename("mlp"), cl.rename("cl")], axis=1).dropna()
        gana = int((d.mlp < d.cl).sum())
        filas.append({
            "semilla": seed,
            "n": len(d),
            "absE_mediana": e.median(),
            "mejor_que_CL_pct": 100.0 * gana / len(d),
            "p_signos": float(binomtest(gana, len(d), 0.5).pvalue),
        })
        print(f"  semilla {seed}: mediana |E| {e.median():.1f} %, "
              f"mejor que CL en {100 * gana / len(d):.1f} % (p = {filas[-1]['p_signos']:.4f})")
    return pd.DataFrame(filas)


# --------------------------------------------------------------------- 3
def comparacion_casos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    |E| de cada estimador en los dos casos, y correlación entre los órdenes.

    El caso argentino se toma en moneda constante, que es el régimen
    comparable: las carteras de Schedule P están en una moneda estable, de modo
    que contrastar contra la corrida nominal mediría sobre todo inflación.
    """
    arg42 = (pd.read_csv(OUT / "metricas_moneda_constante.csv")
             .set_index("model")["pct_error_ibnr"].abs())
    sim = pd.read_csv(OUT / "simulaciones_resumen.csv")
    arg30 = sim[sim.unidad == "constante"].set_index("modelo")["absE_media"]

    res = pd.read_csv(OUT_EMP / "resultados_por_compania.csv", dtype={"group_code": str})
    diag = pd.read_csv(OUT_EMP / "homogeneidad.csv", dtype={"group_code": str})
    r = res[res.model != "_mack_cobertura"].merge(
        diag[["lob", "group_code", "homogeneous"]], on=["lob", "group_code"])
    pp_hom = r[(r.lob == "private_passenger_auto") & r.homogeneous]

    filas = []
    for arg_name, sp_name in MODELOS.items():
        filas.append({
            "modelo": sp_name,
            "arg_semilla42": arg42.get(arg_name, np.nan),
            "arg_media30": arg30.get(arg_name, np.nan),
            "sp_autos_homogeneas": pp_hom.loc[pp_hom.model == sp_name, "E"].abs().median(),
            "sp_187": r.loc[r.model == sp_name, "E"].abs().median(),
        })
    tab = pd.DataFrame(filas).set_index("modelo").sort_values("arg_semilla42")

    corr = []
    for col in ("sp_autos_homogeneas", "sp_187"):
        rho, p = spearmanr(tab["arg_semilla42"], tab[col])
        corr.append({"contra": col, "spearman_vs_arg_semilla42": rho, "p": p,
                     "n_modelos": len(tab)})
    return tab, pd.DataFrame(corr)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    OUT_EMP.mkdir(parents=True, exist_ok=True)

    print("=== 1. Homogeneidad del triángulo argentino (moneda constante) ===")
    hom = homogeneidad_argentina()
    hom.to_csv(OUT / "homogeneidad_argentina.csv", index=False)
    print(hom.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n=== 2. MLP agrupado (C2) por semilla ===")
    sem = semillas_mlp_agrupado()
    sem.to_csv(OUT_EMP / "semillas_mlp_agrupado.csv", index=False)
    print(sem.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n=== 3. Caso argentino vs. Schedule P ===")
    tab, corr = comparacion_casos()
    tab.reset_index().to_csv(OUT / "comparacion_casos.csv", index=False)
    print(tab.to_string(float_format=lambda v: f"{v:,.1f}"))
    print()
    print(corr.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print(f"\nSalidas → {OUT}/ y {OUT_EMP}/")


if __name__ == "__main__":
    main()
