# Reservas IBNR — Automotores Casco (SSN)

Código de la tesina *Análisis comparativo de métodos tradicionales en la proyección
de reservas de siniestros no reportados y modelos de Inteligencia Artificial*.

Compara ocho estimadores de reserva IBNR sobre un triángulo de run-off del ramo
Automotores–Casco, ajustando cada uno **dos veces**: sobre el triángulo nominal y
sobre el triángulo deflactado a moneda constante, con la reserva proyectada
re-inflada a los años calendario en que se va a pagar.

```
python main.py
```

Escribe las tablas en `resultados/` y las figuras en `figures/`.

## De dónde salen los datos

El Anuario Estadístico de la SSN (`ssn_2025_siniestro_casco.xlsx`, Anexo II)
publica exposición, siniestros y frecuencia por hecho generador y por año, pero
**no publica montos ni desarrollo por año de ocurrencia**, y los datos
individuales de siniestros son confidenciales. Un triángulo de run-off no se
puede leer de la fuente oficial: hay que simularlo a partir de ella. Lo que sí
queda auditable es el límite entre lo que viene del dato y lo que es supuesto.

| Componente | Origen |
|---|---|
| Vehículos expuestos a casco | Cuadro 1, `TOTAL − categoría A` (A es RC sin cobertura de casco) |
| Siniestros y frecuencia | Cuadros 2 y 4 |
| Mix por hecho generador | Cuadro 2, año por año |
| Tendencia de severidad | Serie de IPC (`src/data/inflation.py`) |
| Severidad base, dispersión, patrón de pago | **Supuestos** |

Los años 2015–2024 salen del Anuario; los anteriores se retro-extrapolan y quedan
marcados como tales en `calibration_provenance()`.
`ssn_anuario.verify_consistency()` cruza los tres cuadros entre sí: el cociente
`Cuadro 2 / Cuadro 4` reproduce la exposición derivada del Cuadro 1 con un desvío
menor al 0,02%.

La serie de IPC tiene la fuente anotada por año. INDEC estuvo intervenido entre
2007 y 2015, así que ese tramo empalma índices provinciales y alternativos.
Para usar otra serie, apuntá `IBNR_CPI_CSV` a un CSV con columnas
`anio,variacion_pct`.

## Modelos

| Sección de la tesis | Modelo | Módulo |
|---|---|---|
| 2.1 | Chain Ladder | `traditional/chainladder_model.py` |
| 2.2 | Bornhuetter-Ferguson | `traditional/bf_model.py` |
| 2.3 | Mack Chain Ladder, con MSEP | `traditional/mack_model.py` |
| 3.1 | Bayesian Chain Ladder | `ai/bayesian_chainladder.py` |
| 3.2 | MLP | `ai/pytorch_mlp.py` |
| 3.3 | GRU / DeepTriangle | `ai/gru_triangle.py` |
| 4.1 | CANN sobre GLM Poisson sobredisperso | `ai/cann.py` |
| 4.2 | Híbrido de credibilidad | `ai/hybrid_models.py` |

## Cómo se mide el error

Contra el **ultimate realmente observado** por año de ocurrencia, que se conoce
porque el triángulo es sintético (hoja `ultimates_reales` del workbook generado).

Nunca contra la última diagonal. La diagonal es el pagado a la fecha, así que el
"error" que devuelve esa comparación es la reserva misma: un modelo que constituye
IBNR cero saca MAPE 0 y R² 1. `compare_models_results` acepta esa comparación como
fallback pero la etiqueta como sesgada.

## Estructura

```
main.py                     corrida completa: nominal, moneda constante, sensibilidades
src/data/
  ssn_anuario.py            lectura y verificación cruzada de los Cuadros SSN
  inflation.py              serie de IPC, deflactado y re-inflado
  sample_policies.py        generador calibrado a la SSN + ground truth
  transformer.py            triángulo de run-off con corte a fecha de valuación
  features.py               parametrización en log-factor de desarrollo
src/traditional/            Chain Ladder, BF, Mack
src/ai/                     Bayesian CL, MLP, GRU, CANN, híbrido
src/evaluation/
  real_terms.py             ajuste en moneda constante y vuelta a nominal
  sensitivity.py            barridos del ancla y de la inflación futura
  export.py                 tablas numeradas para el documento
src/visualization/          figuras
```

## Supuestos que hay que declarar en el documento

- **Severidad base por tipo de siniestro.** El Anuario de Casco no tiene ningún
  dato monetario. Es el único parámetro que no se pudo anclar a la fuente oficial.
- **Inflación futura.** Reservar en moneda constante obliga a explicitarla: la
  reserva se paga en años que todavía no ocurrieron. Un Chain Ladder nominal no
  evita el supuesto, lo esconde — sus factores extrapolan implícitamente la
  inflación que traiga el triángulo histórico. `sensitivity.py` reporta el barrido.
- **Peso del ancla en el híbrido.** Es un juicio, no un resultado.
