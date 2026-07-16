# CIPA_FRAUD

Ejecución de punta a punta del framework **CIPA** (*Characterization, Indexing,
Profiling, Action*) sobre un conjunto extendido de datasets públicos de **fraude
digital**, y estudio comparativo transversal de su dificultad estructural.

Doctorado en Ciencias e Ingeniería de la Computación — IIMAS-UNAM.
Autor: Luis García Rodríguez. Ver [`PLAN.md`](PLAN.md) para el plan completo.

## Qué hace

Para cada uno de los **8 datasets tabulares** de fraude (ya procesados por el
proyecto comparativo hermano), construye la matriz `(X, y)` que exige CIPA y
ejecuta el framework real en un diseño **2 capas (clean/features) × 3 escalas
(10k/50k/full)** = 48 corridas, produciendo por dataset: dimensiones D1–D7,
Difficulty Score (DS), Complexity Signature y recomendaciones de acción. Luego
consolida un estudio comparativo (ranking, perfiles, firmas, efecto de las
features, sensibilidad a la escala, validación del proxy heurístico previo y
reconciliación con el benchmark original de CIPA), un informe HTML/Markdown y
materiales de tesis.

Los datos **no se copian**: se leen in situ desde
`/home/luisgarcia/projects/unam/dcic/2027-1/fraud_dataset_comparative/data/processed`.

## Setup

```bash
python3 -m venv .venv                 # venv aislado (NO --system-site-packages)
source .venv/bin/activate
pip install -e .                      # este paquete
pip install -e /home/luisgarcia/projects/unam/dcic/2026-2/statistical_analysis  # CIPA
cipa-fraud doctor                     # verifica entorno, rutas y datos
```

## Uso

El pipeline completo se reproduce de forma determinista (semilla global 42):

```bash
cipa-fraud list-datasets              # inventario del estudio
cipa-fraud doctor                     # diagnostico de entorno/rutas/datos
cipa-fraud adapt ulb_cc --layer features --n 50000            # inspecciona (X, y)  (F1)
cipa-fraud run ulb_cc --layer both --sweep 10000,50000,full   # una corrida        (F2)
cipa-fraud run-all                    # barrido 8×2×3 = 48 corridas               (F3)
cipa-fraud consistency                # reconciliacion con el benchmark (RQ8)      (F4)
cipa-fraud compare                    # analisis comparativo (RQ1–RQ7)            (F5)
cipa-fraud report                     # informe HTML/MD + reproducibility.json     (F6)
cipa-fraud writeup                    # Table 2 (LaTeX/MD) + narrativa de tesis     (F7)
```

Las salidas viven en `results/` y `reports/`. Son **artefactos derivados,
regenerables** con los comandos de arriba, por lo que **no se versionan** (solo el
código y `PLAN.md` son la fuente de verdad); `report` deja además la trazabilidad
completa en `reports/reproducibility.json` (versiones, semilla, parámetros y
SHA-256 de cada Parquet de entrada).

## Resultados

Ranking de dificultad de los 8 datasets en el corte canónico *features / N=full*
(escala CIPA: DS ∈ [0,1], mayor = más difícil):

| # | Dataset | Dominio | Origen | DS | Banda | Firma |
|---|---|---|---|---|---|---|
| 1 | fdb | e-commerce | real | 0.529 | **High** | III |
| 2 | ulb_cc | card-present | real | 0.487 | Moderate | V |
| 3 | ieee_cis | card-not-present | real | 0.476 | Moderate | V |
| 4 | baf | account-opening | sintético | 0.446 | Moderate | V |
| 5 | saml_d | AML | sintético | 0.426 | Moderate | V |
| 6 | sparkov | card-not-present | sintético | 0.386 | Moderate | V |
| 7 | paysim | mobile-money | sintético | 0.276 | Moderate | V |
| 8 | banksim | bank-payments | sintético | 0.262 | Moderate | V |

Hallazgos principales (RQ1–RQ8):

- **Dificultad intermedia y compuesta.** El fraude tabular se concentra en la banda
  *Moderate* (7/8) y en la firma **V — Compound** (7/8): la dificultad no viene de
  una sola causa, sino de la conjunción de **desbalance extremo (D1)** y
  **fragmentación de la clase minoritaria (D4)**, las dos dimensiones más intensas
  del dominio.
- **Las features facilitan el problema sin recategorizarlo (E-FEAT).** La ingeniería
  de características baja el DS en **8/8** datasets (ΔDS medio −0.028), reduciendo
  sobre todo la dureza (D3) y la informatividad requerida (D6), sin cambiar ninguna
  firma.
- **Robustez a la escala (E-SCALE).** Bajo el barrido multi-N, la dimensión más
  sensible al submuestreo es **D1** (vía la razón de desbalanceo efectiva); la banda
  se mantiene estable en 7/16 configuraciones.
- **El proxy heurístico no predice la dificultad real (RQ7).** El índice de
  dificultad del proyecto comparativo **no** correlaciona con el DS de CIPA (Spearman
  ρ = 0.19, p = 0.65); el caso extremo es *fdb*, que el proxy califica como trivial
  pero CIPA identifica como el más difícil del conjunto.
- **Reproduce el benchmark original (RQ8).** Sobre `ulb_cc`/`paysim`/`ieee_cis`
  —también presentes en el benchmark de CIPA— banda y firma coinciden en **6/6**
  comparaciones; el DS se desplaza a lo sumo 0.11, atribuible a la limpieza +
  ingeniería de características frente a la carga cruda genérica.

El estudio completo (tablas, figuras CVD-safe, informe HTML y narrativa) se
reconstruye con `cipa-fraud compare && cipa-fraud report && cipa-fraud writeup`.

## Estado

**F0–F7 completas: estudio terminado end-to-end.** Setup, adaptador `(X, y)`,
orquestación (`run`/`run-all`), barrido completo, consistencia (RQ8), análisis
comparativo (RQ1–RQ7), informe + `reproducibility.json` y write-up de tesis. Detalle
por fase en [`PLAN.md`](PLAN.md).
