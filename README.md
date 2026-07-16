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
(10k/50k/full)**, produciendo por dataset: dimensiones D1–D7, Difficulty Score,
Complexity Signature y recomendaciones de acción. Luego consolida un estudio
comparativo (ranking, perfiles, firmas, validación del proxy heurístico previo y
reconciliación con el benchmark original de CIPA).

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

```bash
cipa-fraud list-datasets              # inventario del estudio
cipa-fraud doctor                     # diagnostico de entorno/rutas/datos
cipa-fraud adapt --dataset ulb_cc --layer features --n 50000   # (F1)
cipa-fraud run --dataset ulb_cc --layer both --sweep 10000,50000,full  # (F2/F3)
cipa-fraud run-all                    # (F3)
cipa-fraud consistency                # (F4)
cipa-fraud compare                    # (F5)
cipa-fraud report                     # (F6)
```

## Estado

F0 (setup) completo. Fases siguientes en `PLAN.md` (F1–F7).
