# PLAN — CIPA_FRAUD

Ejecución de punta a punta del framework **CIPA** (*Characterization, Indexing,
Profiling, Action*) sobre un conjunto extendido de datasets públicos de fraude
digital, y **estudio comparativo transversal** de la dificultad estructural entre
dominios de fraude.

- **Autor:** Luis García Rodríguez — Doctorado en Ciencias e Ingeniería de la
  Computación, IIMAS-UNAM.
- **Marco:** CIPA (COMIA 2026). Publicación: *"CIPA: A Multi-Domain Statistical
  Framework for Characterizing Imbalanced Datasets and Computing a Difficulty
  Score"*.
- **Fecha de planificación:** 2026-07-15.

---

## 0. Objetivo

Caracterizar, con el framework CIPA **real** (no un proxy), la dificultad
estructural de aprendizaje de un conjunto **extendido** de datasets públicos de
fraude digital, y compararlos entre sí y contra el benchmark multi-dominio
original de CIPA. Para cada dataset se obtiene:

- las 7 dimensiones de complejidad **D1–D7** ∈ [0,1],
- el **Difficulty Score** DS ∈ [0,1] y su banda (Low/Moderate/High/Extreme),
- la **Complexity Signature** (I–V),
- las **recomendaciones de acción** (métricas, preprocesamiento, familias de
  modelo, protocolo de validación).

El estudio produce un ranking de dificultad, perfiles por dimensión, distribución
de firmas y un análisis del **dominio fraude** como familia dentro del panorama
multi-dominio de CIPA.

---

## 1. Hallazgos del análisis previo (fundamento del proyecto)

Análisis directo de los tres proyectos involucrados:

### 1.1 CIPA (`.../2026-2/statistical_analysis`)
- Paquete Python maduro `src/cipa/` (v1.2.0): `dataset.py`, `pipeline.py`,
  `dimensions/d1..d7`, `ecol/` (F3, N1, N2, L1), `indexing/profiling/action`.
- **API pública estable:** `CIPADataset.from_arrays(X, y, name)` + `CIPAPipeline`.
  Contrato de entrada estricto (§5.2). Resultados serializables a `dict`/JSON.
- **`CIPAPipeline`** ya soporta escala: `knn_subsample` (subsample selectivo
  asimétrico para D2/D3/D4/D7 manteniendo full-N en D1/D5/D6), `n1_max_exact`,
  `large_n_subsample`. `random_state` para reproducibilidad.
- **Ya existe un arnés de experimentos** (`experiments/validate_table2.py` +
  `ADDING_DATASETS.md`) con un registro `LOADERS`/`TIER`/`TABLE_2` y un helper
  `_asymmetric_subsample`. Es el patrón a **reutilizar/imitar** (no reinventar).
- **Solapamiento clave:** el benchmark de CIPA ya incluye **3 datasets de fraude**
  como *Tier-2* (subsample genérico a N=10k, cargados desde CSV crudos):

  | CIPA key | ≙ dataset comparativo | DS (v1.1.0) | Banda | Firma |
  |----------|-----------------------|-------------|-------|-------|
  | CreditCard | `ulb_cc` | 0.3547 | Moderate | V |
  | PaySim | `paysim` | 0.3502 | Moderate | V |
  | IEEE-CIS | `ieee_cis` | 0.5679 | High | V |

  → Esto habilita un **check de consistencia** directo (CIPA_FRAUD recomputa estos
  3 sobre datos limpios + con features; ¿coincide la banda/firma? ¿cuánto mueve el
  DS el preprocesamiento vs. la carga cruda genérica del benchmark?).

### 1.2 Proyecto comparativo de fraude (`.../2027-1/fraud_dataset_comparative`)
- Pipeline S0–S9 **completo**: 11 datasets descargados, validados e ingeridos a
  **Parquet** por capas (`raw → interim → clean → features`), con AED, limpieza e
  ingeniería de características (188 features nuevas) y un informe comparativo.
- **Datos ya disponibles localmente** en `data/processed/<id>/{clean,features}/`
  → CIPA_FRAUD **no descarga ni copia datos**; los lee *in situ* (ahorro de disco,
  objetivo explícito).
- Cada dataset trae `_roles.json` (target, `feature_cols`, `id_cols`,
  `leakage_cols`, `added_cols`, `positive_label`) y `_features.json` (features
  derivadas) → **metadatos listos** para construir `(X, y)` sin adivinar columnas.
- **Gap que CIPA_FRAUD llena:** el "índice de dificultad" del informe comparativo
  (`reports/comparative/comparative.md` §2) es solo una **heurística barata**
  ("alineada con las dimensiones de CIPA": 4 componentes promediados —
  desbalanceo, dimensionalidad, faltantes, proxy de separabilidad). **No es CIPA.**
  CIPA_FRAUD ejecuta el framework riguroso completo → permite además **validar el
  proxy** contra el DS real (correlación de Spearman).

### 1.3 Inventario de datasets (registro canónico del proyecto comparativo)
11 datasets; **3 son de grafo (excluidos por decisión, §2)**. Set de trabajo =
**8 datasets tabulares**:

| id | Nombre | Dominio de fraude | N (registros) | ~Tasa fraude | Origen |
|----|--------|-------------------|---------------|--------------|--------|
| `ulb_cc` | ULB Credit Card | Tarjeta (card-present) | 284,807 | 0.17 % | Real |
| `ieee_cis` | IEEE-CIS (Vesta) | Card-not-present | 590,540 | 3.5 % | Real |
| `paysim` | PaySim | Dinero móvil | 6,362,620 | 0.13 % | Sintético |
| `saml_d` | SAML-D | Lavado (AML, 28 tipologías) | 9,504,852 | 0.10 % | Sintético |
| `banksim` | BankSim | Pagos bancarios | 594,643 | 1.2 % | Sintético |
| `sparkov` | Sparkov | Card-not-present | 1,852,394 | 0.52 % | Sintético |
| `baf` | Bank Account Fraud (Base) | Apertura de cuentas | 1,000,000 | 1.1 % | Sintético |
| `fdb` | FDB / fraudecom | E-commerce | ~151,000 | 9.4 % | Real |

**Excluidos (grafo):** `elliptic`, `elliptic_pp`, `amlworld`. Se documenta la
exclusión y se deja como trabajo futuro (caracterización topológica fuera del
alcance de CIPA, que es tabular-binario).

> Cobertura del estudio: 6 sintéticos + 2 reales; card-present, card-not-present,
> dinero móvil, AML, apertura de cuentas, e-commerce. Rango de N de 10⁵ a 10⁷ y de
> tasa de fraude de 0.10 % a 9.4 % → excelente diversidad para el comparativo.

---

## 2. Decisiones fijadas (con el usuario)

1. **Capa de entrada = AMBAS (`clean/` y `features/`).** Se corre CIPA sobre las
   columnas canónicas limpias **y** sobre la matriz con ingeniería de features, y
   se compara cuánto mueve la ingeniería al DS/firma (**sub-estudio E-FEAT**).
2. **Grafos excluidos.** Estudio sobre los **8 datasets tabulares**. Los 3 de
   grafo quedan como trabajo futuro.
3. **Escala = barrido multi-N.** Cada dataset se corre a **N ∈ {10k, 50k,
   full-N}**, donde *full-N* usa el subsample interno selectivo de CIPA
   (`knn_subsample`) para las dimensiones costosas y valores exactos en el resto.
   Se reporta la **sensibilidad del DS a N** (**sub-estudio E-SCALE**).

Defaults técnicos adoptados (revisables, §11):
- Semilla global `random_state = 42`.
- Subsample asimétrico: preserva **toda** la clase minoritaria, submuestrea la
  mayoritaria (patrón `_asymmetric_subsample` de CIPA). Se registra `IR_eff`.
- `knn_subsample` para full-N: **50,000** (equilibra fidelidad y tiempo; tabla de
  rendimiento de CIPA: N≤1M, d≤50 → ≤300 s).
- Imputación en el adaptador (no en la fuente): mediana por columna numérica;
  se descartan columnas con >50 % faltantes; `±Inf → NaN → imputa`. Todo
  registrado en el manifiesto (la fuente es inmutable; el proyecto comparativo
  aplicó política "sin imputación", así que puede quedar `NaN`, sobre todo en
  `ieee_cis`).

---

## 3. Contrato de datos: construcción de `(X, y)` (núcleo técnico)

CIPA exige (validado por `CIPADataset`): `X` 2-D `float64` **sin NaN/Inf**; `y`
1-D entera con **exactamente 2 valores**, minoría = `1`; `N ≥ 10`,
`n_minority ≥ 2`. El **adaptador** (`cipa_fraud/adapt.py`) transforma cada
`data.parquet` a `(X, y)` cumpliendo el contrato:

1. **Leer** `data/processed/<id>/<layer>/data.parquet` (Polars *lazy* / PyArrow).
   No se copia; se lee por ruta absoluta del proyecto comparativo.
2. **Target `y`:** columna `target` de `_roles.json`; `y = (col == positive_label)`
   → entero, minoría = 1 (verificar frecuencia; si la minoría no queda en 1,
   invertir y avisar).
3. **Columnas de `X`:** `feature_cols` de `_roles.json` (capa `clean`) o el
   conjunto de features de `_features.json` (capa `features`), **excluyendo**
   siempre: `target`, `id_cols`, `leakage_cols`, `added_cols`
   (`split`, `is_duplicate`, `is_labeled`, `*__outlier`, …).
4. **Codificación de categóricas:** columnas string/categoría → códigos ordinales
   (`pandas.Categorical.codes` / `polars` `.to_physical`) → `float64`. (Afecta a
   `banksim`, `sparkov`, `saml_d`, `ieee_cis`, `fdb`.)
5. **Faltantes/Inf:** `±Inf → NaN`; columnas con >50 % NaN se descartan; el resto
   se imputa con la mediana. Se cuenta y registra el % imputado.
6. **Cast** final a `float64`, verificación de forma y de dos clases.
7. **Escala (multi-N):** si `N > n_target`, aplicar subsample asimétrico al
   `n_target` del punto del barrido (10k, 50k) o dejar full-N con `knn_subsample`.
8. **Manifiesto** por corrida: `id`, capa, `N`, `d`, `IR`, `IR_eff`, hash del
   input, columnas usadas/descartadas, % imputado, versión CIPA, pesos, params de
   subsample, semilla, runtime.

> **Riesgo dimensionalidad:** `ieee_cis` tiene ~431 columnas crudas y muchos
> faltantes; el estimador KSG de D6 y la MST de N1 escalan con `d`. Mitigación:
> descartar columnas >50 % NaN, `knn_subsample`, y considerar tope de `d` si el
> tiempo excede la tabla de rendimiento (registrar como desviación).

---

## 4. Preguntas de investigación (estudio comparativo)

- **RQ1 — Ranking de dificultad.** ¿Cómo se ordenan los 8 datasets de fraude por
  DS? ¿Qué bandas dominan? ¿Coincide con la intuición del dominio (p. ej. AML y
  card-not-present más difíciles que pagos sintéticos)?
- **RQ2 — Perfiles y firmas.** ¿Qué dimensiones dominan en fraude (¿overlap D2,
  informatividad D6, dureza D3?)? ¿El dominio fraude tiende a la firma **V
  (Compound)**, como sugieren los 3 ya en el benchmark?
- **RQ3 — El fraude dentro del panorama multi-dominio de CIPA.** ¿Dónde caen estos
  datasets frente a los 13 del benchmark (Finance/Medical/Cyber/Industry/Bio)?
  ¿Forma el fraude un cluster reconocible en el espacio de 7 dimensiones?
- **RQ4 — Real vs. sintético / tipo de fraude.** ¿Difieren sistemáticamente los
  perfiles CIPA entre datasets reales y sintéticos, o entre tipos de fraude
  (tarjeta, AML, apertura de cuentas, e-commerce)?
- **RQ5 (E-FEAT) — Efecto de la ingeniería de features.** ¿Cuánto y en qué
  dirección mueve la ingeniería de características (S5) el DS y la firma respecto a
  la capa limpia? ¿Reduce el overlap / aumenta la informatividad?
- **RQ6 (E-SCALE) — Robustez a la escala.** ¿Es estable el DS/banda/firma bajo el
  barrido multi-N? ¿Qué dimensiones son sensibles al subsample (esperado: D1 por
  `IR_eff`)?
- **RQ7 — Validación del proxy.** ¿Correlaciona el "índice de dificultad"
  heurístico del proyecto comparativo con el DS real de CIPA (Spearman ρ)? ¿Qué
  componentes del proxy son fieles y cuáles engañan?
- **RQ8 (consistencia) — Reproducción del benchmark.** Para `ulb_cc`/`paysim`/
  `ieee_cis`: ¿coincide la banda/firma con las de CIPA v1.1.0? Diferencias
  atribuibles a limpieza + features vs. carga cruda genérica.
- **RQ9 (opcional) — Acciones.** Consolidado de recomendaciones (preprocesamiento,
  métricas, modelos) por dataset: ¿qué estrategias sugiere CIPA para el dominio?

---

## 5. Arquitectura y estructura del repositorio

Stack alineado con los proyectos hermanos: paquete Python modular en `src/`, CLI
(`typer`), lectura Parquet con **Polars/PyArrow**, reportes HTML/Markdown
auto-generados, figuras con `matplotlib`. **venv aislado** (lección del proyecto
comparativo: NO `--system-site-packages`; rompe la ABI de NumPy 2.x contra
Anaconda). `uv` no disponible.

```
cipa-fraud/
├── PLAN.md                      # este plan (fuente de verdad de fases)
├── CLAUDE.md                    # guía operativa para sesiones/agentes
├── README.md
├── pyproject.toml               # deps: cipa (editable local), polars, pyarrow,
│                                #   pandas, scikit-learn, matplotlib, typer, pydantic
├── requirements-lock.txt
├── src/cipa_fraud/
│   ├── settings.py              # rutas: raíz del proyecto comparativo, CIPA, salidas
│   ├── registry.py              # 8 datasets tabulares: id, capa, target, dominio,
│   │                            #   real/sintético, overlap con benchmark CIPA
│   ├── adapt.py                 # Parquet(clean|features) → (X, y) float64 (§3)
│   ├── run.py                   # orquesta CIPAPipeline por dataset×capa×N + manifiesto
│   ├── consistency.py           # RQ8: recompute vs CIPA v1.1.0 (ulb/paysim/ieee)
│   ├── compare/                 # RQ1–RQ7: tablas, rankings, clustering, correlaciones
│   ├── report/                  # HTML/MD comparativo + figuras (radar, heatmap, …)
│   └── cli.py                   # `cipa-fraud <cmd>`
├── results/
│   ├── <id>/<layer>/<N>.json    # CIPAResult.to_dict() + manifiesto por corrida
│   └── all_results.parquet      # consolidado tidy (una fila por corrida)
└── reports/
    ├── comparative/             # informe comparativo, tablas, figuras
    ├── consistency/             # RQ8
    └── reproducibility.json
```

**CLI previsto:**
```
cipa-fraud adapt   --dataset ulb_cc --layer features --n 50000   # inspecciona (X,y)
cipa-fraud run     --dataset ulb_cc --layer both --sweep 10000,50000,full
cipa-fraud run-all                                               # 8 × 2 capas × 3 N
cipa-fraud consistency                                           # RQ8
cipa-fraud compare                                              # RQ1–RQ7 → tablas/figuras
cipa-fraud report                                              # HTML/MD final
```

Instalación de CIPA: `pip install -e /home/luisgarcia/projects/unam/dcic/2026-2/statistical_analysis`
(dependencia editable local; sin duplicar el código del framework).

---

## 6. Fases de ejecución

Estilo S0–S9 del proyecto comparativo. Cada fase deja artefacto verificable y se
marca aquí el avance.

- [x] **F0 — Setup. ✅ COMPLETA (2026-07-15).** venv aislado (Python 3.12.7,
  sin `--system-site-packages`); CIPA v1.2.0 instalado editable + deps; paquete
  `cipa_fraud` con CLI `typer`; `settings.py` (rutas al comparativo, sin copiar
  datos) y `registry.py` de los 8 datasets. *Verificado:* `cipa-fraud doctor`
  sale con exit 0 — rutas OK, `import cipa` OK, los 8 datasets presentes en
  ambas capas (clean/features). `cipa-fraud list-datasets` OK.
- [x] **F1 — Adaptador `(X, y)`. ✅ COMPLETA (2026-07-15).** `adapt.py` lee
  `_roles`/`_features` y construye `(X, y)` cumpliendo el contrato CIPA. Reglas:
  capa **clean** = `feature_cols` (categóricas → códigos ordinales); capa
  **features** = numéricas + derivadas (`__te`/`__freq`/temporales/monto),
  categóricas crudas descartadas (representación de modelado). Faltantes: descarta
  cols >50% nulos e imputa mediana (solo afecta a `ieee_cis`). Escala: submuestreo
  **asimétrico** (minoría ≤ N/2, preserva positivos) o **estratificado** (preserva
  IR), registrando `mode`/`IR`/`IR_eff`. *Verificado:* los 16 combos (8×2 capas)
  cumplen contrato (float64, finito, 2 clases); ruta de memoria acotada para los
  gigantes (paysim 6.3M, saml_d 9.5M → 10k en <1s); `ruff` limpio. CLI:
  `cipa-fraud adapt <ds> --layer <clean|features> --n <10000|50000|full>`.
- [ ] **F2 — Smoke run.** Correr CIPA end-to-end sobre los más chicos (`fdb`,
  `banksim`) en las 3 configs de N; validar `CIPAResult` (DS, firma, acciones) y
  el formato de manifiesto/JSON. *Entregable:* resultados de 2 datasets + revisión
  de tiempos.
- [ ] **F3 — Ejecución completa.** `run-all`: 8 datasets × {clean, features} ×
  {10k, 50k, full}. Persistir JSON por corrida + `all_results.parquet`. Vigilar
  tiempos de `ieee_cis`/`saml_d`/`paysim` (alto N/d). *Entregable:* matriz de
  resultados completa + log de desviaciones.
- [ ] **F4 — Consistencia (RQ8).** Comparar `ulb_cc`/`paysim`/`ieee_cis` (config
  Tier-2 a 10k) contra CIPA v1.1.0 (banda/firma; ±0.10 en DS si aplica). Documentar
  el efecto limpieza+features vs. carga cruda. *Entregable:* `reports/consistency/`.
- [ ] **F5 — Análisis comparativo (RQ1–RQ7).** Tablas consolidadas, ranking DS,
  heatmap D1–D7, radar por dataset, distribución de firmas, clustering en el
  espacio de 7-D, ubicación frente al benchmark multi-dominio, real vs. sintético,
  delta E-FEAT (clean↔features), sensibilidad E-SCALE, correlación DS↔proxy (RQ7).
  *Entregable:* `src/cipa_fraud/compare/` + figuras.
- [ ] **F6 — Reporte.** Informe comparativo HTML/MD (panorama + ficha por dataset +
  hallazgos por RQ), `reproducibility.json` (versiones, hashes, semillas, params).
  *Entregable:* `reports/comparative/`.
- [ ] **F7 (opcional) — Write-up.** Materiales para artículo/capítulo de tesis:
  tabla estilo "Table 2" extendida al dominio fraude, figuras finales, narrativa.

---

## 7. Reproducibilidad

- Semilla global fija (`42`); `random_state` propagado a `CIPAPipeline` y al
  subsample.
- Manifiesto por corrida (§3.8) + `reproducibility.json` global (versión CIPA,
  versiones de deps, hash de cada `data.parquet` de entrada, pesos, params).
- Datos crudos/procesados **no** se copian ni versionan (se leen del proyecto
  comparativo). `results/` y `reports/` sí se versionan; JSON grandes opcionalmente
  ignorados y regenerables por CLI.
- Determinismo: evitar operaciones dependientes del orden de columnas; fijar orden
  de `feature_cols` desde el registro.

---

## 8. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|--------|---------|-----------|
| `ieee_cis`: d alto + muchos faltantes | D6/N1 lentos o inestables | descartar cols >50 % NaN, `knn_subsample`, tope de `d` documentado |
| N enorme (`saml_d` 9.5M, `paysim` 6.3M) | tiempos/memoria | Polars *lazy*; full-N vía `knn_subsample=50k`; barrido controla el costo |
| Subsample desplaza `IR_eff` → D1/DS | comparabilidad | registrar `IR_eff`; interpretar D1 con cuidado; Tier-2 solo para banda/firma |
| Categóricas de alta cardinalidad | ruido en distancias | códigos ordinales + nota; evaluar drop de IDs disfrazados |
| Datos con "sin imputación" (NaN) | CIPA falla | imputación en adaptador (mediana), documentada, no muta la fuente |
| Rutas absolutas al proyecto comparativo | fragilidad | centralizar en `settings.py`; validar existencia en F0 |
| Deriva de versión de CIPA | resultados no reproducibles | fijar commit/tag de CIPA en `reproducibility.json` |

---

## 9. Decisiones abiertas (menores, con default)

1. **`knn_subsample` para full-N:** default 50k. ¿Subir a 100k para `ulb_cc`
   (N=285k, d bajo)? — decidir tras medir tiempos en F2.
2. **Puntos del barrido N:** default {10k, 50k, full}. ¿Añadir 100k? — según F2.
3. **Pesos CIPA:** usar los por defecto `(0.10, 0.22, 0.18, 0.15, 0.10, 0.12,
   0.13)`. ¿Réplica del `weight_sensitivity` de CIPA sobre el subconjunto fraude?
   — opcional en F7.
4. **Tope de `d`:** solo si `ieee_cis` excede presupuesto de tiempo; registrar.
5. **Versionado de CIPA:** ¿instalar desde ruta local (editable) o pinear al tag
   de GitHub? — default: editable local; registrar el hash del commit.

---

## 10. Resumen ejecutivo

CIPA_FRAUD toma los **8 datasets tabulares** de fraude ya procesados por el
proyecto comparativo (sin recopiar datos), construye para cada uno la matriz
`(X, y)` que exige CIPA, y ejecuta el framework **real** en un diseño factorial
**2 capas (clean/features) × 3 escalas (10k/50k/full)**. Con los resultados
produce un estudio comparativo (ranking de dificultad, perfiles D1–D7, firmas,
ubicación del fraude en el panorama multi-dominio de CIPA), valida el proxy
heurístico previo contra el DS real, y reconcilia los 3 datasets de fraude que ya
estaban en el benchmark de CIPA. Entregable final: paquete reproducible + informe
comparativo.
