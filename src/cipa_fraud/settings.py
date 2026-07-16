"""Configuración central de rutas, semilla y parámetros de escala de CIPA_FRAUD.

Fuente única de rutas del proyecto. Principios que este módulo materializa:

- **Los datos no se copian:** se leen in situ del proyecto comparativo de fraude
  (:data:`PROCESSED_ROOT`), para ahorrar disco. Aquí solo viven rutas, nunca
  copias de los Parquet.
- **La fuente es inmutable:** el *target* y los roles de columna se leen del
  ``_roles.json`` de origen (:func:`roles_path`), no se hardcodean.
- **Reproducibilidad:** la semilla global (:data:`RANDOM_STATE`) y la política de
  escala (:data:`SWEEP_N`, :data:`KNN_SUBSAMPLE_FULL`) están centralizadas aquí
  para que toda corrida sea trazable desde un solo lugar.

Las rutas externas se validan con :func:`check_paths` (invocada por la CLI en el
comando ``doctor``).

--------------------------------------------------------------------------
Universidad Nacional Autónoma de México (UNAM)
Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
Programa de Posgrado en Ciencia e Ingeniería de la Computación (PCIC)

Autor:  Luis García Rodríguez  <luis.garcia@unam.edu>
Tutor:  José Antonio Neme Castillo  <antonio.neme@iimas.unam.mx>

Proyecto CIPA_FRAUD. Licencia: MIT — ver el archivo LICENSE.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

# --- Raíz de este proyecto -------------------------------------------------
#: Raíz del repositorio CIPA_FRAUD (dos niveles arriba de ``src/cipa_fraud``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Proyectos hermanos (fuentes externas, solo lectura) -------------------
#: Raíz del proyecto comparativo de fraude que provee los Parquet procesados.
FRAUD_COMPARATIVE_ROOT = Path(
    "/home/luisgarcia/projects/unam/dcic/2027-1/fraud_dataset_comparative"
)
#: Directorio con los datos procesados por dataset y capa (clean/features).
PROCESSED_ROOT = FRAUD_COMPARATIVE_ROOT / "data" / "processed"

#: Raíz del framework CIPA (instalado como editable; ruta usada para trazabilidad).
CIPA_ROOT = Path("/home/luisgarcia/projects/unam/dcic/2026-2/statistical_analysis")

# --- Salidas de este proyecto ----------------------------------------------
#: Directorio de resultados por corrida (JSON) y consolidado (Parquet).
RESULTS_DIR = PROJECT_ROOT / "results"
#: Directorio de reportes (informe comparativo, figuras, reproducibilidad).
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Reproducibilidad y escala ---------------------------------------------
#: Semilla global. Se propaga a CIPAPipeline y a todo submuestreo del proyecto.
RANDOM_STATE = 42

#: Capas de entrada disponibles en ``data/processed/<id>/<layer>/data.parquet``.
#: ``clean`` = columnas canónicas limpias; ``features`` = con ingeniería (S5).
LAYERS = ("clean", "features")

#: Barrido multi-N: tamaños de submuestreo del estudio de escala (E-SCALE).
#: ``"full"`` = N completo con submuestreo interno de CIPA para las dimensiones
#: costosas (D2/D3/D4/D7) y valores exactos en el resto.
SWEEP_N = (10_000, 50_000, "full")

#: ``knn_subsample`` de :class:`cipa.CIPAPipeline` para la corrida ``"full"``.
#: Equilibra fidelidad y tiempo (tabla de rendimiento de CIPA: N≤1M, d≤50 → ≤300 s).
KNN_SUBSAMPLE_FULL = 50_000

#: Umbral de faltantes del adaptador: las columnas con una fracción de valores
#: ausentes mayor que esto se descartan (la fuente es inmutable; la imputación
#: ocurre solo en memoria). Ver :mod:`cipa_fraud.adapt`.
MAX_MISSING_FRACTION = 0.50


def processed_path(dataset_id: str, layer: str) -> Path:
    """Devuelve la ruta al Parquet procesado de un dataset en la capa dada.

    CIPA_FRAUD solo trabaja datasets tabulares, cuya matriz vive siempre en
    ``data.parquet`` (los datasets de grafo, que usarían ``nodes.parquet``, están
    fuera del alcance del estudio).

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset en el registro (p. ej. ``"ulb_cc"``).
    layer : str
        Capa de entrada: ``"clean"`` o ``"features"`` (ver :data:`LAYERS`).

    Returns
    -------
    pathlib.Path
        Ruta absoluta a ``data/processed/<dataset_id>/<layer>/data.parquet`` en
        el proyecto comparativo. La existencia no se garantiza; validar aparte.
    """
    return PROCESSED_ROOT / dataset_id / layer / "data.parquet"


def roles_path(dataset_id: str, layer: str) -> Path:
    """Devuelve la ruta al ``_roles.json`` con el target y los roles de columna.

    Los roles canónicos (``target``, ``id_cols``, ``leakage_cols``,
    ``added_cols``, ``positive_label``) se definen en la capa ``clean`` y la capa
    ``features`` los hereda; por eso siempre se apunta a ``clean``,
    independientemente de ``layer``. Este archivo es la **fuente de verdad** del
    target: el adaptador lo lee en lugar de asumir nombres de columna.

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset en el registro.
    layer : str
        Capa solicitada. Se acepta por simetría de la API pero no altera la ruta
        (los roles viven en ``clean``).

    Returns
    -------
    pathlib.Path
        Ruta a ``data/processed/<dataset_id>/clean/_roles.json``.
    """
    return PROCESSED_ROOT / dataset_id / "clean" / "_roles.json"


def features_meta_path(dataset_id: str) -> Path:
    """Devuelve la ruta al ``_features.json`` con la descripción de features S5.

    Contiene el catálogo de features derivadas en la etapa de ingeniería del
    proyecto comparativo (nombre, tipo y definición). El adaptador lo consulta al
    trabajar la capa ``features``.

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset en el registro.

    Returns
    -------
    pathlib.Path
        Ruta a ``data/processed/<dataset_id>/features/_features.json``.
    """
    return PROCESSED_ROOT / dataset_id / "features" / "_features.json"


def check_paths() -> list[str]:
    """Valida la existencia de las rutas externas de las que depende el proyecto.

    Comprueba el proyecto comparativo, su directorio de datos procesados y la raíz
    del framework CIPA. No verifica los Parquet por dataset (de eso se encarga el
    comando ``doctor`` de la CLI).

    Returns
    -------
    list[str]
        Lista de mensajes de problema (una entrada por ruta faltante). Lista vacía
        si todas las rutas existen.
    """
    problems: list[str] = []
    for label, path in (
        ("FRAUD_COMPARATIVE_ROOT", FRAUD_COMPARATIVE_ROOT),
        ("PROCESSED_ROOT", PROCESSED_ROOT),
        ("CIPA_ROOT", CIPA_ROOT),
    ):
        if not path.exists():
            problems.append(f"{label} no existe: {path}")
    return problems
