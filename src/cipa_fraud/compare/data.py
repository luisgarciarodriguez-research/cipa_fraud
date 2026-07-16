"""Cargadores y vistas de datos para el análisis comparativo (RQ1–RQ7). Fase F5.

Reúne en un solo lugar las tres fuentes que consume :mod:`cipa_fraud.compare`:

1. el consolidado tidy de CIPA (``results/all_results.parquet``, una fila por
   corrida dataset × capa × N), producido por F3;
2. los metadatos descriptivos del registro (dominio, origen real/sintético) para
   los cortes por grupo (RQ4);
3. el *proxy* de dificultad heurístico del proyecto comparativo
   (``reports/comparative/comparison.csv``) para su validación contra el DS (RQ7).

Las dimensiones de grafo (``elliptic``, ``elliptic_pp``, ``amlworld``) quedan fuera:
se filtra siempre a los 8 datasets tabulares del registro.

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

import polars as pl

from cipa_fraud import registry, settings
from cipa_fraud.run import all_results_path

#: Las siete dimensiones CIPA, en orden.
DIMENSIONS = ("D1", "D2", "D3", "D4", "D5", "D6", "D7")

#: Nombres legibles de las dimensiones (para ejes de figuras y tablas).
DIMENSION_NAMES = {
    "D1": "Desbalance",
    "D2": "Solapamiento",
    "D3": "Dureza",
    "D4": "Fragmentación",
    "D5": "Dimensionalidad",
    "D6": "Informatividad",
    "D7": "Frontera",
}

#: Capa y escala canónicas para los cortes transversales (representación de
#: modelado a N completo). Los estudios E-FEAT/E-SCALE varían estos ejes aparte.
CANONICAL_LAYER = "features"
CANONICAL_N = "full"

#: Ruta al proxy de dificultad del proyecto comparativo (fuente de RQ7).
PROXY_CSV = (
    settings.FRAUD_COMPARATIVE_ROOT
    / "reports" / "comparative" / "comparison.csv"
)


def load_results() -> pl.DataFrame:
    """Carga el consolidado tidy del estudio (``results/all_results.parquet``).

    Returns
    -------
    polars.DataFrame
        Una fila por corrida (dataset × capa × N) con escala, DS, banda, firma y
        D1–D7. ``n_target`` es texto (``"10000"``/``"50000"``/``"full"``).

    Raises
    ------
    FileNotFoundError
        Si el consolidado no existe (correr ``cipa-fraud run-all`` antes).
    """
    path = all_results_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta 'cipa-fraud run-all' (F3) primero."
        )
    return pl.read_parquet(path)


def load_metadata() -> pl.DataFrame:
    """Metadatos descriptivos de los 8 datasets (dominio, origen, tasa de fraude).

    Returns
    -------
    polars.DataFrame
        Columnas ``dataset_id``, ``name``, ``domain``, ``origin``, ``fraud_rate``,
        ``cipa_benchmark_key``.
    """
    rows = []
    for ds_id in registry.all_ids():
        s = registry.get(ds_id)
        rows.append({
            "dataset_id": s.id,
            "name": s.name,
            "domain": s.domain.value,
            "origin": s.origin.value,
            "fraud_rate": s.fraud_rate,
            "cipa_benchmark_key": s.cipa_benchmark_key,
        })
    return pl.DataFrame(rows)


def load_proxy() -> pl.DataFrame:
    """Carga el *proxy* de dificultad del proyecto comparativo (RQ7).

    Filtra a los 8 datasets tabulares del registro (descarta los de grafo).

    Returns
    -------
    polars.DataFrame
        Columnas ``dataset_id`` y ``proxy`` (índice heurístico 0–1, mayor = más
        difícil). Vacío si el CSV de origen no está disponible.
    """
    if not PROXY_CSV.exists():
        return pl.DataFrame({"dataset_id": [], "proxy": []})
    ids = set(registry.all_ids())
    df = pl.read_csv(PROXY_CSV)
    return (
        df.select(
            pl.col("id").alias("dataset_id"),
            pl.col("dificultad").alias("proxy"),
        )
        .filter(pl.col("dataset_id").is_in(ids))
    )


def canonical_view(
    results: pl.DataFrame | None = None,
    layer: str = CANONICAL_LAYER,
    n_target: str = CANONICAL_N,
) -> pl.DataFrame:
    """Vista transversal: una fila por dataset a capa/escala fijas, con metadatos.

    Parameters
    ----------
    results : polars.DataFrame, optional
        Consolidado ya cargado; si es ``None`` se lee de disco.
    layer : str, optional
        Capa a fijar (por defecto :data:`CANONICAL_LAYER`).
    n_target : str, optional
        Escala a fijar, como texto (por defecto :data:`CANONICAL_N`).

    Returns
    -------
    polars.DataFrame
        Filas de la corrida (dataset) seleccionada, unidas a los metadatos del
        registro, ordenadas por ``DS`` descendente.
    """
    res = results if results is not None else load_results()
    view = res.filter(
        (pl.col("layer") == layer) & (pl.col("n_target") == n_target)
    )
    return view.join(load_metadata(), on="dataset_id", how="left").sort(
        "DS", descending=True
    )
