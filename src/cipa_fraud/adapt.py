"""Adaptador de datos: Parquet(clean|features) → (X, y) para CIPA.

Núcleo técnico del proyecto (ver ``PLAN.md``, sección 3). Transforma cada
``data.parquet`` procesado en el par ``(X, y)`` que exige el contrato de entrada
de CIPA, **sin mutar la fuente** (toda imputación/codificación ocurre en memoria).

Contrato de salida (validado por :class:`cipa.CIPADataset`):

- ``X`` de forma ``(N, d)``, ``float64``, **sin NaN ni Inf**.
- ``y`` de forma ``(N,)``, entera, con **exactamente dos valores**, fraude = 1
  (que es la clase minoritaria en los 8 datasets).
- ``N ≥ 10`` y ``n_minoría ≥ 2``.

Selección y codificación de columnas (difiere por capa, a propósito, para que el
contraste clean↔features del sub-estudio E-FEAT sea informativo):

- **clean:** ``X`` = ``feature_cols`` del ``_roles.json``. Las categóricas string
  se codifican a códigos ordinales (determinista por orden de aparición). Es la
  representación mínima numérica del dataset limpio, sin ingeniería.
- **features:** ``X`` = columnas numéricas de ``feature_cols`` + todas las features
  derivadas (``_features.json``: ``__log1p``/``__z``, ``__te``/``__freq``,
  temporales, ``freq_orig``/``freq_dest``). Las categóricas string crudas se
  **descartan** porque sus encodings numéricos (``__te``/``__freq``) ya están
  presentes. Es la representación de modelado, puramente numérica.

Faltantes (solo ``ieee_cis`` los tiene): se descartan las columnas con fracción de
nulos > :data:`cipa_fraud.settings.MAX_MISSING_FRACTION`; el resto se imputa con la
mediana. ``±Inf`` se trata como nulo antes de imputar.

Escala (barrido multi-N, ver :func:`_subsample_indices`): el fraude se mantiene
siempre como minoría; el modo (asimétrico vs. estratificado) se elige según quepa
o no toda la minoría en el tamaño objetivo, y se registra en el manifiesto junto
con ``IR`` (original) e ``IR_eff`` (tras submuestreo).

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

import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from cipa_fraud import settings


@dataclass
class AdaptResult:
    """Resultado del adaptado: la matriz ``(X, y)`` lista para CIPA y su manifiesto.

    Attributes
    ----------
    X : numpy.ndarray
        Matriz de características de forma ``(N, d)``, ``float64``, sin NaN ni Inf.
    y : numpy.ndarray
        Vector de etiquetas de forma ``(N,)``, entero, con el fraude (clase
        minoritaria) codificado como 1 y el resto como 0.
    feature_names : list[str]
        Nombres de las ``d`` columnas de ``X``, en el orden de las columnas.
    manifest : dict
        Metadatos trazables de la corrida (ver :func:`adapt`).
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    manifest: dict


# Conjunto de columnas que nunca son features, más allá de los roles declarados.
_NEVER_FEATURE = {"split", "is_duplicate", "is_labeled", "event_time"}


def _load_roles(dataset_id: str) -> dict:
    """Lee el ``_roles.json`` (capa clean) con target y roles de columna."""
    return json.loads(settings.roles_path(dataset_id, "clean").read_text())


def _load_new_features(dataset_id: str) -> list[str]:
    """Devuelve los nombres de las features derivadas (``_features.json``)."""
    meta = json.loads(settings.features_meta_path(dataset_id).read_text())
    return [f["name"] for f in (meta.get("new_features") or [])]


def _is_categorical(dtype: pl.DataType) -> bool:
    """Indica si un dtype de polars es texto/categórico (requiere codificación)."""
    return dtype in (pl.String, pl.Categorical, pl.Enum, pl.Object)


def _is_temporal(dtype: pl.DataType) -> bool:
    """Indica si un dtype de polars es fecha/hora (se convierte a época numérica)."""
    return isinstance(dtype, (pl.Datetime, pl.Date, pl.Time, pl.Duration))


def _feature_expr(name: str, dtype: pl.DataType) -> pl.Expr:
    """Construye la expresión que codifica una columna a ``Float64`` limpia.

    Aplica la conversión según el tipo (booleano → 0/1, categórico → código
    ordinal, temporal → época en segundos, numérico → tal cual) y marca los
    ``±Inf`` como nulos para que la imputación posterior los trate.

    Parameters
    ----------
    name : str
        Nombre de la columna de origen.
    dtype : polars.DataType
        Tipo de la columna en el esquema del Parquet.

    Returns
    -------
    polars.Expr
        Expresión ``Float64`` con alias ``name``, con ``±Inf`` convertidos a nulo.
    """
    col = pl.col(name)
    if dtype == pl.Boolean:
        expr = col.cast(pl.Int8)
    elif _is_categorical(dtype):
        # Código ordinal determinista (orden de aparición, estable por archivo).
        expr = col.cast(pl.Categorical).to_physical()
    elif _is_temporal(dtype):
        expr = col.dt.epoch(time_unit="s")
    else:
        expr = col
    expr = expr.cast(pl.Float64)
    # ±Inf → nulo (se imputará con la mediana como el resto de faltantes).
    expr = pl.when(expr.is_infinite()).then(None).otherwise(expr)
    return expr.alias(name)


def _select_columns(
    dataset_id: str, layer: str, schema: pl.Schema
) -> tuple[list[str], list[str]]:
    """Determina las columnas de ``X`` y las categóricas descartadas por capa.

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset.
    layer : str
        Capa de entrada: ``"clean"`` o ``"features"``.
    schema : polars.Schema
        Esquema del Parquet de la capa (nombres y dtypes disponibles).

    Returns
    -------
    tuple[list[str], list[str]]
        ``(feature_cols, dropped_categorical)`` — columnas seleccionadas para
        ``X`` y, en la capa ``features``, las categóricas crudas descartadas por
        estar ya representadas por sus encodings numéricos.
    """
    roles = _load_roles(dataset_id)
    base = list(roles.get("feature_cols", []))
    exclude = (
        {roles.get("target")}
        | set(roles.get("id_cols", []))
        | set(roles.get("leakage_cols", []))
        | set(roles.get("added_cols", []))
        | _NEVER_FEATURE
    )

    if layer == "features":
        base = base + _load_new_features(dataset_id)

    names = set(schema.names())
    selected: list[str] = []
    dropped_categorical: list[str] = []
    seen: set[str] = set()
    for name in base:
        if name in exclude or name in seen or name not in names:
            continue
        seen.add(name)
        dtype = schema[name]
        # En la capa features, las categóricas crudas se descartan (sus encodings
        # __te/__freq ya están entre las features derivadas).
        if layer == "features" and (_is_categorical(dtype) or _is_temporal(dtype)):
            dropped_categorical.append(name)
            continue
        selected.append(name)
    return selected, dropped_categorical


def _subsample_indices(
    y: np.ndarray, n_target: int, rng: np.random.Generator
) -> tuple[np.ndarray | None, str]:
    """Calcula los índices a conservar para llevar el dataset a ``n_target`` filas.

    Preserva siempre al fraude como clase minoritaria. Elige el modo según quepa o
    no toda la minoría en el objetivo:

    - **asimétrico** (``n_minoría ≤ n_target/2``): conserva toda la minoría y
      submuestrea la mayoría hasta ``n_target − n_minoría``. Preserva todos los
      positivos raros pero distorsiona la razón de desbalanceo (efecto Tier-2).
    - **estratificado** (en otro caso): submuestrea ambas clases a la proporción
      ``n_target / N``, preservando la razón de desbalanceo original.

    Parameters
    ----------
    y : numpy.ndarray
        Etiquetas (0/1) del dataset completo.
    n_target : int
        Número objetivo de filas.
    rng : numpy.random.Generator
        Generador aleatorio sembrado, para reproducibilidad.

    Returns
    -------
    tuple[numpy.ndarray | None, str]
        ``(indices, modo)``. ``indices`` es un arreglo ordenado de posiciones a
        conservar, o ``None`` si no hay submuestreo (modo ``"full"``).
    """
    n = y.shape[0]
    if n <= n_target:
        return None, "full"

    idx_min = np.flatnonzero(y == 1)
    idx_maj = np.flatnonzero(y == 0)
    m = idx_min.shape[0]

    if m <= n_target // 2:
        maj_keep = n_target - m
        sel_maj = rng.choice(idx_maj, size=maj_keep, replace=False)
        keep = np.concatenate([idx_min, sel_maj])
        mode = "asymmetric"
    else:
        frac = n_target / n
        k_min = max(2, round(m * frac))
        k_maj = max(1, n_target - k_min)
        sel_min = rng.choice(idx_min, size=min(k_min, m), replace=False)
        sel_maj = rng.choice(idx_maj, size=min(k_maj, idx_maj.shape[0]), replace=False)
        keep = np.concatenate([sel_min, sel_maj])
        mode = "stratified"

    keep.sort()
    return keep, mode


def _normalize_n_target(n_target: int | str) -> int | str:
    """Normaliza el punto de escala: ``"full"`` o un entero (acepta texto numérico)."""
    if isinstance(n_target, str):
        if n_target == "full":
            return "full"
        if n_target.isdigit():
            return int(n_target)
        raise ValueError(f"n_target inválido: {n_target!r} (usa un entero o 'full').")
    return n_target


def adapt(
    dataset_id: str,
    layer: str,
    n_target: int | str = "full",
    random_state: int = settings.RANDOM_STATE,
) -> AdaptResult:
    """Construye ``(X, y)`` desde el Parquet procesado de un dataset.

    Lee la capa indicada, selecciona y codifica las columnas de features, aplica la
    política de escala del barrido y la imputación de faltantes, y devuelve la
    matriz lista para CIPA junto con su manifiesto. No modifica los archivos de
    origen.

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset en el registro (p. ej. ``"ulb_cc"``).
    layer : str
        Capa de entrada: ``"clean"`` o ``"features"``.
    n_target : int | str, optional
        Punto del barrido de escala. Un entero aplica submuestreo (asimétrico o
        estratificado, ver :func:`_subsample_indices`); ``"full"`` usa N completo.
        Por defecto ``"full"``.
    random_state : int, optional
        Semilla del submuestreo. Por defecto
        :data:`cipa_fraud.settings.RANDOM_STATE`.

    Returns
    -------
    AdaptResult
        La matriz ``(X, y)``, los nombres de columnas y el manifiesto. El
        manifiesto incluye: ``dataset_id``, ``layer``, ``n_target``, ``N`` final,
        ``d``, ``n_minority``/``n_majority``, ``IR`` (original) e ``IR_eff``,
        ``subsample_mode``, columnas descartadas por ser categóricas
        (``dropped_categorical``) o por exceso de nulos (``dropped_high_null``),
        columnas imputadas (``imputed``), ``random_state`` y la ruta de origen.

    Raises
    ------
    FileNotFoundError
        Si no existe el Parquet de la capa indicada.
    ValueError
        Si el resultado no cumple el contrato de CIPA (menos de dos clases o
        valores no finitos remanentes).
    """
    layer = layer.lower()
    if layer not in settings.LAYERS:
        raise ValueError(f"Capa inválida: {layer!r}. Usa una de {settings.LAYERS}.")
    n_target = _normalize_n_target(n_target)

    path = settings.processed_path(dataset_id, layer)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Parquet: {path}")

    roles = _load_roles(dataset_id)
    target = roles["target"]
    positive = int(roles.get("positive_label", "1"))

    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    feature_cols, dropped_categorical = _select_columns(dataset_id, layer, schema)
    if not feature_cols:
        raise ValueError(f"{dataset_id}/{layer}: no se seleccionó ninguna feature.")

    # Etiquetas del dataset completo (barato: una sola columna).
    y_full = (
        lf.select(pl.col(target)).collect().to_series() == positive
    ).cast(pl.Int8).to_numpy()
    n_full = y_full.shape[0]
    n_min_full = int((y_full == 1).sum())
    n_maj_full = n_full - n_min_full
    ir_full = (n_maj_full / n_min_full) if n_min_full else float("inf")

    # Índices a conservar según la política de escala.
    rng = np.random.default_rng(random_state)
    keep, mode = (None, "full")
    if n_target != "full":
        keep, mode = _subsample_indices(y_full, int(n_target), rng)

    # Materializa solo las columnas de features codificadas (y solo las filas
    # conservadas, para acotar memoria en los datasets grandes).
    enc = lf.select([_feature_expr(c, schema[c]) for c in feature_cols])
    if keep is not None:
        enc = (
            enc.with_row_index("__ridx")
            .filter(pl.col("__ridx").is_in(pl.Series(keep, dtype=pl.UInt32)))
            .drop("__ridx")
        )
    xdf = enc.collect()
    y = y_full if keep is None else y_full[keep]

    # Descarta columnas con exceso de nulos; imputa el resto con la mediana.
    n_rows = xdf.height
    null_counts = xdf.null_count().row(0)
    cols = xdf.columns
    dropped_high_null: list[str] = []
    imputed: list[str] = []
    for name, nulls in zip(cols, null_counts, strict=True):
        frac = nulls / n_rows if n_rows else 0.0
        if frac > settings.MAX_MISSING_FRACTION:
            dropped_high_null.append(name)
        elif nulls > 0:
            imputed.append(name)
    if dropped_high_null:
        xdf = xdf.drop(dropped_high_null)
    if imputed:
        xdf = xdf.with_columns(
            [pl.col(c).fill_null(pl.col(c).median()) for c in imputed]
        )

    final_features = xdf.columns
    X = xdf.to_numpy().astype(np.float64, copy=False)

    # Verificación del contrato de CIPA.
    if not np.isfinite(X).all():
        raise ValueError(f"{dataset_id}/{layer}: X contiene valores no finitos.")
    if np.unique(y).shape[0] != 2:
        raise ValueError(f"{dataset_id}/{layer}: y no tiene exactamente 2 clases.")

    n_min = int((y == 1).sum())
    n_maj = int((y == 0).sum())
    ir_eff = (n_maj / n_min) if n_min else float("inf")

    manifest = {
        "dataset_id": dataset_id,
        "layer": layer,
        "n_target": n_target,
        "subsample_mode": mode,
        "source_path": str(path),
        "source_rows": n_full,
        "N": int(X.shape[0]),
        "d": int(X.shape[1]),
        "n_minority": n_min,
        "n_majority": n_maj,
        "fraud_rate_eff": (n_min / X.shape[0]) if X.shape[0] else 0.0,
        "IR": ir_full,
        "IR_eff": ir_eff,
        "target": target,
        "positive_label": positive,
        "n_features": len(final_features),
        "dropped_categorical": dropped_categorical,
        "dropped_high_null": dropped_high_null,
        "imputed": imputed,
        "random_state": random_state,
    }
    return AdaptResult(X=X, y=y, feature_names=final_features, manifest=manifest)
