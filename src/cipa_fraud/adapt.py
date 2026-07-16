"""Adaptador de datos: Parquet(clean|features) → (X, y) para CIPA.

Núcleo técnico del proyecto (ver ``PLAN.md``, sección 3). Transforma cada
``data.parquet`` procesado en el par ``(X, y)`` que exige el contrato de entrada
de CIPA, **sin mutar la fuente** (toda imputación/codificación ocurre en memoria).

Contrato de salida (validado por :class:`cipa.CIPADataset`):

- ``X`` de forma ``(N, d)``, ``float64``, **sin NaN ni Inf**.
- ``y`` de forma ``(N,)``, entera, con **exactamente dos valores**, minoría = 1.
- ``N ≥ 10`` y ``n_minoría ≥ 2``.

Pasos del adaptado (implementación en la fase F1):

1. Leer el Parquet de la capa indicada (``clean`` o ``features``).
2. Derivar ``y`` desde el ``target`` del ``_roles.json`` (fuente de verdad),
   con la clase fraude codificada a 1.
3. Seleccionar las columnas de ``X`` (``feature_cols``), excluyendo target,
   identificadores, columnas de fuga y columnas añadidas (``split``,
   ``is_duplicate``, ``*__outlier``, …).
4. Codificar categóricas a códigos ordinales ``float64``.
5. Descartar columnas con exceso de faltantes (:data:`settings.MAX_MISSING_FRACTION`)
   e imputar el resto con la mediana; ``±Inf → NaN → imputa``.
6. Aplicar la política de escala del barrido (submuestreo asimétrico o full-N).
7. Emitir el manifiesto de la corrida.

Este módulo expone la firma estable que consume :mod:`cipa_fraud.run`.

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

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptResult:
    """Resultado del adaptado: la matriz ``(X, y)`` lista para CIPA y su manifiesto.

    Attributes
    ----------
    X : numpy.ndarray
        Matriz de características de forma ``(N, d)``, ``float64``, sin NaN ni Inf.
    y : numpy.ndarray
        Vector de etiquetas de forma ``(N,)``, entero, con la clase minoritaria
        (fraude) codificada como 1 y la mayoritaria como 0.
    manifest : dict
        Metadatos trazables de la corrida del adaptado: ``N``, ``d``, razón de
        desbalanceo original (``IR``) y efectiva tras submuestreo (``IR_eff``),
        fracción imputada, columnas usadas y descartadas, tamaño objetivo del
        submuestreo y semilla. Se serializa junto al resultado de CIPA.
    """

    X: np.ndarray  # (N, d) float64, sin NaN/Inf
    y: np.ndarray  # (N,) int, minoria = 1
    manifest: dict  # N, d, IR, IR_eff, %imputado, columnas usadas/descartadas, ...


def adapt(
    dataset_id: str,
    layer: str,
    n_target: int | str = "full",
    random_state: int = 42,
) -> AdaptResult:  # pragma: no cover - F1
    """Construye ``(X, y)`` desde el Parquet procesado de un dataset.

    Lee la capa indicada, aplica la selección/codificación/imputación de columnas
    y la política de escala, y devuelve la matriz lista para CIPA junto con su
    manifiesto. No modifica los archivos de origen.

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset en el registro (p. ej. ``"ulb_cc"``).
    layer : str
        Capa de entrada: ``"clean"`` o ``"features"`` (ver
        :data:`cipa_fraud.settings.LAYERS`).
    n_target : int | str, optional
        Punto del barrido de escala. Un entero aplica submuestreo asimétrico a ese
        tamaño (preservando toda la clase minoritaria); ``"full"`` usa N completo
        delegando el submuestreo de las dimensiones costosas a CIPA. Por defecto
        ``"full"``.
    random_state : int, optional
        Semilla para el submuestreo, por reproducibilidad. Por defecto ``42``.

    Returns
    -------
    AdaptResult
        La matriz ``(X, y)`` y el manifiesto de la corrida.

    Raises
    ------
    NotImplementedError
        Hasta que se implemente en la fase F1 (ver ``PLAN.md``).
    """
    raise NotImplementedError("adapt() se implementa en la fase F1 (ver PLAN.md).")
