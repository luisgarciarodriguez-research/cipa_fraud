"""Orquestación de CIPAPipeline por dataset × capa × N, con manifiesto. Fases F2/F3.

Puente entre el adaptador (:mod:`cipa_fraud.adapt`) y el framework CIPA. Para una
corrida concreta: construye ``(X, y)``, envuelve el par en un
:class:`cipa.CIPADataset`, ejecuta el :class:`cipa.CIPAPipeline` con la semilla y
los parámetros de escala del proyecto, y serializa el :class:`cipa.CIPAResult`
junto con el manifiesto del adaptado a ``results/<id>/<layer>/<N>.json``.

La implementación se aborda en dos fases: F2 (prueba de humo sobre los datasets
más pequeños) y F3 (ejecución completa: 8 datasets × 2 capas × 3 escalas). Este
módulo fija la firma estable que consumen la CLI y el consolidado de resultados.

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

from cipa_fraud import settings


def run_one(
    dataset_id: str,
    layer: str,
    n_target: int | str = "full",
    random_state: int = settings.RANDOM_STATE,
) -> dict:  # pragma: no cover - F2/F3
    """Ejecuta una corrida CIPA completa para un dataset/capa/escala.

    Flujo previsto: adaptar → ``CIPADataset.from_arrays`` → ``CIPAPipeline.run`` →
    serializar el resultado y el manifiesto. Persiste el JSON de la corrida y lo
    devuelve como diccionario.

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset en el registro (p. ej. ``"ulb_cc"``).
    layer : str
        Capa de entrada: ``"clean"`` o ``"features"``.
    n_target : int | str, optional
        Punto del barrido de escala: un entero (submuestreo asimétrico) o
        ``"full"`` (N completo con submuestreo interno de CIPA). Por defecto
        ``"full"``.
    random_state : int, optional
        Semilla propagada al adaptado y al pipeline. Por defecto
        :data:`cipa_fraud.settings.RANDOM_STATE`.

    Returns
    -------
    dict
        Resultado de CIPA (``CIPAResult.to_dict()``) fusionado con el manifiesto
        del adaptado: dimensiones D1–D7, DS, banda, firma, recomendaciones y
        metadatos de trazabilidad.

    Raises
    ------
    NotImplementedError
        Hasta que se implemente en las fases F2/F3 (ver ``PLAN.md``).
    """
    raise NotImplementedError("run_one() se implementa en F2/F3 (ver PLAN.md).")
