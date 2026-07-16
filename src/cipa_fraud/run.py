"""Orquestación de CIPAPipeline por dataset × capa × N, con manifiesto. Fases F2/F3.

Puente entre el adaptador (:mod:`cipa_fraud.adapt`) y el framework CIPA. Para una
corrida concreta: construye ``(X, y)``, lo envuelve en un
:class:`cipa.CIPADataset` (con el fraude fijado como clase minoritaria = 1),
ejecuta el :class:`cipa.CIPAPipeline` con la semilla y los parámetros de escala del
proyecto, y serializa el :class:`cipa.CIPAResult` junto con el manifiesto del
adaptado a ``results/<id>/<layer>/<N>.json``.

En la corrida ``"full"`` se activa el submuestreo interno de CIPA
(:data:`cipa_fraud.settings.KNN_SUBSAMPLE_FULL`) para las dimensiones costosas
(D2/D3/D4/D7), manteniendo exactas las baratas (D1/D5/D6); en los puntos de
submuestreo explícito (10k/50k) el pipeline corre exacto sobre toda la submuestra.

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
import time
from datetime import datetime, timezone
from pathlib import Path

from cipa_fraud import adapt as _adapt_mod
from cipa_fraud import settings


def _n_label(n_target: int | str) -> str:
    """Etiqueta de archivo para un punto de escala (``"full"`` o el entero)."""
    return "full" if n_target == "full" else str(int(n_target))


def output_path(dataset_id: str, layer: str, n_target: int | str) -> Path:
    """Ruta del JSON de resultado de una corrida: ``results/<id>/<layer>/<N>.json``."""
    return settings.RESULTS_DIR / dataset_id / layer / f"{_n_label(n_target)}.json"


def summary_row(out: dict) -> dict:
    """Extrae una fila plana (tidy) de una corrida para el consolidado comparativo.

    Parameters
    ----------
    out : dict
        Diccionario de salida producido por :func:`run_one`.

    Returns
    -------
    dict
        Fila con identificación (``dataset_id``, ``layer``, ``n_target``), escala
        (``N``, ``d``, ``IR``, ``IR_eff``, ``subsample_mode``), el resultado CIPA
        (``DS``, ``band``, ``signature``, ``signature_name`` y ``D1``…``D7``) y el
        tiempo de cómputo (``runtime_s``).
    """
    m = out["manifest"]
    ds = out["cipa"]["difficulty_score"]
    prof = out["cipa"]["profile"]
    row = {
        "dataset_id": m["dataset_id"],
        "layer": m["layer"],
        "n_target": m["n_target"],
        "subsample_mode": m["subsample_mode"],
        "N": m["N"],
        "d": m["d"],
        "IR": m["IR"],
        "IR_eff": m["IR_eff"],
        "DS": ds["value"],
        "band": ds["band"],
        "signature": prof["signature"],
        "signature_name": prof["signature_name"],
        "runtime_s": out["runtime_s"],
    }
    for dim in ds["dimensions"]:
        row[dim["dimension_id"]] = dim["value"]
    return row


def run_one(
    dataset_id: str,
    layer: str,
    n_target: int | str = "full",
    random_state: int = settings.RANDOM_STATE,
    write: bool = True,
) -> dict:
    """Ejecuta una corrida CIPA completa para un dataset/capa/escala.

    Flujo: adaptar → :class:`cipa.CIPADataset` → :meth:`cipa.CIPAPipeline.run` →
    serializar el resultado y el manifiesto. Opcionalmente persiste el JSON.

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset en el registro (p. ej. ``"ulb_cc"``).
    layer : str
        Capa de entrada: ``"clean"`` o ``"features"``.
    n_target : int | str, optional
        Punto del barrido de escala: un entero (submuestreo) o ``"full"``
        (N completo con submuestreo interno de CIPA). Por defecto ``"full"``.
    random_state : int, optional
        Semilla propagada al adaptado y al pipeline. Por defecto
        :data:`cipa_fraud.settings.RANDOM_STATE`.
    write : bool, optional
        Si ``True`` (por defecto) escribe el JSON en :func:`output_path`.

    Returns
    -------
    dict
        Salida serializable con claves: ``cipa`` (``CIPAResult.to_dict()``),
        ``manifest`` (del adaptado), ``pipeline`` (parámetros usados),
        ``runtime_s``, ``cipa_version`` y ``timestamp``.
    """
    import cipa
    from cipa import CIPADataset, CIPAPipeline

    res = _adapt_mod.adapt(dataset_id, layer, n_target, random_state)

    # El fraude es la clase 1 y siempre la minoría en el adaptador.
    dataset = CIPADataset(
        res.X,
        res.y,
        minority_label=1,
        majority_label=0,
        name=f"{dataset_id}/{layer}/n={_n_label(n_target)}",
    )

    # Submuestreo interno solo en la corrida full (para las dimensiones costosas).
    knn = settings.KNN_SUBSAMPLE_FULL if n_target == "full" else None
    # N1 (D2) es el único punto O(N²) en memoria densa: se acota su umbral exacto
    # por debajo del barrido para forzar su submuestreo y evitar el OOM del SO.
    pipeline = CIPAPipeline(
        random_state=random_state,
        knn_subsample=knn,
        n1_max_exact=settings.N1_MAX_EXACT,
        large_n_subsample=settings.N1_SUBSAMPLE,
    )

    t0 = time.perf_counter()
    result = pipeline.run(dataset)
    runtime_s = round(time.perf_counter() - t0, 3)

    out = {
        "cipa": result.to_dict(),
        "manifest": res.manifest,
        "pipeline": {
            "weights": list(pipeline._weights),
            "knn_subsample": knn,
            "random_state": random_state,
        },
        "runtime_s": runtime_s,
        "cipa_version": cipa.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if write:
        path = output_path(dataset_id, layer, n_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out
