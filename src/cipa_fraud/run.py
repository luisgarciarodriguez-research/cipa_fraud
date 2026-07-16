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
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from cipa_fraud import adapt as _adapt_mod
from cipa_fraud import registry, settings


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
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if write:
        path = output_path(dataset_id, layer, n_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


#: Ruta del consolidado tidy del estudio (una fila por corrida).
def all_results_path() -> Path:
    """Ruta del consolidado tidy: ``results/all_results.parquet``."""
    return settings.RESULTS_DIR / "all_results.parquet"


def _iter_grid(
    datasets: Iterable[str],
    layers: Iterable[str],
    sweep: Iterable[int | str],
) -> Iterator[tuple[str, str, int | str]]:
    """Genera las tuplas ``(dataset, layer, n_target)`` del barrido, en orden."""
    for dataset_id in datasets:
        for layer in layers:
            for n_target in sweep:
                yield dataset_id, layer, n_target


def consolidate(write: bool = True):
    """Escanea los JSON de ``results/`` y arma el consolidado tidy del estudio.

    Recorre ``results/<id>/<layer>/<N>.json`` para todos los datasets del registro
    y ambas capas, extrae una fila plana por corrida con :func:`summary_row` y las
    apila en un ``polars.DataFrame`` (una fila por corrida). El disco es la única
    fuente de verdad: consolida cualquier corrida ya persistida, con independencia
    de qué invocación la produjo.

    Parameters
    ----------
    write : bool, optional
        Si ``True`` (por defecto) escribe :func:`all_results_path`.

    Returns
    -------
    polars.DataFrame
        Consolidado tidy (columnas de :func:`summary_row`; ``n_target`` normalizado
        a texto para que ``"full"`` y los enteros convivan en una sola columna).
        Vacío si aún no hay ninguna corrida en disco.
    """
    import polars as pl

    rows: list[dict] = []
    for dataset_id in registry.all_ids():
        for layer in settings.LAYERS:
            layer_dir = settings.RESULTS_DIR / dataset_id / layer
            if not layer_dir.exists():
                continue
            for json_file in sorted(layer_dir.glob("*.json")):
                out = json.loads(json_file.read_text())
                row = summary_row(out)
                row["n_target"] = str(row["n_target"])  # "full" ∪ enteros → texto
                rows.append(row)

    df = pl.DataFrame(rows)
    if write:
        settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(all_results_path())
    return df


def run_all(
    datasets: Iterable[str] | None = None,
    layers: Iterable[str] | None = None,
    sweep: Iterable[int | str] | None = None,
    random_state: int = settings.RANDOM_STATE,
    force: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Ejecuta el barrido completo del estudio y consolida los resultados. (F3)

    Corre CIPA sobre el producto ``datasets × capas × escalas`` (por defecto los 8
    datasets del registro × ``clean``/``features`` × :data:`settings.SWEEP_N`).
    Es **reanudable**: una corrida cuyo JSON ya existe se omite (salvo ``force``),
    de modo que un barrido interrumpido se retoma sin recomputar. Cada corrida se
    aísla: si una falla, se registra como *desviación* y el barrido continúa. Al
    final reconstruye ``results/all_results.parquet`` desde el disco.

    Parameters
    ----------
    datasets : iterable of str, optional
        Identificadores a correr. Por defecto todos los del registro.
    layers : iterable of str, optional
        Capas a correr. Por defecto :data:`settings.LAYERS`.
    sweep : iterable of int | str, optional
        Puntos de escala. Por defecto :data:`settings.SWEEP_N`.
    random_state : int, optional
        Semilla propagada a cada corrida. Por defecto
        :data:`settings.RANDOM_STATE`.
    force : bool, optional
        Si ``True`` recomputa aunque el JSON ya exista. Por defecto ``False``.
    progress : callable, optional
        Se invoca con un dict por corrida (claves: ``dataset_id``, ``layer``,
        ``n_target``, ``status`` ∈ {``ok``, ``skip``, ``fail``}, y ``row`` o
        ``error``). Permite a la CLI mostrar avance en vivo sin acoplar E/S aquí.

    Returns
    -------
    dict
        Resumen del barrido: ``n_total``, ``n_ok``, ``n_skip``, ``n_fail``,
        ``deviations`` (lista de dicts con la corrida y el error) y
        ``consolidated_path`` (ruta del parquet, o ``None`` si no hubo filas).
    """
    datasets = list(datasets) if datasets is not None else registry.all_ids()
    layers = list(layers) if layers is not None else list(settings.LAYERS)
    sweep = list(sweep) if sweep is not None else list(settings.SWEEP_N)

    deviations: list[dict] = []
    counts = {"ok": 0, "skip": 0, "fail": 0}

    for dataset_id, layer, n_target in _iter_grid(datasets, layers, sweep):
        rec: dict = {
            "dataset_id": dataset_id,
            "layer": layer,
            "n_target": _n_label(n_target),
        }
        path = output_path(dataset_id, layer, n_target)
        try:
            if path.exists() and not force:
                out = json.loads(path.read_text())
                rec["status"] = "skip"
            else:
                out = run_one(dataset_id, layer, n_target, random_state)
                rec["status"] = "ok"
            rec["row"] = summary_row(out)
        except Exception as exc:
            rec["status"] = "fail"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            deviations.append(rec)
        counts[rec["status"]] += 1
        if progress is not None:
            progress(rec)

    n_total = counts["ok"] + counts["skip"] + counts["fail"]
    df = consolidate(write=True)
    return {
        "n_total": n_total,
        "n_ok": counts["ok"],
        "n_skip": counts["skip"],
        "n_fail": counts["fail"],
        "deviations": deviations,
        "consolidated_path": str(all_results_path()) if len(df) else None,
    }
