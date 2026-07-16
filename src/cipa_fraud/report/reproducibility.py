"""Manifiesto global de reproducibilidad del estudio (``reproducibility.json``). Fase F6.

Reúne todo lo necesario para reproducir el estudio de forma trazable: versiones (de
CIPA y de las dependencias clave), la semilla global, los parámetros de escala y los
pesos del pipeline, el hash SHA-256 de cada Parquet de entrada (los datos no se
copian: se referencia su contenido por hash) y un resumen por corrida (N, d, IR,
IR_eff, runtime, versión de CIPA) tomado de los JSON de ``results/``.

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

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cipa_fraud import registry, settings

#: Dependencias cuyas versiones se registran en el manifiesto.
_TRACKED_DEPS = (
    "cipa", "numpy", "scipy", "scikit-learn", "polars", "pyarrow", "matplotlib",
    "pydantic", "typer",
)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    """SHA-256 y tamaño en bytes de un archivo, leído por bloques (memoria acotada).

    Parameters
    ----------
    path : pathlib.Path
        Archivo a hashear.
    chunk : int, optional
        Tamaño de bloque de lectura (por defecto 1 MiB).

    Returns
    -------
    tuple[str, int]
        ``(hexdigest, n_bytes)``.
    """
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
            n += len(data)
    return h.hexdigest(), n


def dep_versions() -> dict[str, str]:
    """Versiones instaladas de las dependencias rastreadas (más Python)."""
    out = {"python": platform.python_version(), "platform": sys.platform}
    for pkg in _TRACKED_DEPS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover - dep ausente
            out[pkg] = "n/d"
    return out


def input_hashes() -> list[dict]:
    """Hash SHA-256 de cada Parquet de entrada (8 datasets × 2 capas).

    Returns
    -------
    list[dict]
        Una entrada por Parquet existente: ``dataset_id``, ``layer``, ``path``,
        ``sha256``, ``bytes``. La fuente es inmutable; sólo se referencia por hash.
    """
    rows: list[dict] = []
    for ds_id in registry.all_ids():
        for layer in settings.LAYERS:
            path = settings.processed_path(ds_id, layer)
            if not path.exists():
                continue
            digest, nbytes = _sha256_file(path)
            rows.append({
                "dataset_id": ds_id,
                "layer": layer,
                "path": str(path),
                "sha256": digest,
                "bytes": nbytes,
            })
    return rows


def _run_summaries() -> list[dict]:
    """Resumen por corrida leído de los JSON de ``results/`` (trazabilidad)."""
    rows: list[dict] = []
    for ds_id in registry.all_ids():
        for layer in settings.LAYERS:
            layer_dir = settings.RESULTS_DIR / ds_id / layer
            if not layer_dir.exists():
                continue
            for jf in sorted(layer_dir.glob("*.json")):
                o = json.loads(jf.read_text())
                m = o["manifest"]
                rows.append({
                    "dataset_id": m["dataset_id"],
                    "layer": m["layer"],
                    "n_target": str(m["n_target"]),
                    "N": m["N"],
                    "d": m["d"],
                    "IR": m["IR"],
                    "IR_eff": m["IR_eff"],
                    "subsample_mode": m["subsample_mode"],
                    "runtime_s": o["runtime_s"],
                    "cipa_version": o.get("cipa_version"),
                })
    return rows


def build(write: bool = True, hash_inputs: bool = True) -> dict:
    """Ensambla el manifiesto global de reproducibilidad del estudio.

    Parameters
    ----------
    write : bool, optional
        Si ``True`` (por defecto) escribe ``reports/reproducibility.json``.
    hash_inputs : bool, optional
        Si ``True`` (por defecto) calcula el SHA-256 de los Parquet de entrada
        (~1.6 GB de lectura). Ponerlo en ``False`` acelera pruebas.

    Returns
    -------
    dict
        Manifiesto con versiones, semilla, parámetros de escala, pesos, hashes de
        entrada y resumen por corrida.
    """
    runs = _run_summaries()
    cipa_versions = sorted({r["cipa_version"] for r in runs if r["cipa_version"]})
    manifest = {
        "generated": datetime.now(UTC).isoformat(),
        "study": "CIPA_FRAUD",
        "random_state": settings.RANDOM_STATE,
        "cipa_versions": cipa_versions,
        "cipa_root": str(settings.CIPA_ROOT),
        "scale_params": {
            "sweep_n": [str(n) for n in settings.SWEEP_N],
            "knn_subsample_full": settings.KNN_SUBSAMPLE_FULL,
            "n1_max_exact": settings.N1_MAX_EXACT,
            "n1_subsample": settings.N1_SUBSAMPLE,
            "max_missing_fraction": settings.MAX_MISSING_FRACTION,
        },
        "dependencies": dep_versions(),
        "n_runs": len(runs),
        "runs": runs,
        "inputs": input_hashes() if hash_inputs else [],
    }
    if write:
        settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (settings.REPORTS_DIR / "reproducibility.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
    return manifest
