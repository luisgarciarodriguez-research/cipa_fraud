"""Fixtures y utilidades compartidas por la suite de CIPA_FRAUD.

Provee datos sintéticos deterministas (un consolidado tidy completo y salidas de
corrida ``out`` mínimas) que permiten ejercitar la orquestación, la consistencia y
el análisis comparativo sin leer los Parquet de origen ni ejecutar CIPA, además del
detector de disponibilidad de datos que habilita las pruebas de contrato.
"""

from __future__ import annotations

import polars as pl
import pytest

from cipa_fraud import registry, settings
from cipa_fraud.compare.data import DIMENSIONS

#: Bandas de dificultad en orden ascendente (para asignaciones deterministas).
_BANDS = ("Low", "Moderate", "High", "Extreme")


def make_out(
    dataset_id: str,
    layer: str,
    n_target: str,
    ds: float = 0.5,
    band: str = "Moderate",
    signature: str = "V",
) -> dict:
    """Construye un ``out`` mínimo con la forma que producen las corridas reales.

    Reproduce solo las claves que consumen :func:`cipa_fraud.run.summary_row`,
    :func:`cipa_fraud.run.consolidate` y el manifiesto de reproducibilidad, con las
    siete dimensiones fijadas a un gradiente determinista.

    Parameters
    ----------
    dataset_id, layer : str
        Identificación de la corrida.
    n_target : str
        Punto de escala como texto (``"10000"``/``"50000"``/``"full"``).
    ds : float, optional
        Valor de DS a incrustar. Por defecto ``0.5``.
    band : str, optional
        Banda de dificultad. Por defecto ``"Moderate"``.
    signature : str, optional
        Firma del perfil. Por defecto ``"V"``.

    Returns
    -------
    dict
        Estructura ``out`` serializable equivalente a la de ``run_one``.
    """
    dims = [
        {"dimension_id": d, "value": round(0.1 + 0.1 * i, 4)}
        for i, d in enumerate(DIMENSIONS)
    ]
    return {
        "cipa": {
            "dataset_name": f"{dataset_id}/{layer}/n={n_target}",
            "difficulty_score": {"value": ds, "band": band, "dimensions": dims},
            "profile": {"signature": signature, "signature_name": f"{signature}-nombre"},
            "action": {
                "evaluation_metrics": ["AUC-PR"],
                "preprocessing_strategy": ["SMOTE"],
                "model_families": ["Random Forest"],
            },
        },
        "manifest": {
            "dataset_id": dataset_id,
            "layer": layer,
            "n_target": n_target,
            "subsample_mode": "full" if n_target == "full" else "stratified",
            "N": 10_000,
            "d": 12,
            "IR": 50.0,
            "IR_eff": 45.0,
        },
        "pipeline": {"weights": [0.1] * 7, "knn_subsample": None, "random_state": 42},
        "runtime_s": 1.23,
        "cipa_version": "test",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


@pytest.fixture
def synthetic_results() -> pl.DataFrame:
    """Consolidado tidy sintético: 8 datasets × 2 capas × 3 escalas (48 filas).

    Los valores de DS y de las dimensiones son deterministas y varían por dataset,
    capa y escala, de modo que las pruebas de análisis puedan re-derivar los
    resultados esperados desde el mismo DataFrame (sin números mágicos). En cada
    dataset la capa ``features`` tiene un DS ligeramente menor que ``clean`` (para
    ejercitar E-FEAT) y el DS decrece con N (para E-SCALE).

    Returns
    -------
    polars.DataFrame
        Columnas equivalentes a las de :func:`cipa_fraud.run.summary_row`.
    """
    ids = registry.all_ids()
    n_labels = ("10000", "50000", "full")
    rows: list[dict] = []
    for di, ds_id in enumerate(ids):
        base = 0.20 + 0.05 * di  # DS base creciente por dataset
        for layer in settings.LAYERS:
            layer_off = -0.03 if layer == "features" else 0.0  # features más fácil
            for ni, n in enumerate(n_labels):
                ds = round(base + layer_off + 0.02 * (2 - ni), 4)  # decrece con N
                row = {
                    "dataset_id": ds_id,
                    "layer": layer,
                    "n_target": n,
                    "subsample_mode": "full" if n == "full" else "stratified",
                    "N": 10_000 * (ni + 1),
                    "d": 12,
                    "IR": 50.0 + di,
                    "IR_eff": 45.0 + di,
                    "DS": ds,
                    "band": _BANDS[min(int(ds * 4), 3)],
                    "signature": "V" if di else "III",
                    "signature_name": "Compound" if di else "Hardness",
                }
                for k, dim in enumerate(DIMENSIONS):
                    row[dim] = round(0.1 + 0.08 * k + 0.01 * di, 4)
                rows.append(row)
    return pl.DataFrame(rows)


@pytest.fixture
def data_available() -> bool:
    """``True`` si los Parquet del proyecto comparativo están disponibles.

    Habilita las pruebas de contrato del adaptador; si es ``False`` se omiten.
    """
    return not settings.check_paths()
