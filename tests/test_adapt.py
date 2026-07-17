"""Pruebas de contrato del adaptador ``(X, y)`` (:mod:`cipa_fraud.adapt`).

Verifican que el par construido cumpla el contrato de entrada de CIPA. Requieren los
Parquet procesados del proyecto comparativo; si faltan, se omiten (marca
``needs_data``). Se usa ``fdb`` (el dataset con menos registros) a N=10k para que la
adaptación sea rápida y no ejecute CIPA.
"""

from __future__ import annotations

import numpy as np
import pytest

from cipa_fraud import settings
from cipa_fraud.adapt import adapt

pytestmark = pytest.mark.needs_data

_HAS_FDB = settings.processed_path("fdb", "clean").exists()
_skip = pytest.mark.skipif(not _HAS_FDB, reason="faltan los Parquet de fdb/clean")


@pytest.fixture(scope="module")
def adapted():
    """Adapta ``fdb/clean`` a N=10k una vez para todas las pruebas de contrato."""
    return adapt("fdb", "clean", 10_000)


@_skip
def test_x_es_float64_finito_2d(adapted) -> None:
    """X es una matriz 2-D ``float64`` sin NaN/Inf."""
    X = adapted.X
    assert X.ndim == 2
    assert X.dtype == np.float64
    assert np.isfinite(X).all()


@_skip
def test_y_binaria_minoria_es_uno(adapted) -> None:
    """y es binaria entera con el fraude (=1) como minoría y n_minority ≥ 2."""
    y = adapted.y
    assert set(np.unique(y)) <= {0, 1}
    n_min = int((y == 1).sum())
    assert 2 <= n_min <= len(y) - n_min  # 1 es la clase minoritaria


@_skip
def test_dimensiones_y_contrato_minimo(adapted) -> None:
    """X e y están alineadas y cumplen N ≥ 10 (contrato de CIPA)."""
    assert adapted.X.shape[0] == adapted.y.shape[0]
    assert adapted.X.shape[0] >= 10


@_skip
def test_manifiesto_tiene_trazabilidad(adapted) -> None:
    """El manifiesto registra identificación, escala y desbalanceo."""
    m = adapted.manifest
    assert m["dataset_id"] == "fdb"
    assert m["layer"] == "clean"
    assert {"N", "d", "IR", "IR_eff", "subsample_mode", "n_minority"} <= set(m)
