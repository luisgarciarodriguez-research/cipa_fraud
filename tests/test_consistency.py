"""Pruebas de la consistencia con el benchmark (:mod:`cipa_fraud.consistency`).

La comparación se ejercita de forma hermética sustituyendo ``_recomputed_row`` por
filas sintéticas controladas, para verificar el cálculo de ``ΔDS``, la coincidencia
de banda/firma, la tolerancia y los conteos del resumen sin correr CIPA.
"""

from __future__ import annotations

import pytest

from cipa_fraud import consistency, registry


def test_referencia_tiene_los_tres_datasets() -> None:
    """La tabla de referencia cubre las 3 claves del benchmark con sus campos."""
    assert set(consistency.BENCHMARK_REFERENCE) == {"CreditCard", "PaySim", "IEEE-CIS"}
    for ref in consistency.BENCHMARK_REFERENCE.values():
        assert {"DS", "band", "signature"} <= set(ref)


def _mock_recomputed(delta: float, monkeypatch) -> None:
    """Simula ``_recomputed_row`` devolviendo banda/firma de referencia y DS+delta."""
    def _row(dataset_id, layer, random_state):
        key = registry.get(dataset_id).cipa_benchmark_key
        ref = consistency.BENCHMARK_REFERENCE[key]
        return {
            "DS": ref["DS"] + delta,
            "band": ref["band"],
            "signature": ref["signature"],
            "cipa_version": "test",
        }
    monkeypatch.setattr(consistency, "_recomputed_row", _row)


def test_check_reproduccion_exacta(monkeypatch) -> None:
    """Con DS = referencia: banda/firma 6/6 y DS dentro de tolerancia 6/6."""
    _mock_recomputed(0.0, monkeypatch)
    report = consistency.check(write=False)
    s = report["summary"]
    assert s["n_comparisons"] == 6  # 3 datasets × 2 capas
    assert s["band_match"] == 6
    assert s["signature_match"] == 6
    assert s["ds_within_tol"] == 6
    assert s["max_abs_delta_DS"] == 0.0


def test_check_ds_fuera_de_tolerancia(monkeypatch) -> None:
    """Un ΔDS de 0.2 (> tol 0.10) coincide en banda/firma pero no en DS."""
    _mock_recomputed(0.2, monkeypatch)
    report = consistency.check(write=False)
    s = report["summary"]
    assert s["band_match"] == 6  # banda/firma intactas
    assert s["ds_within_tol"] == 0
    assert report["comparisons"][0]["within_tol"] is False
    assert report["comparisons"][0]["delta_DS"] == pytest.approx(0.2, abs=1e-6)


def test_check_delta_signo(monkeypatch) -> None:
    """``ΔDS`` = recomputado − referencia (negativo si el recómputo es menor)."""
    _mock_recomputed(-0.05, monkeypatch)
    report = consistency.check(write=False)
    assert all(c["delta_DS"] < 0 for c in report["comparisons"])
