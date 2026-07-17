"""Pruebas del registro de datasets (:mod:`cipa_fraud.registry`)."""

from __future__ import annotations

import pytest

from cipa_fraud import registry


def test_all_ids_son_ocho_unicos() -> None:
    """El estudio declara exactamente 8 datasets tabulares, sin duplicados."""
    ids = registry.all_ids()
    assert len(ids) == 8
    assert len(set(ids)) == 8


def test_benchmark_overlap_son_los_tres_conocidos() -> None:
    """El solapamiento con el benchmark son ulb_cc, paysim e ieee_cis."""
    overlap = registry.benchmark_overlap()
    assert set(overlap) == {"ulb_cc", "paysim", "ieee_cis"}
    assert overlap["ulb_cc"] == "CreditCard"


def test_get_desconocido_lanza_keyerror() -> None:
    """Pedir un id inexistente lanza ``KeyError`` con la lista disponible."""
    with pytest.raises(KeyError):
        registry.get("no_existe")


def test_fraud_rate_en_rango() -> None:
    """La tasa de fraude de cada dataset es una proporción en [0, 1]."""
    for ds_id in registry.all_ids():
        assert 0.0 <= registry.get(ds_id).fraud_rate <= 1.0
