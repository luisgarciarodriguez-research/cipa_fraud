"""Pruebas del análisis comparativo (:mod:`cipa_fraud.compare.analysis`).

Se alimenta cada RQ con el consolidado sintético del fixture, de modo que los
resultados esperados se re-derivan del mismo DataFrame (sin números mágicos) y sin
tocar disco ni datos externos.
"""

from __future__ import annotations

import polars as pl
import pytest

from cipa_fraud.compare import analysis
from cipa_fraud.compare import data as _cdata
from cipa_fraud.compare.data import DIMENSIONS


def test_rq1_ranking_ordenado_desc(synthetic_results) -> None:
    """El ranking va por DS descendente y expone banda y rango."""
    rep = analysis.rq1_ranking(synthetic_results)
    ds = rep["table"]["DS"].to_list()
    assert ds == sorted(ds, reverse=True)
    assert rep["stats"]["DS_max"] == max(ds)
    assert rep["stats"]["hardest"] == rep["table"]["dataset_id"][0]


def test_rq2_firmas_suman_ocho(synthetic_results) -> None:
    """La distribución de firmas cubre los 8 datasets y hay 7 medias de dimensión."""
    rep = analysis.rq2_profiles(synthetic_results)
    assert sum(rep["stats"]["signature_distribution"].values()) == 8
    assert set(rep["stats"]["dim_means"]) == set(DIMENSIONS)


def test_rq3_cluster_particiona_ocho(synthetic_results) -> None:
    """El clustering asigna los 8 datasets a k∈{2,3,4} grupos."""
    rep = analysis.rq3_cluster(synthetic_results)
    assert rep["stats"]["k_best"] in (2, 3, 4)
    assert sum(rep["stats"]["cluster_sizes"].values()) == 8
    assert rep["table"].height == 8


def test_rq4_gap_real_sintetico(synthetic_results) -> None:
    """RQ4 agrupa por origen y calcula la brecha de DS real − sintético."""
    rep = analysis.rq4_groups(synthetic_results)
    assert "origin" in rep["table"].columns
    assert rep["stats"]["DS_mean_real_minus_synth"] is not None


def test_rq5_efeat_delta_determinista(synthetic_results) -> None:
    """En el sintético features = clean − 0.03 a N=full: ΔDS medio −0.03, 8/8 fáciles."""
    rep = analysis.rq5_efeat(synthetic_results)
    assert rep["stats"]["mean_dDS"] == pytest.approx(-0.03, abs=1e-6)
    assert rep["stats"]["n_easier_with_features"] == 8
    assert rep["stats"]["n_harder_with_features"] == 0


def test_rq6_escale_configs_y_sensibilidad(synthetic_results) -> None:
    """E-SCALE cubre 16 configuraciones (8×2) y nombra la dimensión más sensible."""
    rep = analysis.rq6_escale(synthetic_results)
    assert rep["stats"]["n_configs"] == 16
    assert rep["stats"]["most_sensitive_dim"] in DIMENSIONS


def test_rq7_proxy_correlacion_perfecta(synthetic_results, monkeypatch) -> None:
    """Con proxy = DS, Spearman ρ = 1 sobre los 8 datasets."""
    view = _cdata.canonical_view(synthetic_results).select("dataset_id", "DS")
    proxy = view.rename({"DS": "proxy"})
    monkeypatch.setattr(_cdata, "load_proxy", lambda: proxy)
    rep = analysis.rq7_proxy(synthetic_results)
    assert rep["stats"]["n"] == 8
    assert rep["stats"]["spearman_rho"] == pytest.approx(1.0, abs=1e-6)


def test_rq7_proxy_ausente(synthetic_results, monkeypatch) -> None:
    """Sin proxy disponible, RQ7 degrada a ρ nulo sin fallar."""
    empty = pl.DataFrame(schema={"dataset_id": pl.String, "proxy": pl.Float64})
    monkeypatch.setattr(_cdata, "load_proxy", lambda: empty)
    rep = analysis.rq7_proxy(synthetic_results)
    assert rep["stats"]["n"] == 0
    assert rep["stats"]["spearman_rho"] is None
