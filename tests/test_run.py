"""Pruebas de la orquestación de corridas (:mod:`cipa_fraud.run`).

Cubre los helpers puros y —de forma hermética, con ``run_one`` simulado y un
directorio de resultados temporal— la reanudabilidad de ``run_all``, el aislamiento
de fallos y la consolidación tidy.
"""

from __future__ import annotations

import json

import pytest

from cipa_fraud import run, settings
from tests.conftest import make_out


def test_n_label() -> None:
    """``_n_label`` normaliza enteros y conserva ``"full"``."""
    assert run._n_label("full") == "full"
    assert run._n_label(10_000) == "10000"


def test_output_path(tmp_path, monkeypatch) -> None:
    """La ruta de salida sigue ``results/<id>/<layer>/<N>.json``."""
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path)
    p = run.output_path("fdb", "clean", 10_000)
    assert p == tmp_path / "fdb" / "clean" / "10000.json"


def test_summary_row_extrae_campos() -> None:
    """``summary_row`` aplana identificación, escala, DS/firma y D1–D7."""
    out = make_out("fdb", "features", "full", ds=0.53, band="High", signature="III")
    row = run.summary_row(out)
    assert row["dataset_id"] == "fdb"
    assert row["DS"] == 0.53
    assert row["band"] == "High"
    assert row["signature"] == "III"
    assert {f"D{i}" for i in range(1, 8)} <= set(row)


def test_iter_grid_producto_cartesiano() -> None:
    """``_iter_grid`` recorre datasets × capas × escalas en orden."""
    grid = list(run._iter_grid(["a", "b"], ["clean"], [10_000, "full"]))
    assert grid == [
        ("a", "clean", 10_000), ("a", "clean", "full"),
        ("b", "clean", 10_000), ("b", "clean", "full"),
    ]


@pytest.fixture
def fake_run_one(monkeypatch, tmp_path):
    """Sustituye ``run_one`` por una versión que escribe un ``out`` sintético.

    Redirige ``RESULTS_DIR`` a un temporal y cuenta las invocaciones, para poder
    verificar cuándo se recomputa y cuándo se omite por cache.
    """
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path)
    calls: list[tuple] = []

    def _fake(dataset_id, layer, n_target, random_state=settings.RANDOM_STATE):
        calls.append((dataset_id, layer, run._n_label(n_target)))
        out = make_out(dataset_id, layer, run._n_label(n_target))
        path = run.output_path(dataset_id, layer, n_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return out

    monkeypatch.setattr(run, "run_one", _fake)
    return calls


def test_run_all_reanudable_omite_existentes(fake_run_one) -> None:
    """La segunda corrida omite lo ya escrito; ``--force`` lo recomputa."""
    kwargs = {"datasets": ["fdb"], "layers": ["clean"], "sweep": [10_000]}

    first = run.run_all(**kwargs)
    assert first["n_ok"] == 1 and first["n_skip"] == 0
    assert len(fake_run_one) == 1

    second = run.run_all(**kwargs)
    assert second["n_ok"] == 0 and second["n_skip"] == 1
    assert len(fake_run_one) == 1  # no se volvió a llamar

    forced = run.run_all(**kwargs, force=True)
    assert forced["n_ok"] == 1
    assert len(fake_run_one) == 2  # recomputado


def test_run_all_aisla_fallos(monkeypatch, tmp_path) -> None:
    """Una corrida que revienta se registra como desviación sin abortar."""
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path)

    def _boom(dataset_id, layer, n_target, random_state=settings.RANDOM_STATE):
        raise ValueError("fallo simulado")

    monkeypatch.setattr(run, "run_one", _boom)
    summary = run.run_all(datasets=["fdb"], layers=["clean"], sweep=[10_000])
    assert summary["n_fail"] == 1
    assert summary["deviations"][0]["dataset_id"] == "fdb"
    assert "fallo simulado" in summary["deviations"][0]["error"]


def test_consolidate_arma_parquet(fake_run_one) -> None:
    """``consolidate`` reúne los JSON en disco en un tidy con ``n_target`` texto."""
    run.run_all(datasets=["fdb"], layers=["clean"], sweep=[10_000, "full"])
    df = run.consolidate(write=True)
    assert df.height == 2
    assert set(df["n_target"].to_list()) == {"10000", "full"}
    assert run.all_results_path().exists()
