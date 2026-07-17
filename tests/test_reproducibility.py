"""Pruebas del manifiesto de reproducibilidad (:mod:`cipa_fraud.report.reproducibility`)."""

from __future__ import annotations

import hashlib

from cipa_fraud.report import reproducibility as repro


def test_sha256_file_coincide(tmp_path) -> None:
    """El hash y el tamaño por bloques coinciden con el cálculo de referencia."""
    content = b"CIPA_FRAUD" * 1000
    f = tmp_path / "dato.bin"
    f.write_bytes(content)
    digest, nbytes = repro._sha256_file(f)
    assert digest == hashlib.sha256(content).hexdigest()
    assert nbytes == len(content)


def test_dep_versions_incluye_python_y_cipa() -> None:
    """El bloque de versiones reporta Python y las dependencias rastreadas."""
    deps = repro.dep_versions()
    assert "python" in deps
    assert "cipa" in deps
    assert all(isinstance(v, str) for v in deps.values())
