"""Reproducción vs. el benchmark original de CIPA (RQ8). Fase F4.

Verifica la consistencia de CIPA_FRAUD contra el benchmark publicado del framework
para los datasets con solapamiento: ``ulb_cc`` (CreditCard), ``paysim`` (PaySim)
e ``ieee_cis`` (IEEE-CIS), listados por :func:`cipa_fraud.registry.benchmark_overlap`.

Recomputa CIPA sobre estos datasets ya limpios y con ingeniería de features y
compara banda, firma y DS contra los valores del benchmark (CIPA v1.1.0, Tier-2 a
N=10k). Documenta cuánto desplaza el resultado el preprocesamiento del proyecto
comparativo frente a la carga cruda genérica del benchmark original.

El contraste se ancla en **banda y firma** (invariantes cualitativos del perfil);
el DS se compara con una tolerancia (:data:`DS_TOLERANCE`) y se reporta el ``ΔDS``.
Advertencia de versión: la referencia es CIPA **v1.1.0** y la recomputación corre
con la versión instalada (se registra ``cipa_version`` por corrida), de modo que
parte de cualquier ``ΔDS`` puede deberse al cambio de versión además del
preprocesamiento.

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
from datetime import UTC, datetime
from pathlib import Path

from cipa_fraud import registry, settings
from cipa_fraud.run import output_path, run_one, summary_row

#: Valores de referencia del benchmark de CIPA (v1.1.0, config Tier-2 a N=10k,
#: cargados desde los CSV crudos genéricos). Publicados en la tabla del framework;
#: son la fuente externa de comparación de RQ8 (por eso viven fijos aquí, no en el
#: registro de datasets). Clave = ``cipa_benchmark_key`` del registro.
BENCHMARK_REFERENCE: dict[str, dict] = {
    "CreditCard": {"DS": 0.3547, "band": "Moderate", "signature": "V"},
    "PaySim": {"DS": 0.3502, "band": "Moderate", "signature": "V"},
    "IEEE-CIS": {"DS": 0.5679, "band": "High", "signature": "V"},
}

#: Versión del benchmark de referencia (la recomputación puede correr otra).
CIPA_BENCHMARK_VERSION = "1.1.0"

#: Punto de escala del check: Tier-2 del benchmark = submuestreo genérico a N=10k.
TIER2_N = 10_000

#: Tolerancia absoluta de ``ΔDS`` para considerar el DS "reproducido".
DS_TOLERANCE = 0.10


def _recomputed_row(dataset_id: str, layer: str, random_state: int) -> dict:
    """Fila tidy recomputada para un dataset/capa a :data:`TIER2_N`.

    Reutiliza el JSON de la corrida si ya existe en ``results/`` (producido por
    F2/F3); en su defecto ejecuta la corrida al vuelo. Devuelve además la versión
    de CIPA con la que se computó, para la advertencia de versión de RQ8.

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset (con ``cipa_benchmark_key``).
    layer : str
        Capa de entrada: ``"clean"`` o ``"features"``.
    random_state : int
        Semilla propagada a la corrida si hay que recomputar.

    Returns
    -------
    dict
        Fila de :func:`cipa_fraud.run.summary_row` más ``cipa_version``.
    """
    path = output_path(dataset_id, layer, TIER2_N)
    if path.exists():
        out = json.loads(path.read_text())
    else:
        out = run_one(dataset_id, layer, TIER2_N, random_state)
    row = summary_row(out)
    row["cipa_version"] = out.get("cipa_version")
    return row


def check(random_state: int = settings.RANDOM_STATE, write: bool = True) -> dict:
    """Compara banda/firma/DS de los datasets solapados contra el benchmark CIPA.

    Para cada dataset con ``cipa_benchmark_key`` y cada capa (``clean``/
    ``features``) recupera (o recomputa) la corrida Tier-2 a N=10k y la contrasta
    con el valor de referencia publicado: coincidencia de banda y firma, ``ΔDS`` y
    si cae dentro de :data:`DS_TOLERANCE`.

    Parameters
    ----------
    random_state : int, optional
        Semilla para cualquier corrida que haya que recomputar. Por defecto
        :data:`cipa_fraud.settings.RANDOM_STATE`.
    write : bool, optional
        Si ``True`` (por defecto) escribe el reporte en ``reports/consistency/``
        (JSON estructurado + tabla Markdown).

    Returns
    -------
    dict
        Reporte con ``comparisons`` (una entrada por dataset × capa), ``summary``
        (conteos de coincidencia de banda/firma y DS dentro de tolerancia),
        metadatos (``cipa_benchmark_version``, ``cipa_version_recomputed``,
        ``tier2_n``, ``ds_tolerance``) y ``timestamp``.
    """
    overlap = registry.benchmark_overlap()  # {dataset_id -> cipa_benchmark_key}

    comparisons: list[dict] = []
    recomputed_versions: set[str] = set()
    for dataset_id, cipa_key in overlap.items():
        ref = BENCHMARK_REFERENCE[cipa_key]
        for layer in settings.LAYERS:
            row = _recomputed_row(dataset_id, layer, random_state)
            if row.get("cipa_version"):
                recomputed_versions.add(row["cipa_version"])
            delta = row["DS"] - ref["DS"]
            comparisons.append({
                "dataset_id": dataset_id,
                "cipa_key": cipa_key,
                "layer": layer,
                "n_target": str(TIER2_N),
                "DS": round(row["DS"], 4),
                "band": row["band"],
                "signature": row["signature"],
                "ref_DS": ref["DS"],
                "ref_band": ref["band"],
                "ref_signature": ref["signature"],
                "delta_DS": round(delta, 4),
                "band_match": row["band"] == ref["band"],
                "signature_match": row["signature"] == ref["signature"],
                "within_tol": abs(delta) <= DS_TOLERANCE,
            })

    n = len(comparisons)
    summary = {
        "n_comparisons": n,
        "band_match": sum(c["band_match"] for c in comparisons),
        "signature_match": sum(c["signature_match"] for c in comparisons),
        "ds_within_tol": sum(c["within_tol"] for c in comparisons),
        "max_abs_delta_DS": round(max((abs(c["delta_DS"]) for c in comparisons),
                                      default=0.0), 4),
    }
    report = {
        "cipa_benchmark_version": CIPA_BENCHMARK_VERSION,
        "cipa_version_recomputed": sorted(recomputed_versions),
        "tier2_n": TIER2_N,
        "ds_tolerance": DS_TOLERANCE,
        "comparisons": comparisons,
        "summary": summary,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if write:
        _write_report(report)
    return report


def _report_dir() -> Path:
    """Directorio de salida del reporte de consistencia: ``reports/consistency/``."""
    return settings.REPORTS_DIR / "consistency"


def _write_report(report: dict) -> None:
    """Persiste el reporte de consistencia: ``consistency.json`` + ``.md``."""
    out_dir = _report_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "consistency.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    (out_dir / "consistency.md").write_text(_render_markdown(report))


def _render_markdown(report: dict) -> str:
    """Arma la tabla Markdown legible del reporte de consistencia (RQ8)."""
    s = report["summary"]
    ver_rec = ", ".join(report["cipa_version_recomputed"]) or "n/d"
    lines = [
        "# Consistencia RQ8 — CIPA_FRAUD vs. benchmark CIPA",
        "",
        f"- Benchmark de referencia: **CIPA v{report['cipa_benchmark_version']}** "
        f"(Tier-2, N={report['tier2_n']:,}, carga cruda genérica).",
        f"- Recomputado con: **CIPA v{ver_rec}** (capas `clean`/`features` del "
        "proyecto comparativo).",
        f"- Tolerancia de DS: ±{report['ds_tolerance']:.2f}.",
        f"- Generado: {report['timestamp']}.",
        "",
        f"**Resumen:** banda {s['band_match']}/{s['n_comparisons']}, "
        f"firma {s['signature_match']}/{s['n_comparisons']}, "
        f"DS dentro de tolerancia {s['ds_within_tol']}/{s['n_comparisons']} "
        f"(|ΔDS| máx. {s['max_abs_delta_DS']:.4f}).",
        "",
        "| dataset | CIPA key | capa | DS | banda | firma | "
        "DS ref | banda ref | firma ref | ΔDS | banda✓ | firma✓ | DS±tol |",
        "|---|---|---|---|---|---|---|---|---|---|:---:|:---:|:---:|",
    ]
    tick = {True: "✓", False: "✗"}
    for c in report["comparisons"]:
        lines.append(
            f"| {c['dataset_id']} | {c['cipa_key']} | {c['layer']} | "
            f"{c['DS']:.4f} | {c['band']} | {c['signature']} | "
            f"{c['ref_DS']:.4f} | {c['ref_band']} | {c['ref_signature']} | "
            f"{c['delta_DS']:+.4f} | {tick[c['band_match']]} | "
            f"{tick[c['signature_match']]} | {tick[c['within_tol']]} |"
        )
    lines.extend([
        "",
        "**Lectura:** banda y firma son los invariantes cualitativos del perfil "
        "CIPA; su coincidencia indica reproducción del diagnóstico. El `ΔDS` "
        "captura el desplazamiento cuantitativo atribuible a limpieza + ingeniería "
        "de features frente a la carga cruda del benchmark (y, en parte, al cambio "
        "de versión de CIPA).",
        "",
    ])
    return "\n".join(lines)
