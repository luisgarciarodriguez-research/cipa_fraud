"""Orquestador del análisis comparativo (RQ1–RQ7). Fase F5.

Ejecuta las siete RQ transversales de :mod:`cipa_fraud.compare.analysis`, genera las
figuras de :mod:`cipa_fraud.compare.figures` y ensambla los productos en
``reports/comparative/``:

- ``comparative.json`` — todas las tablas (como registros) y escalares por RQ;
- ``comparative.md`` — informe legible con tablas clave y hallazgos por RQ;
- ``tables/RQ*.csv`` — cada tabla de RQ como CSV;
- ``figures/*.png`` — las figuras del estudio.

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

import polars as pl

from cipa_fraud import settings
from cipa_fraud.compare import analysis, figures


def output_dir() -> Path:
    """Directorio de salida del análisis comparativo: ``reports/comparative/``."""
    return settings.REPORTS_DIR / "comparative"


def _md_table(df: pl.DataFrame, max_rows: int = 20) -> str:
    """Renderiza un ``polars.DataFrame`` como tabla Markdown (truncada)."""
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.head(max_rows).iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            elif isinstance(v, bool):
                cells.append("✓" if v else "✗")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_markdown(reports: list[dict], fig_names: list[str]) -> str:
    """Ensambla el informe Markdown del análisis comparativo."""
    by = {r["name"]: r for r in reports}
    ts = datetime.now(UTC).isoformat()
    out = [
        "# Análisis comparativo CIPA del dominio fraude (RQ1–RQ7)",
        "",
        f"Generado: {ts}. Fuente: `results/all_results.parquet` "
        "(8 datasets × 2 capas × 3 escalas). Corte canónico: **features / N=full**.",
        "",
    ]

    def section(name: str, extra: str = "") -> None:
        r = by[name]
        out.append(f"## {name} — {r['question']}")
        out.append("")
        if extra:
            out.append(extra)
            out.append("")
        out.append(_md_table(r["table"]))
        out.append("")
        out.append(f"*Nota:* {r['note']}")
        out.append("")

    s1 = by["RQ1"]["stats"]
    section("RQ1", f"Más difícil: **{s1['hardest']}** (DS {s1['DS_max']}); más "
                   f"fácil: **{s1['easiest']}** (DS {s1['DS_min']}). "
                   f"Bandas: {s1['band_counts']}. Ver `figures/rq1_ranking.png`.")

    s2 = by["RQ2"]["stats"]
    section("RQ2", f"Intensidad media por dimensión: {s2['dim_means']}. "
                   f"Dimensión más intensa: **{s2['dimension_most_intense']}**. "
                   f"Firmas: {s2['signature_distribution']}. "
                   "Ver `figures/rq2_heatmap.png`, `figures/rq2_signatures.png`.")

    s3 = by["RQ3"]["stats"]
    section("RQ3", f"k óptimo = {s3['k_best']} (silueta {s3['silhouette']}); "
                   f"tamaños {s3['cluster_sizes']}.")

    s4 = by["RQ4"]["stats"]
    section("RQ4", f"ΔDS medio real − sintético: {s4['DS_mean_real_minus_synth']}. "
                   "Medias por dominio en `comparative.json`.")

    s5 = by["RQ5"]["stats"]
    section("RQ5", f"ΔDS medio (features − clean): **{s5['mean_dDS']}**. "
                   f"Más fáciles con features: {s5['n_easier_with_features']}/8; "
                   f"firmas que cambian: {s5['n_signature_changed']}. "
                   "Ver `figures/rq5_efeat.png`.")

    s6 = by["RQ6"]["stats"]
    section("RQ6", f"Dimensión más sensible al submuestreo: "
                   f"**{s6['most_sensitive_dim']}**. Banda estable en "
                   f"{s6['n_configs_band_stable']}/{s6['n_configs']} configs; "
                   f"firma estable en {s6['n_configs_sig_stable']}/{s6['n_configs']}. "
                   "Ver `figures/rq6_escale.png`.")

    s7 = by["RQ7"]["stats"]
    section("RQ7", f"Spearman ρ(proxy, DS) = **{s7['spearman_rho']}** "
                   f"(p = {s7['p_value']}, n = {s7['n']}). "
                   "Ver `figures/rq7_proxy.png`.")

    out.append("## Figuras")
    out.append("")
    out.extend(f"- `figures/{n}`" for n in fig_names)
    out.append("")
    return "\n".join(out)


def build(write: bool = True) -> dict:
    """Ejecuta el análisis comparativo completo y escribe los productos de F5.

    Parameters
    ----------
    write : bool, optional
        Si ``True`` (por defecto) escribe ``comparative.json``, ``comparative.md``,
        las tablas CSV y las figuras en ``reports/comparative/``.

    Returns
    -------
    dict
        Resumen: ``reports`` (lista de dicts por RQ), rutas escritas y nombres de
        figuras.
    """
    reports = analysis.run_all()
    by_name = {r["name"]: r for r in reports}

    result: dict = {"reports": reports, "figures": [], "paths": {}}
    if not write:
        return result

    out_dir = output_dir()
    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Tablas CSV por RQ + JSON consolidado (tablas como registros + escalares).
    json_payload: dict = {
        "generated": datetime.now(UTC).isoformat(),
        "canonical": {"layer": "features", "n_target": "full"},
        "rqs": {},
    }
    for r in reports:
        r["table"].write_csv(tables_dir / f"{r['name']}.csv")
        json_payload["rqs"][r["name"]] = {
            "question": r["question"],
            "stats": r["stats"],
            "note": r["note"],
            "table": r["table"].to_dicts(),
        }
    (out_dir / "comparative.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False, default=str)
    )

    # Figuras.
    fig_paths = figures.build_all(by_name, fig_dir)
    fig_names = [p.name for p in fig_paths]

    # Informe Markdown.
    (out_dir / "comparative.md").write_text(_render_markdown(reports, fig_names))

    result["figures"] = fig_names
    result["paths"] = {
        "json": str(out_dir / "comparative.json"),
        "markdown": str(out_dir / "comparative.md"),
        "tables_dir": str(tables_dir),
        "figures_dir": str(fig_dir),
    }
    return result
