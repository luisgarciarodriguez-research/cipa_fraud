"""Ensamblado del informe comparativo final del estudio (HTML + Markdown). Fase F6.

Toma los productos de las fases previas —el consolidado de F3, el análisis
comparativo de F5 (:mod:`cipa_fraud.compare`), la consistencia de F4
(:mod:`cipa_fraud.consistency`) y el manifiesto de reproducibilidad
(:mod:`cipa_fraud.report.reproducibility`)— y ensambla un único informe con:
panorama general, ranking de dificultad, perfiles y firmas, hallazgos por RQ,
consistencia con el benchmark, una ficha por dataset (perfil D1–D7 + recomendaciones
de acción de CIPA) y el bloque de reproducibilidad.

Escribe ``reports/comparative/report.md`` (referencia las figuras en ``figures/``) y
``reports/comparative/report.html`` (autocontenido: CSS embebido y figuras en
base64), más ``reports/reproducibility.json``.

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

import base64
import html
import json
from datetime import UTC, datetime
from pathlib import Path

from cipa_fraud import consistency as _consistency
from cipa_fraud import registry, settings
from cipa_fraud.compare import build as _compare
from cipa_fraud.compare import data as _cdata
from cipa_fraud.compare.data import DIMENSION_NAMES, DIMENSIONS
from cipa_fraud.report import reproducibility as _repro
from cipa_fraud.run import output_path


def output_dir() -> Path:
    """Directorio de salida del informe: ``reports/comparative/``."""
    return settings.REPORTS_DIR / "comparative"


def _finding(rep: dict) -> str:
    """Sintetiza una línea de hallazgo por RQ a partir de sus escalares."""
    s, name = rep["stats"], rep["name"]
    if name == "RQ1":
        return (f"Más difícil **{s['hardest']}** (DS {s['DS_max']}), más fácil "
                f"**{s['easiest']}** (DS {s['DS_min']}); bandas {s['band_counts']}.")
    if name == "RQ2":
        return (f"La firma **{max(s['signature_distribution'], key=s['signature_distribution'].get)}** "
                f"domina ({s['signature_distribution']}); dimensión más intensa "
                f"**{s['dimension_most_intense']}** "
                f"({DIMENSION_NAMES[s['dimension_most_intense']]}).")
    if name == "RQ3":
        return (f"Clustering interno k={s['k_best']} con silueta {s['silhouette']} "
                f"(débil ⇒ fraude heterogéneo en 7-D); tamaños {s['cluster_sizes']}.")
    if name == "RQ4":
        return (f"ΔDS medio real − sintético = {s['DS_mean_real_minus_synth']} "
                "(descriptivo; n por grupo pequeño).")
    if name == "RQ5":
        return (f"Las features **bajan el DS en {s['n_easier_with_features']}/8** "
                f"datasets (ΔDS medio {s['mean_dDS']}); firmas que cambian: "
                f"{s['n_signature_changed']}.")
    if name == "RQ6":
        return (f"Dimensión más sensible al submuestreo **{s['most_sensitive_dim']}** "
                f"({DIMENSION_NAMES[s['most_sensitive_dim']]}); banda estable en "
                f"{s['n_configs_band_stable']}/{s['n_configs']} configs.")
    if name == "RQ7":
        rho = s.get("spearman_rho")
        verdict = "no correlaciona" if rho is None or abs(rho) < 0.5 else "correlaciona"
        return (f"El proxy heurístico **{verdict}** con el DS real "
                f"(Spearman ρ={rho}, p={s.get('p_value')}, n={s['n']}).")
    return rep.get("note", "")


def _dataset_cards() -> list[dict]:
    """Ficha por dataset a features/full: perfil, escala y recomendaciones."""
    meta = {r["dataset_id"]: r for r in _cdata.load_metadata().to_dicts()}
    cards: list[dict] = []
    for ds_id in registry.all_ids():
        path = output_path(ds_id, _cdata.CANONICAL_LAYER, _cdata.CANONICAL_N)
        if not path.exists():
            continue
        o = json.loads(path.read_text())
        ds = o["cipa"]["difficulty_score"]
        prof = o["cipa"]["profile"]
        act = o["cipa"].get("action", {})
        m = o["manifest"]
        dims = {d["dimension_id"]: round(d["value"], 4) for d in ds["dimensions"]}
        cards.append({
            "dataset_id": ds_id,
            "name": meta[ds_id]["name"],
            "domain": meta[ds_id]["domain"],
            "origin": meta[ds_id]["origin"],
            "fraud_rate": meta[ds_id]["fraud_rate"],
            "N": m["N"], "d": m["d"], "IR": m["IR"], "IR_eff": m["IR_eff"],
            "DS": round(ds["value"], 4),
            "band": ds["band"],
            "signature": prof["signature"],
            "signature_name": prof.get("signature_name", ""),
            "dims": dims,
            "evaluation_metrics": act.get("evaluation_metrics", []),
            "preprocessing_strategy": act.get("preprocessing_strategy", []),
            "model_families": act.get("model_families", []),
        })
    return cards


def _build_model(hash_inputs: bool = True) -> dict:
    """Reúne todo el contenido del informe en un modelo serializable."""
    compare_out = _compare.build(write=True)
    reports = {r["name"]: r for r in compare_out["reports"]}
    consistency = _consistency.check(write=True)
    repro = _repro.build(write=True, hash_inputs=hash_inputs)

    view = _cdata.canonical_view()
    bands = dict(
        view.group_by("band").len().sort("len", descending=True)
        .iter_rows()
    )
    sigs = dict(
        view.group_by("signature").len().sort("len", descending=True)
        .iter_rows()
    )
    ranking = reports["RQ1"]["table"].to_dicts()

    return {
        "generated": datetime.now(UTC).isoformat(),
        "panorama": {
            "n_datasets": view.height,
            "bands": bands,
            "signatures": sigs,
            "hardest": ranking[0]["dataset_id"],
            "easiest": ranking[-1]["dataset_id"],
            "dominant_signature": max(sigs, key=sigs.get),
        },
        "ranking": ranking,
        "rq_findings": [
            {"name": r["name"], "question": r["question"],
             "finding": _finding(r), "note": r["note"]}
            for r in compare_out["reports"]
        ],
        "consistency": consistency,
        "datasets": _dataset_cards(),
        "reproducibility": {
            "cipa_versions": repro["cipa_versions"],
            "random_state": repro["random_state"],
            "scale_params": repro["scale_params"],
            "dependencies": repro["dependencies"],
            "n_runs": repro["n_runs"],
            "n_inputs": len(repro["inputs"]),
        },
        "figures": compare_out["figures"],
    }


# --------------------------------------------------------------------------
# Renderizado Markdown
# --------------------------------------------------------------------------

def _md_dims_row(dims: dict) -> str:
    """Fila Markdown de las 7 dimensiones."""
    return " | ".join(f"{dims[d]:.3f}" for d in DIMENSIONS)


def render_markdown(model: dict) -> str:
    """Informe en Markdown (referencia las figuras en ``figures/``)."""
    p = model["panorama"]
    out = [
        "# Estudio comparativo CIPA del dominio fraude digital",
        "",
        f"*Generado: {model['generated']}.*",
        "",
        "## Panorama general",
        "",
        f"Se caracterizaron **{p['n_datasets']} datasets tabulares de fraude** con "
        "el framework CIPA (8 × 2 capas × 3 escalas = 48 corridas). En el corte "
        "canónico *features / N=full*: bandas de dificultad "
        f"{p['bands']}, firmas {p['signatures']}. La firma dominante es "
        f"**{p['dominant_signature']}**; el dataset más difícil es "
        f"**{p['hardest']}** y el más fácil **{p['easiest']}**.",
        "",
        "## Ranking de dificultad (RQ1)",
        "",
        "| # | dataset | dominio | origen | DS | banda | firma |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(model["ranking"], 1):
        out.append(f"| {i} | {r['dataset_id']} | {r['domain']} | {r['origin']} | "
                   f"{r['DS']:.4f} | {r['band']} | {r['signature']} |")
    out += ["", "![Ranking](figures/rq1_ranking.png)", "",
            "## Perfiles y firmas (RQ2)", "",
            "![Heatmap D1–D7](figures/rq2_heatmap.png)", "",
            "![Firmas](figures/rq2_signatures.png)", "",
            "## Hallazgos por pregunta de investigación", ""]
    for f in model["rq_findings"]:
        out.append(f"- **{f['name']} — {f['question']}:** {f['finding']}")
    out += ["", "![E-FEAT](figures/rq5_efeat.png)", "",
            "![E-SCALE](figures/rq6_escale.png)", "",
            "![Proxy vs DS](figures/rq7_proxy.png)", ""]

    c = model["consistency"]
    cs = c["summary"]
    out += [
        "## Consistencia con el benchmark de CIPA (RQ8)", "",
        f"Recomputado con CIPA v{', '.join(c['cipa_version_recomputed'])} vs. "
        f"referencia v{c['cipa_benchmark_version']} (Tier-2, N={c['tier2_n']:,}). "
        f"Banda {cs['band_match']}/{cs['n_comparisons']}, firma "
        f"{cs['signature_match']}/{cs['n_comparisons']}, DS±{c['ds_tolerance']} "
        f"{cs['ds_within_tol']}/{cs['n_comparisons']}.", "",
        "| dataset | capa | DS | ref | ΔDS | banda✓ | firma✓ |",
        "|---|---|---|---|---|---|---|",
    ]
    for cc in c["comparisons"]:
        out.append(f"| {cc['dataset_id']} | {cc['layer']} | {cc['DS']:.4f} | "
                   f"{cc['ref_DS']:.4f} | {cc['delta_DS']:+.4f} | "
                   f"{'✓' if cc['band_match'] else '✗'} | "
                   f"{'✓' if cc['signature_match'] else '✗'} |")

    out += ["", "## Fichas por dataset", "",
            "| dataset | dominio | origen | N | DS | banda | firma | "
            + " | ".join(DIMENSIONS) + " |",
            "|---|---|---|---|---|---|---|"
            + "|".join(["---"] * len(DIMENSIONS)) + "|"]
    for d in model["datasets"]:
        out.append(f"| {d['dataset_id']} | {d['domain']} | {d['origin']} | "
                   f"{d['N']:,} | {d['DS']:.4f} | {d['band']} | {d['signature']} | "
                   + _md_dims_row(d["dims"]) + " |")
    out.append("")
    for d in model["datasets"]:
        out += [
            f"### {d['dataset_id']} — {d['name']}",
            "",
            f"- Dominio {d['domain']} · origen {d['origin']} · N={d['N']:,} · "
            f"d={d['d']} · IR={d['IR']:.1f} (efectivo {d['IR_eff']:.1f}).",
            f"- DS **{d['DS']:.4f}** · banda **{d['band']}** · firma "
            f"**{d['signature']}** ({d['signature_name']}).",
            f"- Métricas sugeridas: {', '.join(d['evaluation_metrics']) or '—'}.",
            f"- Preprocesamiento: {', '.join(d['preprocessing_strategy']) or '—'}.",
            "",
        ]

    r = model["reproducibility"]
    out += [
        "## Reproducibilidad", "",
        f"- CIPA v{', '.join(r['cipa_versions'])} · semilla {r['random_state']} · "
        f"{r['n_runs']} corridas · {r['n_inputs']} Parquet de entrada (hash "
        "SHA-256 en `reproducibility.json`).",
        f"- Escala: {r['scale_params']}.",
        f"- Dependencias: {r['dependencies']}.",
        "",
        "*Detalle completo (hashes por archivo, runtime por corrida) en "
        "`reports/reproducibility.json`.*",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Renderizado HTML (autocontenido)
# --------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55;
  color: #1a1a1a; background: #fff; }
h1 { border-bottom: 3px solid #0072B2; padding-bottom: .3rem; }
h2 { border-bottom: 1px solid #ddd; padding-bottom: .2rem; margin-top: 2.2rem; }
h3 { margin-top: 1.6rem; color: #0072B2; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .92rem; }
th, td { border: 1px solid #ddd; padding: .35rem .55rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f2f6fa; }
tr:nth-child(even) td { background: #fafafa; }
img { max-width: 100%; height: auto; display: block; margin: 1rem auto;
  border: 1px solid #eee; border-radius: 6px; }
code { background: #f2f2f2; padding: .1rem .3rem; border-radius: 3px; }
.card { border: 1px solid #e3e3e3; border-radius: 8px; padding: .8rem 1.1rem;
  margin: 1rem 0; background: #fbfcfd; }
.badge { display: inline-block; padding: .1rem .5rem; border-radius: 12px;
  font-size: .8rem; color: #fff; }
.muted { color: #666; font-size: .9rem; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e6e6; background: #1b1b1b; }
  th { background: #23303a; } tr:nth-child(even) td { background: #222; }
  td, th { border-color: #333; } .card { background: #222; border-color: #333; }
  code { background: #2a2a2a; } h3 { color: #56B4E9; }
}
"""

_BAND_BADGE = {"Low": "#56B4E9", "Moderate": "#0072B2",
               "High": "#D55E00", "Extreme": "#000"}


def _img_data_uri(path: Path) -> str:
    """Devuelve un ``data:`` URI base64 de un PNG (para HTML autocontenido)."""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _fig_tag(fig_dir: Path, name: str, alt: str) -> str:
    """Etiqueta ``<img>`` embebida si la figura existe; si no, cadena vacía."""
    path = fig_dir / name
    if not path.exists():
        return ""
    return f'<img src="{_img_data_uri(path)}" alt="{html.escape(alt)}">'


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    """Tabla HTML a partir de encabezados y filas (celdas ya formateadas)."""
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(model: dict, fig_dir: Path) -> str:
    """Informe HTML autocontenido (CSS embebido, figuras en base64)."""
    p = model["panorama"]
    parts = [
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Estudio comparativo CIPA — fraude digital</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>Estudio comparativo CIPA del dominio fraude digital</h1>",
        f"<p class='muted'>Generado: {html.escape(model['generated'])}.</p>",
        "<h2>Panorama general</h2>",
        f"<p>Se caracterizaron <b>{p['n_datasets']} datasets tabulares de fraude</b> "
        "con el framework CIPA (8 × 2 capas × 3 escalas = 48 corridas). Corte "
        f"canónico <i>features / N=full</i>. Bandas: {html.escape(str(p['bands']))}; "
        f"firmas: {html.escape(str(p['signatures']))}. Firma dominante "
        f"<b>{p['dominant_signature']}</b>; más difícil <b>{p['hardest']}</b>, "
        f"más fácil <b>{p['easiest']}</b>.</p>",
        "<h2>Ranking de dificultad (RQ1)</h2>",
        _html_table(
            ["#", "dataset", "dominio", "origen", "DS", "banda", "firma"],
            [[i, r["dataset_id"], r["domain"], r["origin"], f"{r['DS']:.4f}",
              r["band"], r["signature"]]
             for i, r in enumerate(model["ranking"], 1)],
        ),
        _fig_tag(fig_dir, "rq1_ranking.png", "Ranking de dificultad"),
        "<h2>Perfiles y firmas (RQ2)</h2>",
        _fig_tag(fig_dir, "rq2_heatmap.png", "Heatmap D1-D7"),
        _fig_tag(fig_dir, "rq2_signatures.png", "Distribución de firmas"),
        "<h2>Hallazgos por pregunta de investigación</h2><ul>",
    ]
    for f in model["rq_findings"]:
        parts.append(f"<li><b>{f['name']} — {html.escape(f['question'])}:</b> "
                     f"{_inline_md(f['finding'])}</li>")
    parts.append("</ul>")
    parts += [
        _fig_tag(fig_dir, "rq5_efeat.png", "E-FEAT"),
        _fig_tag(fig_dir, "rq6_escale.png", "E-SCALE"),
        _fig_tag(fig_dir, "rq7_proxy.png", "Proxy vs DS"),
    ]

    c = model["consistency"]
    cs = c["summary"]
    parts += [
        "<h2>Consistencia con el benchmark de CIPA (RQ8)</h2>",
        f"<p>Recomputado con CIPA v{', '.join(c['cipa_version_recomputed'])} vs. "
        f"referencia v{c['cipa_benchmark_version']} (Tier-2, N={c['tier2_n']:,}). "
        f"Banda {cs['band_match']}/{cs['n_comparisons']}, firma "
        f"{cs['signature_match']}/{cs['n_comparisons']}, DS±{c['ds_tolerance']} "
        f"{cs['ds_within_tol']}/{cs['n_comparisons']}.</p>",
        _html_table(
            ["dataset", "capa", "DS", "ref", "ΔDS", "banda✓", "firma✓"],
            [[cc["dataset_id"], cc["layer"], f"{cc['DS']:.4f}", f"{cc['ref_DS']:.4f}",
              f"{cc['delta_DS']:+.4f}", "✓" if cc["band_match"] else "✗",
              "✓" if cc["signature_match"] else "✗"]
             for cc in c["comparisons"]],
        ),
        "<h2>Fichas por dataset</h2>",
    ]
    for d in model["datasets"]:
        badge = _BAND_BADGE.get(d["band"], "#666")
        dims_tbl = _html_table(
            [f"{k} {DIMENSION_NAMES[k]}" for k in DIMENSIONS],
            [[f"{d['dims'][k]:.3f}" for k in DIMENSIONS]],
        )
        parts.append(
            f"<div class='card'><h3>{d['dataset_id']} — {html.escape(d['name'])}</h3>"
            f"<p class='muted'>{d['domain']} · {d['origin']} · N={d['N']:,} · "
            f"d={d['d']} · IR={d['IR']:.1f} (ef. {d['IR_eff']:.1f})</p>"
            f"<p>DS <b>{d['DS']:.4f}</b> · <span class='badge' "
            f"style='background:{badge}'>{d['band']}</span> · firma "
            f"<b>{d['signature']}</b> ({html.escape(d['signature_name'])})</p>"
            f"{dims_tbl}"
            f"<p class='muted'>Métricas: {html.escape(', '.join(d['evaluation_metrics']) or '—')}<br>"
            f"Preprocesamiento: {html.escape(', '.join(d['preprocessing_strategy']) or '—')}<br>"
            f"Modelos: {html.escape(', '.join(d['model_families']) or '—')}</p></div>"
        )

    r = model["reproducibility"]
    parts += [
        "<h2>Reproducibilidad</h2>",
        f"<p>CIPA v{', '.join(r['cipa_versions'])} · semilla {r['random_state']} · "
        f"{r['n_runs']} corridas · {r['n_inputs']} Parquet de entrada "
        "(hash SHA-256 en <code>reproducibility.json</code>).</p>",
        f"<p class='muted'>Escala: {html.escape(str(r['scale_params']))}<br>"
        f"Dependencias: {html.escape(str(r['dependencies']))}</p>",
        "</body></html>",
    ]
    return "".join(parts)


def _inline_md(text: str) -> str:
    """Convierte ``**negritas**`` de una línea Markdown a ``<b>`` en HTML."""
    parts = text.split("**")
    out = []
    for i, seg in enumerate(parts):
        seg = html.escape(seg)
        out.append(f"<b>{seg}</b>" if i % 2 == 1 else seg)
    return "".join(out)


def build(write: bool = True, hash_inputs: bool = True) -> dict:
    """Ensambla el informe final del estudio (Markdown + HTML) y lo persiste.

    Parameters
    ----------
    write : bool, optional
        Si ``True`` (por defecto) escribe ``report.md``, ``report.html`` y (vía las
        sub-fases) ``comparative.*``, ``consistency.*`` y ``reproducibility.json``.
    hash_inputs : bool, optional
        Propagado a :func:`cipa_fraud.report.reproducibility.build` (hash de los
        Parquet de entrada). Por defecto ``True``.

    Returns
    -------
    dict
        ``model`` del informe y rutas escritas.
    """
    model = _build_model(hash_inputs=hash_inputs)
    result = {"model": model, "paths": {}}
    if not write:
        return result

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    (out_dir / "report.md").write_text(render_markdown(model))
    (out_dir / "report.html").write_text(render_html(model, fig_dir))
    result["paths"] = {
        "markdown": str(out_dir / "report.md"),
        "html": str(out_dir / "report.html"),
        "reproducibility": str(settings.REPORTS_DIR / "reproducibility.json"),
    }
    return result
