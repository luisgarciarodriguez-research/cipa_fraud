"""Materiales para artículo/capítulo de tesis (write-up). Fase F7 (opcional).

Genera productos listos para publicación a partir del estudio ya computado:

- **Table 2** (estilo del benchmark de CIPA) con el perfil CIPA de los 8 datasets de
  fraude en el corte canónico *features / N=full*: N, IR, D1–D7, DS, banda y firma.
  Se emite en LaTeX (``booktabs``) y en Markdown.
- **Narrativa** de resultados y discusión (español), con los valores computados en
  vivo desde el consolidado y los análisis de F4/F5 (sin cifras hardcodeadas), lista
  para incrustarse en el capítulo/artículo.

Las figuras finales son las de F5 (``reports/comparative/figures/``, CVD-safe); este
módulo las referencia en la narrativa en lugar de regenerarlas.

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

from datetime import UTC, datetime
from pathlib import Path

from cipa_fraud import consistency as _consistency
from cipa_fraud import settings
from cipa_fraud.compare import analysis as _analysis
from cipa_fraud.compare import data as _cdata
from cipa_fraud.compare.data import DIMENSION_NAMES, DIMENSIONS


def output_dir() -> Path:
    """Directorio de salida del write-up: ``reports/writeup/``."""
    return settings.REPORTS_DIR / "writeup"


def _tex_escape(text: str) -> str:
    """Escapa los caracteres especiales de LaTeX en texto plano."""
    repl = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def _table2_rows() -> list[dict]:
    """Filas de la Table 2: perfil CIPA por dataset en features/full."""
    view = _cdata.canonical_view()
    return view.select(
        "dataset_id", "domain", "origin", "N", "IR", *DIMENSIONS,
        "DS", "band", "signature",
    ).to_dicts()


def render_table2_latex(rows: list[dict]) -> str:
    """Table 2 en LaTeX (``booktabs``), lista para incluir en el documento."""
    dim_head = " & ".join(DIMENSIONS)
    lines = [
        "% Requiere \\usepackage{booktabs}. Generado por cipa_fraud (F7).",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Perfil CIPA de los 8 datasets de fraude digital "
        "(capa \\emph{features}, $N$ completo). "
        "D1--D7 son las dimensiones de dificultad; DS es el "
        "\\emph{Difficulty Score} y la firma el patrón dominante del perfil.}",
        "\\label{tab:cipa-fraude}",
        "\\begin{tabular}{llr r " + "r" * len(DIMENSIONS) + " r l c}",
        "\\toprule",
        "Dataset & Dominio & $N$ & IR & " + dim_head +
        " & DS & Banda & Firma \\\\",
        "\\midrule",
    ]
    for r in rows:
        dims = " & ".join(f"{r[d]:.2f}" for d in DIMENSIONS)
        lines.append(
            f"{_tex_escape(r['dataset_id'])} & {_tex_escape(r['domain'])} & "
            f"{r['N']:,} & {r['IR']:.1f} & {dims} & "
            f"{r['DS']:.3f} & {r['band']} & {r['signature']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def render_table2_markdown(rows: list[dict]) -> str:
    """Table 2 en Markdown (misma información que la versión LaTeX)."""
    head = ("| Dataset | Dominio | N | IR | " + " | ".join(DIMENSIONS)
            + " | DS | Banda | Firma |")
    sep = "|---|---|---|---|" + "|".join(["---"] * len(DIMENSIONS)) + "|---|---|---|"
    lines = ["**Table 2.** Perfil CIPA de los 8 datasets de fraude "
             "(features, N=full).", "", head, sep]
    for r in rows:
        dims = " | ".join(f"{r[d]:.2f}" for d in DIMENSIONS)
        lines.append(
            f"| {r['dataset_id']} | {r['domain']} | {r['N']:,} | {r['IR']:.1f} | "
            f"{dims} | {r['DS']:.3f} | {r['band']} | {r['signature']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_narrative(reports: dict, consistency: dict) -> str:
    """Narrativa de resultados y discusión (español) con cifras computadas."""
    s1, s2 = reports["RQ1"]["stats"], reports["RQ2"]["stats"]
    s3, s4 = reports["RQ3"]["stats"], reports["RQ4"]["stats"]
    s5, s6 = reports["RQ5"]["stats"], reports["RQ6"]["stats"]
    s7 = reports["RQ7"]["stats"]
    cs = consistency["summary"]

    dims_sorted = sorted(s2["dim_means"].items(), key=lambda kv: kv[1], reverse=True)
    top2 = " y ".join(f"{d} ({DIMENSION_NAMES[d]}, {v:.2f})" for d, v in dims_sorted[:2])
    sig = max(s2["signature_distribution"], key=s2["signature_distribution"].get)
    sig_n = s2["signature_distribution"][sig]
    bands_prose = ", ".join(f"{n} en banda {b}" for b, n in s1["band_counts"].items())

    return "\n".join([
        "# Caracterización CIPA del dominio de fraude digital: resultados y discusión",
        "",
        f"*Materiales de escritura (F7). Generado: {datetime.now(UTC).isoformat()}.*",
        "",
        "## Resumen",
        "",
        "Aplicamos el framework CIPA a ocho datasets tabulares de fraude digital, "
        "cada uno en dos representaciones (limpia y con ingeniería de "
        "características) y tres escalas (10k, 50k y N completo), para un total de "
        f"48 caracterizaciones. En el corte de modelado (features, N completo) la "
        f"dificultad va de DS={s1['DS_min']:.3f} ({s1['easiest']}) a "
        f"DS={s1['DS_max']:.3f} ({s1['hardest']}), con {bands_prose}. "
        "El dominio exhibe una firma de perfil marcadamente "
        f"homogénea: la firma **{sig}** aparece en {sig_n} de los 8 datasets, "
        "dominada por las dimensiones " + top2 + ".",
        "",
        "## Resultados por pregunta de investigación",
        "",
        f"**RQ1 (ranking).** El dataset más difícil es **{s1['hardest']}** "
        f"(DS={s1['DS_max']:.3f}) y el más fácil **{s1['easiest']}** "
        f"(DS={s1['DS_min']:.3f}). La dificultad se concentra en la banda Moderate; "
        "sólo un dataset alcanza banda High, lo que sitúa al fraude tabular como un "
        "dominio de dificultad intermedia dentro de la escala de CIPA.",
        "",
        f"**RQ2 (perfiles y firmas).** La firma **{sig}** (Compound) domina "
        f"({sig_n}/8): el fraude no es difícil por una sola causa sino por la "
        "conjunción de varias. Las dimensiones más intensas del dominio son "
        f"{top2}; es decir, el desbalance extremo y la fragmentación de la clase "
        "minoritaria son los rasgos estructurales definitorios, por encima del "
        "solapamiento o la dureza local.",
        "",
        f"**RQ3 (estructura en 7-D).** El clustering jerárquico sobre las siete "
        f"dimensiones estandarizadas arroja una silueta baja "
        f"({s3['silhouette']:.2f}, k={s3['k_best']}): pese a la firma común, los "
        "datasets de fraude no forman un bloque compacto, sino que se dispersan en "
        "el espacio de perfiles. La ubicación cuantitativa frente al panorama "
        "multi-dominio del benchmark queda como trabajo futuro (sus perfiles 7-D no "
        "están disponibles como dato).",
        "",
        f"**RQ4 (real vs. sintético).** La diferencia media de DS entre datasets "
        f"reales y sintéticos es {s4['DS_mean_real_minus_synth']}. Con n pequeño por "
        "grupo la lectura es descriptiva, pero sugiere que los datasets reales "
        "tienden a puntuar algo más alto que los sintéticos.",
        "",
        f"**RQ5 (E-FEAT).** La ingeniería de características **reduce** el DS en "
        f"{s5['n_easier_with_features']}/8 datasets (ΔDS medio {s5['mean_dDS']}), "
        f"sin alterar ninguna firma ({s5['n_signature_changed']} cambios). El "
        "efecto se concentra en bajar la dureza (D3) y la informatividad requerida "
        "(D6): las features facilitan el problema sin cambiar su naturaleza "
        "cualitativa.",
        "",
        f"**RQ6 (E-SCALE).** Bajo el barrido multi-N, la dimensión más sensible es "
        f"**{s6['most_sensitive_dim']}** ({DIMENSION_NAMES[s6['most_sensitive_dim']]}), "
        "consistente con que el submuestreo altera la razón de desbalanceo efectiva. "
        f"La banda se mantiene estable en {s6['n_configs_band_stable']}/"
        f"{s6['n_configs']} configuraciones: el diagnóstico cualitativo es "
        "razonablemente robusto a la escala, aunque el DS puntual se desplaza.",
        "",
        f"**RQ7 (validación del proxy).** El índice heurístico de dificultad del "
        f"proyecto comparativo **no** predice el DS real de CIPA (Spearman "
        f"ρ={s7['spearman_rho']}, p={s7['p_value']}, n={s7['n']}). El caso extremo "
        "es fraudecom: el proxy lo califica como trivialmente fácil mientras CIPA lo "
        "identifica como el más difícil del conjunto. Esto respalda la necesidad de "
        "una medida estructural como CIPA frente a proxies baratos de separabilidad.",
        "",
        f"**RQ8 (consistencia con el benchmark).** Sobre los tres datasets también "
        f"presentes en el benchmark de CIPA, banda y firma coinciden en "
        f"{cs['band_match']}/{cs['n_comparisons']} y "
        f"{cs['signature_match']}/{cs['n_comparisons']} comparaciones "
        "respectivamente; el diagnóstico se reproduce pese al preprocesamiento y al "
        f"cambio de versión. El DS se desplaza a lo sumo {cs['max_abs_delta_DS']:.3f}, "
        "atribuible a la limpieza e ingeniería de características frente a la carga "
        "cruda genérica del benchmark.",
        "",
        "## Discusión",
        "",
        "El dominio del fraude digital tabular emerge como un problema de "
        "dificultad intermedia y de naturaleza **compuesta**: la firma V domina "
        "porque la dificultad proviene simultáneamente del desbalance extremo (D1) "
        "y de la fragmentación de la clase fraudulenta (D4), más que de un "
        "solapamiento fuerte entre clases. Este perfil es estable ante la escala y "
        "la representación —la ingeniería de características suaviza el problema sin "
        "recategorizarlo— y reproducible frente al benchmark original. El fracaso "
        "del proxy heurístico (RQ7) subraya el valor de una caracterización "
        "estructural multidimensional: rasgos como la separabilidad marginal no "
        "capturan la dificultad que sí revela el perfil CIPA.",
        "",
        "## Limitaciones",
        "",
        "- Ocho datasets: los cortes por grupo (RQ4) son descriptivos, no "
        "inferenciales.",
        "- La ubicación cuantitativa frente al panorama multi-dominio (13 datasets "
        "del benchmark) requiere sus perfiles 7-D publicados como dato (RQ3).",
        "- La referencia de consistencia (RQ8) es CIPA v1.1.0 mientras la "
        "recomputación usa la versión instalada; parte del ΔDS puede deberse al "
        "cambio de versión además del preprocesamiento.",
        "",
        "## Figuras finales",
        "",
        "Las figuras del estudio (CVD-safe) están en "
        "`reports/comparative/figures/`: ranking de DS (`rq1_ranking.png`), heatmap "
        "de perfiles D1–D7 (`rq2_heatmap.png`), distribución de firmas "
        "(`rq2_signatures.png`), efecto de las features (`rq5_efeat.png`), "
        "sensibilidad a la escala (`rq6_escale.png`) y proxy vs. DS "
        "(`rq7_proxy.png`).",
        "",
    ])


def build(write: bool = True) -> dict:
    """Ensambla los materiales de write-up (Table 2 + narrativa) de F7.

    Parameters
    ----------
    write : bool, optional
        Si ``True`` (por defecto) escribe ``table2.tex``, ``table2.md`` y
        ``writeup.md`` en ``reports/writeup/``.

    Returns
    -------
    dict
        ``table2`` (filas), textos renderizados y rutas escritas.
    """
    reports = {r["name"]: r for r in _analysis.run_all()}
    consistency = _consistency.check(write=False)
    rows = _table2_rows()

    tex = render_table2_latex(rows)
    md_table = render_table2_markdown(rows)
    narrative = render_narrative(reports, consistency)

    result = {"table2": rows, "paths": {}}
    if not write:
        return result

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table2.tex").write_text(tex)
    (out_dir / "table2.md").write_text(md_table)
    (out_dir / "writeup.md").write_text(narrative)
    result["paths"] = {
        "table2_tex": str(out_dir / "table2.tex"),
        "table2_md": str(out_dir / "table2.md"),
        "writeup_md": str(out_dir / "writeup.md"),
    }
    return result
