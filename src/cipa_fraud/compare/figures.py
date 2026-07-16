"""Figuras del análisis comparativo (RQ1–RQ7). Fase F5.

Genera las figuras estáticas del estudio a partir de los reportes de
:mod:`cipa_fraud.compare.analysis`. Decisiones de diseño (siguiendo el método
forma→color→validar):

- **Categórico** (bandas, capas): paleta **Okabe-Ito**, conjunto publicado y seguro
  para daltonismo, estándar en figuras científicas.
- **Secuencial** (heatmap D1–D7, magnitud 0–1): **cividis**, perceptualmente
  uniforme y CVD-safe.
- **Divergente** (E-FEAT, Δ centrado en 0): **PuOr** con punto medio neutro.
- Marcas finas, rejilla recesiva, texto en gris tinta (no negro puro), etiquetas
  directas cuando hay pocas series y leyenda cuando hay ≥2.

Todas las figuras se guardan como PNG en ``reports/comparative/figures/``.

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

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display (headless)
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from cipa_fraud.compare import data as _data
from cipa_fraud.compare.data import DIMENSION_NAMES, DIMENSIONS

#: Paleta categórica Okabe-Ito (CVD-safe).
OKABE_ITO = {
    "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}
#: Color por banda de dificultad (ordinal Low→Extreme).
BAND_COLOR = {
    "Low": OKABE_ITO["skyblue"], "Moderate": OKABE_ITO["blue"],
    "High": OKABE_ITO["vermillion"], "Extreme": OKABE_ITO["black"],
}
#: Color por capa de entrada.
LAYER_COLOR = {"clean": OKABE_ITO["skyblue"], "features": OKABE_ITO["vermillion"]}

_INK = "#222222"
_MUTED = "#666666"
_GRID = "#DDDDDD"


def _style() -> None:
    """Aplica un estilo base sobrio y legible a matplotlib."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.edgecolor": _MUTED,
        "axes.labelcolor": _INK,
        "axes.titlecolor": _INK,
        "text.color": _INK,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "axes.grid": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Guarda la figura como PNG ajustada y la cierra; devuelve la ruta."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_rq1_ranking(rq1: dict, out_dir: Path) -> Path:
    """Barras horizontales del ranking de DS, coloreadas por banda."""
    _style()
    t = rq1["table"].sort("DS")
    ids = t["dataset_id"].to_list()
    ds = t["DS"].to_numpy()
    colors = [BAND_COLOR[b] for b in t["band"].to_list()]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(ids, ds, color=colors, height=0.66)
    for y, v in enumerate(ds):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=8, color=_INK)
    ax.set_xlabel("Difficulty Score (DS)")
    ax.set_title("RQ1 — Ranking de dificultad CIPA (features, N=full)")
    ax.set_xlim(0, max(ds) * 1.15)
    ax.grid(axis="y", visible=False)
    seen: dict[str, None] = {}
    for b in t["band"].to_list():
        seen.setdefault(b, None)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND_COLOR[b]) for b in seen]
    ax.legend(handles, list(seen), title="Banda", frameon=False, loc="lower right")
    return _save(fig, out_dir, "rq1_ranking.png")


def fig_rq2_heatmap(rq2: dict, out_dir: Path) -> Path:
    """Heatmap datasets × D1–D7 (cividis, secuencial 0–1)."""
    _style()
    t = rq2["table"]
    ids = t["dataset_id"].to_list()
    M = t.select(*DIMENSIONS).to_numpy()

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels([f"{d}\n{DIMENSION_NAMES[d]}" for d in DIMENSIONS], fontsize=7.5)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids)
    ax.grid(visible=False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.55 else "black")
    ax.set_title("RQ2 — Perfiles dimensionales D1–D7 (features, N=full)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="intensidad")
    return _save(fig, out_dir, "rq2_heatmap.png")


def fig_rq2_signatures(rq2: dict, out_dir: Path) -> Path:
    """Distribución de firmas del dominio fraude."""
    _style()
    dist = rq2["stats"]["signature_distribution"]
    labels, counts = list(dist.keys()), list(dist.values())
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(labels, counts, color=OKABE_ITO["green"], width=0.6)
    for x, c in enumerate(counts):
        ax.text(x, c + 0.05, str(c), ha="center", fontsize=9, color=_INK)
    ax.set_ylabel("nº de datasets")
    ax.set_xlabel("firma CIPA")
    ax.set_title("RQ2 — Distribución de firmas (features, N=full)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(counts) * 1.2)
    return _save(fig, out_dir, "rq2_signatures.png")


def fig_rq5_efeat(rq5: dict, out_dir: Path) -> Path:
    """Barras divergentes de ΔDS (features − clean) por dataset."""
    _style()
    t = rq5["table"].sort("dDS")
    ids = t["dataset_id"].to_list()
    d = t["dDS"].to_numpy()
    lim = max(0.02, float(np.abs(d).max()) * 1.15)
    cmap = matplotlib.colormaps["PuOr"]
    colors = [cmap(0.5 + 0.5 * (v / lim)) for v in d]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(ids, d, color=colors, height=0.66)
    ax.axvline(0, color=_MUTED, linewidth=1)
    for y, v in enumerate(d):
        ax.text(v + (0.002 if v >= 0 else -0.002), y, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=_INK)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("ΔDS = features − clean   (← más fácil | más difícil →)")
    ax.set_title("RQ5 (E-FEAT) — Efecto de la ingeniería de features (N=full)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "rq5_efeat.png")


def fig_rq6_escale(out_dir: Path) -> Path:
    """Small multiples: DS vs N por dataset (líneas clean/features)."""
    _style()
    res = _data.load_results()
    order = ["10000", "50000", "full"]
    ids = _data.canonical_view(res)["dataset_id"].to_list()  # orden por DS
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True, sharey=True)
    for ax, ds_id in zip(axes.ravel(), ids, strict=True):
        for layer in ("clean", "features"):
            sub = res.filter(
                (pl.col("dataset_id") == ds_id) & (pl.col("layer") == layer)
            )
            xs = [order.index(n) for n in sub["n_target"].to_list()]
            pairs = sorted(zip(xs, sub["DS"].to_list(), strict=True))
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                    marker="o", markersize=5, linewidth=2,
                    color=LAYER_COLOR[layer], label=layer)
        ax.set_title(ds_id, fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["10k", "50k", "full"], fontsize=8)
        ax.set_ylim(0, 0.7)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.supylabel("Difficulty Score (DS)")
    fig.suptitle("RQ6 (E-SCALE) — DS a lo largo del barrido multi-N", y=0.98)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    return _save(fig, out_dir, "rq6_escale.png")


def fig_rq7_proxy(rq7: dict, out_dir: Path) -> Path:
    """Dispersión proxy vs DS con anotación de Spearman ρ."""
    _style()
    t = rq7["table"]
    if t.is_empty():
        return out_dir / "rq7_proxy.png"  # sin proxy disponible
    proxy = t["proxy"].to_numpy()
    ds = t["DS"].to_numpy()
    ids = t["dataset_id"].to_list()

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(proxy, ds, s=70, color=OKABE_ITO["blue"], zorder=3,
               edgecolor="white", linewidth=0.8)
    for x, y, name in zip(proxy, ds, ids, strict=True):
        ax.annotate(name, (x, y), fontsize=8, color=_MUTED,
                    xytext=(5, 3), textcoords="offset points")
    rho = rq7["stats"].get("spearman_rho")
    p = rq7["stats"].get("p_value")
    if rho is not None:
        ax.text(0.97, 0.05, f"Spearman ρ = {rho:.3f}\np = {p:.3f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
                color=_INK, bbox={"boxstyle": "round", "fc": "white", "ec": _GRID})
    ax.set_xlabel("proxy de dificultad (proyecto comparativo)")
    ax.set_ylabel("DS real (CIPA, features/full)")
    ax.set_title("RQ7 — Validación del proxy vs. DS")
    return _save(fig, out_dir, "rq7_proxy.png")


def build_all(reports: dict[str, dict], out_dir: Path) -> list[Path]:
    """Genera todas las figuras a partir de los reportes indexados por nombre.

    Parameters
    ----------
    reports : dict[str, dict]
        Reportes de :mod:`cipa_fraud.compare.analysis` indexados por ``name``
        (``"RQ1"``…``"RQ7"``).
    out_dir : pathlib.Path
        Directorio de salida de las figuras.

    Returns
    -------
    list[pathlib.Path]
        Rutas de las figuras generadas.
    """
    paths = [
        fig_rq1_ranking(reports["RQ1"], out_dir),
        fig_rq2_heatmap(reports["RQ2"], out_dir),
        fig_rq2_signatures(reports["RQ2"], out_dir),
        fig_rq5_efeat(reports["RQ5"], out_dir),
        fig_rq6_escale(out_dir),
        fig_rq7_proxy(reports["RQ7"], out_dir),
    ]
    return [p for p in paths if p.exists()]
