"""Análisis comparativo transversal: una función por pregunta (RQ1–RQ7). Fase F5.

Cada ``rq*`` consume el consolidado tidy (vía :mod:`cipa_fraud.compare.data`) y
devuelve un dict uniforme ``{"name", "question", "table", "stats", "note"}`` donde
``table`` es un ``polars.DataFrame`` y ``stats``/``note`` capturan los escalares y
las salvedades de interpretación. Esta forma homogénea permite a
:mod:`cipa_fraud.compare.build` serializar y narrar los resultados sin lógica
específica por RQ.

Cortes canónicos: los transversales (RQ1–RQ4, RQ7) se anclan en la capa
``features`` a N ``full`` (representación de modelado a escala real); E-FEAT (RQ5)
contrasta ``clean`` vs ``features`` a ``full`` y E-SCALE (RQ6) recorre el barrido
multi-N.

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

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from cipa_fraud.compare import data as _data
from cipa_fraud.compare.data import CANONICAL_LAYER, CANONICAL_N, DIMENSIONS


def rq1_ranking(results: pl.DataFrame | None = None) -> dict:
    """RQ1 — Ranking de dificultad de los 8 datasets por DS (features/full)."""
    view = _data.canonical_view(results)
    table = view.select(
        "dataset_id", "domain", "origin", "N", "IR", "DS", "band",
        "signature", "signature_name",
    ).with_row_index("rank", offset=1)
    bands = (
        table.group_by("band").len().sort("len", descending=True)
        .to_dict(as_series=False)
    )
    return {
        "name": "RQ1",
        "question": "Ranking de dificultad por DS",
        "table": table,
        "stats": {
            "layer": CANONICAL_LAYER,
            "n_target": CANONICAL_N,
            "DS_min": round(float(table["DS"].min()), 4),
            "DS_max": round(float(table["DS"].max()), 4),
            "hardest": table["dataset_id"][0],
            "easiest": table["dataset_id"][-1],
            "band_counts": dict(zip(bands["band"], bands["len"], strict=True)),
        },
        "note": "Corte canónico features/full; el DS depende de la escala (ver RQ6).",
    }


def rq2_profiles(results: pl.DataFrame | None = None) -> dict:
    """RQ2 — Perfiles D1–D7 y distribución de firmas (features/full)."""
    view = _data.canonical_view(results)
    table = view.select("dataset_id", "signature", "signature_name", *DIMENSIONS)

    # Dimensión dominante por dataset y media por dimensión (perfil del dominio).
    dim_means = {d: round(float(view[d].mean()), 4) for d in DIMENSIONS}
    dominant = (
        table.select("dataset_id", *DIMENSIONS)
        .unpivot(index="dataset_id", variable_name="dim", value_name="value")
        .sort("value", descending=True)
        .group_by("dataset_id", maintain_order=True)
        .first()
        .rename({"dim": "dominant_dim", "value": "dominant_value"})
    )
    table = table.join(dominant, on="dataset_id", how="left")

    sig_dist = (
        view.group_by("signature", "signature_name").len()
        .sort("len", descending=True).rename({"len": "count"})
    )
    top_dim = max(dim_means, key=dim_means.get)
    return {
        "name": "RQ2",
        "question": "Perfiles y firmas dominantes",
        "table": table,
        "stats": {
            "dim_means": dim_means,
            "dimension_most_intense": top_dim,
            "signature_distribution": dict(
                zip(sig_dist["signature"], sig_dist["count"], strict=True)
            ),
        },
        "note": (
            "dim_means = intensidad media por dimensión en el dominio fraude; "
            f"la más intensa es {top_dim} ({_data.DIMENSION_NAMES[top_dim]})."
        ),
    }


def rq3_cluster(results: pl.DataFrame | None = None) -> dict:
    """RQ3 — ¿Forma el fraude un cluster reconocible en el espacio 7-D?

    Clustering jerárquico (Ward) sobre las 7 dimensiones estandarizadas de los 8
    datasets, eligiendo k por silueta. La comparación cuantitativa contra los 13
    datasets del benchmark multi-dominio queda pendiente: sus perfiles 7-D no están
    disponibles como dato en este proyecto (ver ``note``).
    """
    view = _data.canonical_view(results)
    X = view.select(*DIMENSIONS).to_numpy()
    Xz = StandardScaler().fit_transform(X)

    best = {"k": None, "silhouette": -1.0, "labels": None}
    for k in (2, 3, 4):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xz)
        sil = float(silhouette_score(Xz, labels))
        if sil > best["silhouette"]:
            best = {"k": k, "silhouette": round(sil, 4), "labels": labels}

    table = view.select("dataset_id", "domain", "origin", "signature", *DIMENSIONS)
    table = table.with_columns(pl.Series("cluster", best["labels"]))
    return {
        "name": "RQ3",
        "question": "El fraude en el espacio 7-D (clustering)",
        "table": table.sort("cluster"),
        "stats": {
            "k_best": best["k"],
            "silhouette": best["silhouette"],
            "cluster_sizes": {
                int(c): int(n) for c, n in
                zip(*np.unique(best["labels"], return_counts=True), strict=True)
            },
        },
        "note": (
            "Clustering interno de los 8 datasets de fraude. La ubicación frente al "
            "panorama multi-dominio (13 datasets del benchmark) requiere sus "
            "perfiles 7-D publicados, no disponibles como dato en este proyecto; "
            "queda como comparación cualitativa vía RQ8 (3 puntos de fraude)."
        ),
    }


def rq4_groups(results: pl.DataFrame | None = None) -> dict:
    """RQ4 — Perfiles CIPA por origen (real/sintético) y por dominio de fraude."""
    view = _data.canonical_view(results)
    agg = [pl.len().alias("n_datasets"), pl.col("DS").mean().round(4).alias("DS_mean")]
    agg += [pl.col(d).mean().round(4).alias(d) for d in DIMENSIONS]
    by_origin = view.group_by("origin").agg(agg).sort("DS_mean", descending=True)
    by_domain = view.group_by("domain").agg(agg).sort("DS_mean", descending=True)

    real = by_origin.filter(pl.col("origin") == "real")
    synth = by_origin.filter(pl.col("origin") == "synthetic")
    ds_gap = None
    if len(real) and len(synth):
        ds_gap = round(float(real["DS_mean"][0] - synth["DS_mean"][0]), 4)
    return {
        "name": "RQ4",
        "question": "Real vs. sintético / tipo de fraude",
        "table": by_origin,
        "stats": {
            "by_domain": by_domain.to_dicts(),
            "DS_mean_real_minus_synth": ds_gap,
        },
        "note": (
            "Medias por grupo en el corte features/full. n por grupo es pequeño "
            "(8 datasets): leer como descriptivo, no inferencial."
        ),
    }


def rq5_efeat(results: pl.DataFrame | None = None) -> dict:
    """RQ5 (E-FEAT) — Efecto de la ingeniería de features: Δ(features − clean)."""
    res = results if results is not None else _data.load_results()
    full = res.filter(pl.col("n_target") == CANONICAL_N)
    clean = full.filter(pl.col("layer") == "clean")
    feats = full.filter(pl.col("layer") == "features")

    keep = ["dataset_id", "DS", "band", "signature", *DIMENSIONS]
    j = clean.select(keep).join(
        feats.select(keep), on="dataset_id", suffix="_feat"
    )
    deltas = [
        (pl.col(f"{c}_feat") - pl.col(c)).round(4).alias(f"d{c}")
        for c in ("DS", *DIMENSIONS)
    ]
    table = j.with_columns(
        *deltas,
        (pl.col("band") != pl.col("band_feat")).alias("band_changed"),
        (pl.col("signature") != pl.col("signature_feat")).alias("sig_changed"),
    ).select(
        "dataset_id", "DS", "DS_feat", "dDS",
        *[f"d{d}" for d in DIMENSIONS], "band_changed", "sig_changed",
    ).sort("dDS")

    mean_dDS = round(float(table["dDS"].mean()), 4)
    dim_shift = {d: round(float(table[f"d{d}"].mean()), 4) for d in DIMENSIONS}
    return {
        "name": "RQ5",
        "question": "Efecto de la ingeniería de features (E-FEAT)",
        "table": table,
        "stats": {
            "mean_dDS": mean_dDS,
            "n_easier_with_features": int((table["dDS"] < 0).sum()),
            "n_harder_with_features": int((table["dDS"] > 0).sum()),
            "mean_dim_shift": dim_shift,
            "n_signature_changed": int(table["sig_changed"].sum()),
        },
        "note": "Δ = features − clean a N=full; negativo = features baja el DS.",
    }


def rq6_escale(results: pl.DataFrame | None = None) -> dict:
    """RQ6 (E-SCALE) — Estabilidad de DS/banda/firma bajo el barrido multi-N."""
    res = results if results is not None else _data.load_results()

    # Estabilidad por dataset × capa a lo largo de {10k, 50k, full}.
    per = (
        res.group_by("dataset_id", "layer")
        .agg(
            pl.col("DS").std().round(4).alias("DS_std"),
            (pl.col("DS").max() - pl.col("DS").min()).round(4).alias("DS_range"),
            pl.col("band").n_unique().alias("n_bands"),
            pl.col("signature").n_unique().alias("n_signatures"),
        )
        .sort("DS_range", descending=True)
    )
    # Sensibilidad por dimensión: desviación media (entre N) por dataset×capa.
    dim_sens = {}
    for d in DIMENSIONS:
        s = (
            res.group_by("dataset_id", "layer").agg(pl.col(d).std().alias("s"))
        )["s"].mean()
        dim_sens[d] = round(float(s), 4)
    most = max(dim_sens, key=dim_sens.get)
    return {
        "name": "RQ6",
        "question": "Robustez a la escala (E-SCALE)",
        "table": per,
        "stats": {
            "dim_sensitivity": dim_sens,
            "most_sensitive_dim": most,
            "n_configs_band_stable": int((per["n_bands"] == 1).sum()),
            "n_configs_sig_stable": int((per["n_signatures"] == 1).sum()),
            "n_configs": len(per),
        },
        "note": (
            f"Dimensión más sensible al submuestreo: {most} "
            f"({_data.DIMENSION_NAMES[most]}); D1 es sensible vía IR_eff."
        ),
    }


def rq7_proxy(results: pl.DataFrame | None = None) -> dict:
    """RQ7 — Validación del proxy: Spearman ρ entre el índice heurístico y el DS."""
    view = _data.canonical_view(results).select("dataset_id", "DS")
    proxy = _data.load_proxy()
    joined = view.join(proxy, on="dataset_id", how="inner").sort(
        "DS", descending=True
    )
    stats: dict = {"n": len(joined), "layer": CANONICAL_LAYER, "n_target": CANONICAL_N}
    if len(joined) >= 3:
        rho, p = spearmanr(joined["proxy"].to_numpy(), joined["DS"].to_numpy())
        stats["spearman_rho"] = round(float(rho), 4)
        stats["p_value"] = round(float(p), 4)
    else:
        stats["spearman_rho"] = None
        stats["p_value"] = None
    return {
        "name": "RQ7",
        "question": "Validación del proxy de dificultad",
        "table": joined,
        "stats": stats,
        "note": (
            "proxy = índice heurístico del proyecto comparativo (separabilidad, "
            "faltantes, IR…); DS = medida real de CIPA. ρ alto ⇒ el proxy ordena "
            "bien la dificultad."
        ),
    }


#: Todas las RQ transversales de F5, en orden.
ALL_RQS = (
    rq1_ranking, rq2_profiles, rq3_cluster, rq4_groups,
    rq5_efeat, rq6_escale, rq7_proxy,
)


def run_all(results: pl.DataFrame | None = None) -> list[dict]:
    """Ejecuta las siete RQ transversales y devuelve sus reportes en orden."""
    res = results if results is not None else _data.load_results()
    return [rq(res) for rq in ALL_RQS]
