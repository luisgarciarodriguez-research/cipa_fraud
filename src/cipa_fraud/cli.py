"""Interfaz de línea de comandos de CIPA_FRAUD (``cipa-fraud``).

Expone el flujo del proyecto como subcomandos de una app ``typer``. En la fase F0
están operativos ``doctor`` (diagnóstico de entorno) y ``list-datasets``
(inventario del estudio); el resto (``adapt``, ``run``, ``run-all``,
``consistency``, ``compare``, ``report``) queda como stub hasta su fase
correspondiente (ver ``PLAN.md``).

El *entry point* ``cipa-fraud`` se declara en ``pyproject.toml`` y apunta a
:data:`app`. Ejecutar siempre con el entorno virtual del proyecto activado.

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

import typer

from cipa_fraud import __version__, registry, settings

#: Aplicación ``typer`` que agrupa los subcomandos; expuesta como *entry point*.
app = typer.Typer(
    add_completion=False,
    help="Ejecucion del framework CIPA sobre datasets de fraude digital.",
    no_args_is_help=True,
)


@app.command()
def doctor() -> None:
    """Diagnostica el entorno: rutas externas, import de CIPA y datos disponibles.

    Verifica que existan el proyecto comparativo y la raíz de CIPA, que el paquete
    ``cipa`` sea importable (mostrando su versión) y que cada dataset del estudio
    tenga su Parquet en ambas capas (``clean`` y ``features``). Es la comprobación
    de salud a correr tras instalar el proyecto.

    Termina con código de salida 0 si todo está en orden y 1 si falta alguna ruta,
    el import de CIPA o algún Parquet, para poder encadenarlo en scripts.
    """
    typer.echo(f"cipa-fraud v{__version__}")
    ok = True

    problems = settings.check_paths()
    if problems:
        ok = False
        typer.secho("Rutas externas:", fg=typer.colors.RED)
        for p in problems:
            typer.echo(f"  [FALTA] {p}")
    else:
        typer.secho("Rutas externas: OK", fg=typer.colors.GREEN)
        typer.echo(f"  comparativo: {settings.PROCESSED_ROOT}")
        typer.echo(f"  cipa       : {settings.CIPA_ROOT}")

    try:
        import cipa

        typer.secho(f"CIPA import: OK (v{cipa.__version__})", fg=typer.colors.GREEN)
    except ImportError as exc:  # pragma: no cover
        ok = False
        typer.secho(f"CIPA import: FALLA — {exc}", fg=typer.colors.RED)
        typer.echo(
            "  Instala: pip install -e "
            "/home/luisgarcia/projects/unam/dcic/2026-2/statistical_analysis"
        )

    # Presencia de los Parquet procesados por dataset y capa.
    typer.echo("Datos procesados por dataset:")
    for ds_id in registry.all_ids():
        marks = []
        for layer in settings.LAYERS:
            exists = settings.processed_path(ds_id, layer).exists()
            marks.append(f"{layer}={'OK' if exists else 'FALTA'}")
            ok = ok and exists
        typer.echo(f"  {ds_id:10s} {'  '.join(marks)}")

    raise typer.Exit(code=0 if ok else 1)


@app.command("list-datasets")
def list_datasets() -> None:
    """Lista los 8 datasets tabulares del estudio y sus metadatos.

    Imprime una tabla con identificador, dominio de fraude, origen (real/
    sintético), número de registros, tasa de fraude y la clave equivalente en el
    benchmark original de CIPA (si el dataset forma parte de él).
    """
    typer.echo(f"{'id':10s} {'dominio':18s} {'origen':10s} {'N':>11s} "
               f"{'fraude%':>8s}  cipa-bench")
    for ds_id in registry.all_ids():
        s = registry.get(ds_id)
        typer.echo(
            f"{s.id:10s} {s.domain.value:18s} {s.origin.value:10s} "
            f"{s.n_records:>11,d} {s.fraud_rate * 100:>7.3f}%  "
            f"{s.cipa_benchmark_key or '-'}"
        )


def _pending(fase: str) -> None:
    """Aborta el comando indicando en qué fase se implementará (código 2).

    Utilidad interna para los subcomandos aún no implementados: emite un aviso
    consistente y sale con código 2 (distinto de 0/1 de ``doctor``) para
    diferenciar "pendiente" de "error".

    Parameters
    ----------
    fase : str
        Etiqueta de la fase del plan que implementará el comando (p. ej. ``"F1"``).

    Raises
    ------
    typer.Exit
        Siempre, con ``code=2``.
    """
    typer.secho(f"Pendiente: se implementa en la fase {fase} (ver PLAN.md).",
                fg=typer.colors.YELLOW)
    raise typer.Exit(code=2)


@app.command()
def adapt(dataset: str, layer: str = "features", n: str = "full") -> None:
    """Inspecciona la matriz ``(X, y)`` construida para un dataset/capa. (F1)

    Construye el par ``(X, y)`` con :func:`cipa_fraud.adapt.adapt` y muestra el
    manifiesto: forma de ``X``, número de clases, razones de desbalanceo (original
    y efectiva), modo de submuestreo y columnas descartadas/imputadas. No ejecuta
    el pipeline CIPA.

    Parameters
    ----------
    dataset : str
        Identificador del dataset (p. ej. ``ulb_cc``).
    layer : str, optional
        Capa de entrada: ``clean`` o ``features`` (por defecto ``features``).
    n : str, optional
        Punto de escala: un entero como texto o ``full`` (por defecto ``full``).
    """
    from cipa_fraud.adapt import adapt as _adapt

    res = _adapt(dataset, layer, n)
    m = res.manifest
    typer.secho(f"{dataset} / {layer} / n={m['n_target']}", fg=typer.colors.GREEN)
    typer.echo(f"  X: {res.X.shape}  float64  finito=OK")
    typer.echo(f"  y: 2 clases  minoría(fraude)={m['n_minority']:,}  "
               f"mayoría={m['n_majority']:,}")
    typer.echo(f"  submuestreo: {m['subsample_mode']}  "
               f"(fuente={m['source_rows']:,} filas)")
    typer.echo(f"  IR: {m['IR']:.1f}  →  IR_eff: {m['IR_eff']:.1f}  "
               f"(tasa fraude efectiva {m['fraud_rate_eff'] * 100:.3f}%)")
    typer.echo(f"  features: {m['n_features']}")
    if m["dropped_categorical"]:
        typer.echo(f"  categóricas descartadas ({len(m['dropped_categorical'])}): "
                   f"{m['dropped_categorical'][:8]}"
                   + (" …" if len(m["dropped_categorical"]) > 8 else ""))
    if m["dropped_high_null"]:
        typer.echo(f"  descartadas por nulos>50% ({len(m['dropped_high_null'])}): "
                   f"{m['dropped_high_null'][:8]}"
                   + (" …" if len(m["dropped_high_null"]) > 8 else ""))
    if m["imputed"]:
        typer.echo(f"  imputadas (mediana) ({len(m['imputed'])}): "
                   f"{m['imputed'][:8]}" + (" …" if len(m["imputed"]) > 8 else ""))


@app.command()
def run(dataset: str, layer: str = "both", sweep: str = "10000,50000,full") -> None:
    """Ejecuta CIPA sobre un dataset (capas × barrido de N). (F2/F3)

    Corre el pipeline para cada combinación de capa y tamaño del barrido,
    persistiendo un JSON por corrida en ``results/<id>/<layer>/<N>.json`` y
    mostrando una línea de resumen (DS, banda, firma, tiempo) por combinación.

    Parameters
    ----------
    dataset : str
        Identificador del dataset.
    layer : str, optional
        ``clean``, ``features`` o ``both`` (por defecto ``both``).
    sweep : str, optional
        Lista separada por comas de puntos de escala (por defecto
        ``"10000,50000,full"``).
    """
    from cipa_fraud.run import run_one, summary_row

    layers = list(settings.LAYERS) if layer == "both" else [layer]
    points: list[str | int] = [
        p if p == "full" else int(p) for p in (s.strip() for s in sweep.split(","))
    ]
    typer.echo(f"{'layer':9s} {'N':>7s} {'mode':10s} {'DS':>6s} {'band':9s} "
               f"{'sig':4s} {'t(s)':>7s}")
    for lyr in layers:
        for n in points:
            out = run_one(dataset, lyr, n)
            r = summary_row(out)
            n_lbl = "full" if r["n_target"] == "full" else f"{r['N']:,}"
            typer.echo(
                f"{r['layer']:9s} {n_lbl:>7s} {r['subsample_mode']:10s} "
                f"{r['DS']:.3f}  {r['band']:9s} {r['signature']:4s} "
                f"{r['runtime_s']:>7.2f}"
            )


@app.command("run-all")
def run_all() -> None:
    """Ejecuta CIPA sobre los 8 datasets × 2 capas × 3 escalas. (F3)

    Barrido completo del estudio. Persiste el JSON de cada corrida y consolida el
    resultado tidy en ``results/all_results.parquet``.
    """
    _pending("F3")


@app.command()
def consistency() -> None:
    """Reproduce ulb_cc/paysim/ieee_cis vs. el benchmark original de CIPA. (F4)

    Ejecuta el *check* de consistencia (RQ8) y escribe su reporte en
    ``reports/consistency/``.
    """
    _pending("F4")


@app.command()
def compare() -> None:
    """Genera el análisis comparativo transversal (RQ1–RQ7). (F5)

    Consolida los resultados en tablas y figuras (ranking, perfiles, firmas,
    clustering, deltas de features y escala, correlación con el proxy).
    """
    _pending("F5")


@app.command()
def report() -> None:
    """Genera el informe comparativo HTML/Markdown y el manifiesto. (F6)

    Ensambla el informe final del estudio y ``reports/reproducibility.json`` a
    partir de los productos de :mod:`cipa_fraud.compare`.
    """
    _pending("F6")


if __name__ == "__main__":
    app()
