"""Reproducción vs. el benchmark original de CIPA (RQ8). Fase F4.

Verifica la consistencia de CIPA_FRAUD contra el benchmark publicado del framework
para los datasets con solapamiento: ``ulb_cc`` (CreditCard), ``paysim`` (PaySim)
e ``ieee_cis`` (IEEE-CIS), listados por :func:`cipa_fraud.registry.benchmark_overlap`.

Recomputa CIPA sobre estos datasets ya limpios y con ingeniería de features y
compara banda, firma y DS contra los valores del benchmark (CIPA v1.1.0, Tier-2 a
N=10k). Documenta cuánto desplaza el resultado el preprocesamiento del proyecto
comparativo frente a la carga cruda genérica del benchmark original.

Se implementa en la fase F4 (ver ``PLAN.md``).

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


def check() -> dict:  # pragma: no cover - F4
    """Compara banda/firma/DS de los datasets solapados contra el benchmark CIPA.

    Para cada dataset con ``cipa_benchmark_key`` recomputa CIPA (config Tier-2,
    N=10k) y contrasta el resultado con el valor de referencia publicado,
    reportando coincidencia de banda y firma y la diferencia de DS.

    Returns
    -------
    dict
        Reporte de consistencia por dataset: valores recomputados, valores de
        referencia, coincidencia de banda/firma y ``ΔDS``.

    Raises
    ------
    NotImplementedError
        Hasta que se implemente en la fase F4 (ver ``PLAN.md``).
    """
    raise NotImplementedError("consistency.check() se implementa en F4 (ver PLAN.md).")
