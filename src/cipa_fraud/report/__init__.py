"""Generación de reportes HTML/Markdown y figuras del estudio. Fase F6.

Este subpaquete toma los resultados consolidados y los productos del análisis
comparativo (:mod:`cipa_fraud.compare`) y genera el informe final del estudio:

- Informe comparativo en HTML y Markdown (panorama general + ficha por dataset +
  hallazgos por pregunta de investigación).
- Figuras: heatmap de dimensiones D1–D7, radar por dataset, ranking de DS,
  distribución de firmas, dispersión de escala y correlación DS ↔ proxy.
- ``reports/reproducibility.json`` con versiones, hashes de entrada, semillas y
  parámetros de cada corrida.

Se implementa en la fase F6 (ver ``PLAN.md``).

--------------------------------------------------------------------------
Universidad Nacional Autónoma de México (UNAM)
Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
Programa de Posgrado en Ciencia e Ingeniería de la Computación (PCIC)

Autor:  Luis García Rodríguez  <luis.garcia@unam.edu>
Tutor:  José Antonio Neme Castillo  <antonio.neme@iimas.unam.mx>

Proyecto CIPA_FRAUD. Licencia: MIT — ver el archivo LICENSE.
--------------------------------------------------------------------------
"""
