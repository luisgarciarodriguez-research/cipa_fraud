"""Análisis comparativo transversal de los resultados CIPA (RQ1–RQ7). Fase F5.

Este subpaquete consolida los resultados por corrida (dataset × capa × N) en el
estudio comparativo del proyecto. A partir de ``results/all_results.parquet``
produce las tablas y figuras que responden las preguntas de investigación:

- **RQ1** Ranking de dificultad (DS) y distribución de bandas entre datasets.
- **RQ2** Perfiles por dimensión (D1–D7) y distribución de firmas (I–V).
- **RQ3** Ubicación del dominio fraude frente al benchmark multi-dominio de CIPA
  (clustering en el espacio de 7 dimensiones).
- **RQ4** Contraste real vs. sintético y por tipo de fraude.
- **RQ5** Efecto de la ingeniería de features (delta clean ↔ features, E-FEAT).
- **RQ6** Sensibilidad del DS a la escala (barrido multi-N, E-SCALE).
- **RQ7** Validación del proxy heurístico previo contra el DS real (Spearman ρ).

Se implementa en la fase F5 (ver ``PLAN.md``).

--------------------------------------------------------------------------
Universidad Nacional Autónoma de México (UNAM)
Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
Programa de Posgrado en Ciencia e Ingeniería de la Computación (PCIC)

Autor:  Luis García Rodríguez  <luis.garcia@unam.edu>
Tutor:  José Antonio Neme Castillo  <antonio.neme@iimas.unam.mx>

Proyecto CIPA_FRAUD. Licencia: MIT — ver el archivo LICENSE.
--------------------------------------------------------------------------
"""
