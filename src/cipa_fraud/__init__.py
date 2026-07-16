"""CIPA_FRAUD — ejecución del framework CIPA sobre datasets de fraude digital.

Paquete de investigación que ejecuta de punta a punta el framework **CIPA**
(*Characterization, Indexing, Profiling, Action*) sobre un conjunto extendido de
conjuntos de datos públicos de fraude digital, y realiza un estudio comparativo
transversal de su dificultad estructural de aprendizaje.

Para cada dataset tabular se obtiene, sin entrenar ningún modelo: las siete
dimensiones de complejidad (D1–D7), el *Difficulty Score* (DS ∈ [0, 1]) con su
banda, la *Complexity Signature* (I–V) y las recomendaciones de acción. El estudio
consolida ranking de dificultad, perfiles por dimensión, distribución de firmas y
la ubicación del dominio fraude dentro del panorama multi-dominio de CIPA.

Los datos no se copian: se leen in situ del proyecto comparativo de fraude (ver
:mod:`cipa_fraud.settings`). El framework CIPA se consume como dependencia
editable local. El plan completo y las fases (F0–F7) están en ``PLAN.md``.

Submódulos
----------
- :mod:`cipa_fraud.settings`     Rutas externas, semilla y parámetros de escala.
- :mod:`cipa_fraud.registry`     Metadatos de los 8 datasets tabulares del estudio.
- :mod:`cipa_fraud.adapt`        Parquet(clean|features) → (X, y) para CIPA.
- :mod:`cipa_fraud.run`          Orquestación de CIPAPipeline + manifiesto.
- :mod:`cipa_fraud.consistency`  Reproducción vs. el benchmark original de CIPA.
- :mod:`cipa_fraud.compare`      Análisis comparativo transversal (RQ1–RQ7).
- :mod:`cipa_fraud.report`       Generación de reportes HTML/Markdown y figuras.
- :mod:`cipa_fraud.cli`          Interfaz de línea de comandos (``cipa-fraud``).

--------------------------------------------------------------------------
Universidad Nacional Autónoma de México (UNAM)
Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
Programa de Posgrado en Ciencia e Ingeniería de la Computación (PCIC)

Autor:  Luis García Rodríguez  <luis.garcia@unam.edu>
Tutor:  José Antonio Neme Castillo  <antonio.neme@iimas.unam.mx>

Proyecto CIPA_FRAUD. Licencia: MIT — ver el archivo LICENSE.
--------------------------------------------------------------------------
"""

__version__ = "0.1.0"
