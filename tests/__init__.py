"""Suite de pruebas de CIPA_FRAUD.

Cobertura mínima de red de seguridad para las piezas propias del proyecto: el
contrato ``(X, y)`` del adaptador, la orquestación reanudable del barrido, la lógica
de consistencia (RQ8), el análisis comparativo (RQ1–RQ7) y el manifiesto de
reproducibilidad. La mayoría de las pruebas son herméticas (sin datos externos ni
ejecución de CIPA); las que sí requieren los Parquet de origen se marcan con
``needs_data`` y se omiten automáticamente si faltan.
"""
