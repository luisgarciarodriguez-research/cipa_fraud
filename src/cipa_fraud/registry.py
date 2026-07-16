"""Registro de los 8 datasets tabulares de fraude del estudio CIPA_FRAUD.

Catálogo de metadatos para el estudio comparativo (dominio, origen real/sintético,
escala y solapamiento con el benchmark original de CIPA). Deliberadamente **no**
contiene el *target* ni los roles de columna: esos se leen en tiempo de ejecución
del ``_roles.json`` del proyecto comparativo (fuente de verdad), vía
:func:`cipa_fraud.settings.roles_path`. Así el registro describe *qué es* cada
dataset para el análisis, sin duplicar (y arriesgar desincronizar) el contrato de
datos.

Alcance del estudio: 8 datasets tabulares. Excluidos por estructura de grafo
(fuera del dominio tabular-binario de CIPA, dejados como trabajo futuro):
``elliptic``, ``elliptic_pp``, ``amlworld``.

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

from enum import Enum

from pydantic import BaseModel


class Origin(str, Enum):
    """Procedencia de los datos de un dataset.

    Distingue datos de transacciones reales de datos generados por simuladores.
    Es una variable de contraste del estudio (RQ4): ¿difieren sistemáticamente los
    perfiles CIPA entre datasets reales y sintéticos?

    Attributes
    ----------
    REAL : str
        Datos de transacciones reales (posiblemente anonimizados/transformados).
    SYNTHETIC : str
        Datos generados por un simulador o modelo generativo.
    """

    REAL = "real"
    SYNTHETIC = "synthetic"


class FraudDomain(str, Enum):
    """Tipo/dominio de fraude que representa el dataset.

    Agrupa los datasets por modalidad de fraude para el análisis por dominio
    (RQ4). Cada valor corresponde a una familia de transacciones con dinámicas de
    fraude distintas.

    Attributes
    ----------
    CARD_PRESENT : str
        Fraude con tarjeta presente (p. ej. ULB Credit Card).
    CARD_NOT_PRESENT : str
        Fraude sin tarjeta presente / e-commerce con tarjeta (IEEE-CIS, Sparkov).
    MOBILE_MONEY : str
        Dinero móvil: transferencias y retiros (PaySim).
    AML : str
        Anti-lavado de dinero, con tipologías de lavado (SAML-D).
    ACCOUNT_OPENING : str
        Fraude en la apertura de cuentas bancarias (BAF).
    BANK_PAYMENTS : str
        Pagos bancarios con tarjeta (BankSim).
    ECOMMERCE : str
        Fraude en comercio electrónico (FDB / fraudecom).
    """

    CARD_PRESENT = "card_present"
    CARD_NOT_PRESENT = "card_not_present"
    MOBILE_MONEY = "mobile_money"
    AML = "aml"
    ACCOUNT_OPENING = "account_opening"
    BANK_PAYMENTS = "bank_payments"
    ECOMMERCE = "ecommerce"


class DatasetSpec(BaseModel):
    """Especificación de metadatos de un dataset tabular del estudio.

    Reúne la información descriptiva necesaria para interpretar y comparar los
    resultados CIPA de un dataset. No incluye el contrato de datos (target, roles
    de columna): ese se resuelve en tiempo de ejecución desde el ``_roles.json``
    de origen. Los valores de ``n_records`` y ``fraud_rate`` provienen de la
    revisión literaria del proyecto comparativo y sirven de referencia descriptiva.

    Attributes
    ----------
    id : str
        Identificador corto del dataset (clave del :data:`REGISTRY`, p. ej.
        ``"ulb_cc"``). Coincide con el subdirectorio en ``data/processed/``.
    name : str
        Nombre legible completo del dataset.
    domain : FraudDomain
        Dominio/tipo de fraude que representa (ver :class:`FraudDomain`).
    fraud_type : str
        Descripción textual del tipo de fraude.
    origin : Origin
        Procedencia real o sintética (ver :class:`Origin`).
    n_records : int
        Número de registros del dataset completo (referencia descriptiva).
    fraud_rate : float
        Proporción de la clase fraude en [0, 1] (referencia descriptiva; la tasa
        efectiva tras submuestreo se registra por corrida en el manifiesto).
    cipa_benchmark_key : str | None
        Clave equivalente del dataset en el benchmark original de CIPA, si aplica
        (habilita el *check* de consistencia, RQ8). ``None`` si el dataset no
        forma parte del benchmark de CIPA.
    notes : str | None
        Notas relevantes para la interpretación (p. ej. columnas de fuga a
        excluir, alta dimensionalidad, número de pasos temporales).
    """

    id: str
    name: str
    domain: FraudDomain
    fraud_type: str
    origin: Origin
    n_records: int
    fraud_rate: float  # proporcion [0,1]
    # Clave del dataset en el benchmark original de CIPA (para el check de
    # consistencia, RQ8). None si el dataset no esta en el benchmark.
    cipa_benchmark_key: str | None = None
    notes: str | None = None


REGISTRY: dict[str, DatasetSpec] = {
    "ulb_cc": DatasetSpec(
        id="ulb_cc",
        name="ULB Credit Card Fraud Detection",
        domain=FraudDomain.CARD_PRESENT,
        fraud_type="Tarjeta de credito (card-present)",
        origin=Origin.REAL,
        n_records=284_807,
        fraud_rate=0.00173,
        cipa_benchmark_key="CreditCard",
        notes="28 features PCA anonimizadas + Time + Amount.",
    ),
    "ieee_cis": DatasetSpec(
        id="ieee_cis",
        name="IEEE-CIS Fraud Detection (Vesta)",
        domain=FraudDomain.CARD_NOT_PRESENT,
        fraud_type="Card-not-present (e-commerce)",
        origin=Origin.REAL,
        n_records=590_540,
        fraud_rate=0.035,
        cipa_benchmark_key="IEEE-CIS",
        notes="Alta dimensionalidad (~431 cols crudas) y muchos faltantes.",
    ),
    "paysim": DatasetSpec(
        id="paysim",
        name="PaySim Mobile Money",
        domain=FraudDomain.MOBILE_MONEY,
        fraud_type="Dinero movil (transferencias y retiros)",
        origin=Origin.SYNTHETIC,
        n_records=6_362_620,
        fraud_rate=0.0013,
        cipa_benchmark_key="PaySim",
        notes="744 pasos temporales (1h). Fuga: isFlaggedFraud (excluida).",
    ),
    "saml_d": DatasetSpec(
        id="saml_d",
        name="SAML-D (Synthetic AML Dataset)",
        domain=FraudDomain.AML,
        fraud_type="Lavado de dinero (28 tipologias)",
        origin=Origin.SYNTHETIC,
        n_records=9_504_852,
        fraud_rate=0.00104,
        cipa_benchmark_key=None,
        notes="Dataset mas grande del estudio. Fuga: Laundering_type (excluida).",
    ),
    "banksim": DatasetSpec(
        id="banksim",
        name="BankSim",
        domain=FraudDomain.BANK_PAYMENTS,
        fraud_type="Pagos bancarios (tarjeta)",
        origin=Origin.SYNTHETIC,
        n_records=594_643,
        fraud_rate=0.012,
        cipa_benchmark_key=None,
        notes="180 pasos (6 meses). Categoricas: category, age, gender.",
    ),
    "sparkov": DatasetSpec(
        id="sparkov",
        name="Sparkov Credit Card Transactions",
        domain=FraudDomain.CARD_NOT_PRESENT,
        fraud_type="Card-not-present (tarjeta sintetica)",
        origin=Origin.SYNTHETIC,
        n_records=1_852_394,
        fraud_rate=0.0052,
        cipa_benchmark_key=None,
        notes="Features interpretables + coordenadas geograficas.",
    ),
    "baf": DatasetSpec(
        id="baf",
        name="Bank Account Fraud (BAF, Base) NeurIPS 2022",
        domain=FraudDomain.ACCOUNT_OPENING,
        fraud_type="Apertura de cuentas bancarias",
        origin=Origin.SYNTHETIC,
        n_records=1_000_000,
        fraud_rate=0.011,
        cipa_benchmark_key=None,
        notes="Variante Base de 6. Sesgos controlados.",
    ),
    "fdb": DatasetSpec(
        id="fdb",
        name="FDB / fraudecom (Amazon Fraud Dataset Benchmark)",
        domain=FraudDomain.ECOMMERCE,
        fraud_type="E-commerce",
        origin=Origin.REAL,
        n_records=151_112,
        fraud_rate=0.093,
        cipa_benchmark_key=None,
        notes="Sub-dataset fraudecom (Fraud_Data.csv). Mayor tasa de fraude.",
    ),
}


def get(dataset_id: str) -> DatasetSpec:
    """Devuelve la especificación de metadatos de un dataset del registro.

    Parameters
    ----------
    dataset_id : str
        Identificador del dataset (clave del :data:`REGISTRY`).

    Returns
    -------
    DatasetSpec
        Especificación del dataset solicitado.

    Raises
    ------
    KeyError
        Si el identificador no está en el registro. El mensaje incluye la lista de
        identificadores disponibles.
    """
    if dataset_id not in REGISTRY:
        raise KeyError(
            f"Dataset {dataset_id!r} no esta en el registro. "
            f"Disponibles: {', '.join(REGISTRY)}"
        )
    return REGISTRY[dataset_id]


def all_ids() -> list[str]:
    """Devuelve los identificadores de los 8 datasets tabulares del estudio.

    Returns
    -------
    list[str]
        Identificadores en el orden de definición del registro.
    """
    return list(REGISTRY)


def benchmark_overlap() -> dict[str, str]:
    """Devuelve los datasets que también están en el benchmark original de CIPA.

    Base del *check* de consistencia (RQ8): recomputar CIPA sobre estos datasets
    (limpios + con features) y comparar banda/firma/DS contra los valores
    publicados del benchmark.

    Returns
    -------
    dict[str, str]
        Mapa ``{id_del_estudio -> clave_en_el_benchmark_CIPA}`` restringido a los
        datasets con solapamiento (los que tienen ``cipa_benchmark_key``).
    """
    return {k: v.cipa_benchmark_key for k, v in REGISTRY.items() if v.cipa_benchmark_key}
