"""
Centralized configuration, reference currency constants, clinical domain filters,
and standardized essential pharmaceutical catalog for CENABAST procurement modeling.
"""

from dataclasses import dataclass
from typing import Tuple

# Reference macroeconomic exchange rate (1 USD ≈ 950 CLP) for international benchmarking.
CLP_PER_USD: float = 950.0


@dataclass(frozen=True)
class CatalogItem:
    """Represents a standardized essential medication or hospital supply item."""

    active_ingredient: str
    therapeutic_category: str
    dosage_form: str
    category_code: str
    base_unit_price_clp: float
    price_dispersion_sigma: float
    annual_demand_units: int


# Standardized essential formulary modeled after WHO Essential Medicines & CENABAST priority tenders
CATALOG_ITEMS: Tuple[CatalogItem, ...] = (
    CatalogItem(
        active_ingredient="ACETAMINOPHEN (PARACETAMOL) 500 MG",
        therapeutic_category="Analgesics & Antipyretics",
        dosage_form="Oral Tablet",
        category_code="ANA",
        base_unit_price_clp=22.5,
        price_dispersion_sigma=0.28,
        annual_demand_units=12_500_000,
    ),
    CatalogItem(
        active_ingredient="IBUPROFEN 400 MG",
        therapeutic_category="Analgesics & Antipyretics",
        dosage_form="Film-Coated Tablet",
        category_code="ANA",
        base_unit_price_clp=34.0,
        price_dispersion_sigma=0.25,
        annual_demand_units=7_800_000,
    ),
    CatalogItem(
        active_ingredient="KETOROLAC 10 MG",
        therapeutic_category="Analgesics & Antipyretics",
        dosage_form="Oral Tablet",
        category_code="ANA",
        base_unit_price_clp=65.0,
        price_dispersion_sigma=0.30,
        annual_demand_units=3_200_000,
    ),
    CatalogItem(
        active_ingredient="METFORMIN 850 MG",
        therapeutic_category="Diabetes & Metabolic Care",
        dosage_form="Oral Tablet",
        category_code="DIA",
        base_unit_price_clp=28.0,
        price_dispersion_sigma=0.22,
        annual_demand_units=18_000_000,
    ),
    CatalogItem(
        active_ingredient="INSULIN NPH 100 IU/ML",
        therapeutic_category="Diabetes & Metabolic Care",
        dosage_form="10 mL Multi-Dose Vial",
        category_code="DIA",
        base_unit_price_clp=4_250.0,
        price_dispersion_sigma=0.20,
        annual_demand_units=450_000,
    ),
    CatalogItem(
        active_ingredient="LOSARTAN 50 MG",
        therapeutic_category="Cardiovascular & Hypertension",
        dosage_form="Oral Tablet",
        category_code="CAR",
        base_unit_price_clp=24.0,
        price_dispersion_sigma=0.26,
        annual_demand_units=16_500_000,
    ),
    CatalogItem(
        active_ingredient="ATORVASTATIN 20 MG",
        therapeutic_category="Cardiovascular & Hypertension",
        dosage_form="Oral Tablet",
        category_code="CAR",
        base_unit_price_clp=45.0,
        price_dispersion_sigma=0.29,
        annual_demand_units=11_000_000,
    ),
    CatalogItem(
        active_ingredient="ENALAPRIL 10 MG",
        therapeutic_category="Cardiovascular & Hypertension",
        dosage_form="Oral Tablet",
        category_code="CAR",
        base_unit_price_clp=18.5,
        price_dispersion_sigma=0.32,
        annual_demand_units=9_800_000,
    ),
    CatalogItem(
        active_ingredient="AMOXICILLIN 500 MG",
        therapeutic_category="Antibiotics & Anti-Infectives",
        dosage_form="Hard Capsule",
        category_code="ANT",
        base_unit_price_clp=55.0,
        price_dispersion_sigma=0.24,
        annual_demand_units=6_400_000,
    ),
    CatalogItem(
        active_ingredient="AZITHROMYCIN 500 MG",
        therapeutic_category="Antibiotics & Anti-Infectives",
        dosage_form="Oral Tablet",
        category_code="ANT",
        base_unit_price_clp=210.0,
        price_dispersion_sigma=0.27,
        annual_demand_units=2_100_000,
    ),
    CatalogItem(
        active_ingredient="CEFTRIAXONE 1 G",
        therapeutic_category="Antibiotics & Anti-Infectives",
        dosage_form="Powder for IV/IM Injection",
        category_code="ANT",
        base_unit_price_clp=1_150.0,
        price_dispersion_sigma=0.21,
        annual_demand_units=1_400_000,
    ),
    CatalogItem(
        active_ingredient="PEMBROLIZUMAB 100 MG/4 ML",
        therapeutic_category="Oncology & High-Cost Biologics",
        dosage_form="IV Infusion Concentrate",
        category_code="ONC",
        base_unit_price_clp=2_350_000.0,
        price_dispersion_sigma=0.14,
        annual_demand_units=4_800,
    ),
    CatalogItem(
        active_ingredient="TRASTUZUMAB 440 MG",
        therapeutic_category="Oncology & High-Cost Biologics",
        dosage_form="Lyophilized Powder for IV Infusion",
        category_code="ONC",
        base_unit_price_clp=980_000.0,
        price_dispersion_sigma=0.16,
        annual_demand_units=6_200,
    ),
    CatalogItem(
        active_ingredient="INACTIVATED COVID-19 VACCINE (CORONAVAC)",
        therapeutic_category="Vaccines & Immunizations",
        dosage_form="0.5 mL Single-Dose Vial",
        category_code="VAC",
        base_unit_price_clp=7_800.0,
        price_dispersion_sigma=0.12,
        annual_demand_units=1_200_000,
    ),
    CatalogItem(
        active_ingredient="TRIVALENT INFLUENZA VACCINE",
        therapeutic_category="Vaccines & Immunizations",
        dosage_form="0.5 mL Pre-Filled Syringe",
        category_code="VAC",
        base_unit_price_clp=5_600.0,
        price_dispersion_sigma=0.15,
        annual_demand_units=1_800_000,
    ),
    CatalogItem(
        active_ingredient="ISOTONIC SALINE SOLUTION 0.9% 500 ML",
        therapeutic_category="Critical Hospital & Surgical Supplies",
        dosage_form="500 mL Flexible IV Bag",
        category_code="SUP",
        base_unit_price_clp=720.0,
        price_dispersion_sigma=0.22,
        annual_demand_units=4_500_000,
    ),
    CatalogItem(
        active_ingredient="STERILE SURGICAL GLOVES SIZE 7.5",
        therapeutic_category="Critical Hospital & Surgical Supplies",
        dosage_form="Sterile Wrapped Pair",
        category_code="SUP",
        base_unit_price_clp=390.0,
        price_dispersion_sigma=0.35,
        annual_demand_units=8_200_000,
    ),
    CatalogItem(
        active_ingredient="PARTICULATE RESPIRATOR MASK N95 / FFP2",
        therapeutic_category="Critical Hospital & Surgical Supplies",
        dosage_form="Disposable Respirator Unit",
        category_code="SUP",
        base_unit_price_clp=480.0,
        price_dispersion_sigma=0.40,
        annual_demand_units=5_000_000,
    ),
)

# Keywords identifying internal non-clinical operational vendors in CENABAST's public ledger
# (e.g., postal services, staff childcare, security guards, IT consulting, fuel, dairy programs)
NON_CLINICAL_VENDOR_KEYWORDS: Tuple[str, ...] = (
    "CORREOS DE CHILE",
    "JARDIN INFANTIL",
    "SALA CUNA",
    "PRE- ESCOLAR",
    "EDUCACION",
    "EDUCACIONAL",
    "CAPACITACIONES",
    "OTEC",
    "SEGURIDAD",
    "SECURITY",
    "PETROLEOS",
    "COPEC",
    "AIRLINES",
    "SUPERTRANS",
    "IMPRESOS",
    "PRINT",
    "IMPREXPRESS",
    "COMPUTACION",
    "COMPUTACIONAL",
    "SOFTWARE",
    "CLOUDLATAM",
    "SISTEMAS DE INFORMACION",
    "CONSTRUCCION",
    "REMODELATE",
    "ADUANAS",
    "ALIMENTICIOS",
    "FOODS",
    "PROLESUR",
    "NESTLE",
    "TEXTILES",
    "CHILENERGIA",
    "RECIRCULA",
    "BODEGAS",
)

