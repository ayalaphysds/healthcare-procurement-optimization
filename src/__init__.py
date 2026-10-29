"""
Healthcare Procurement Optimization & Supply Chain Analytics Engine.

Case Study: Chile's National Health Supply Center (CENABAST).
Combines ETL data engineering, empirical IQR price-band anomaly detection,
and constrained Linear Programming (SciPy HiGHS) to minimize public pharmaceutical
expenditure while preventing hospital drug stockouts.
"""

from src.config import CATALOG_ITEMS, CLP_PER_USD, CatalogItem
from src.data_processing import (
    clean_and_standardize_data,
    filter_clinical_suppliers,
    generate_transactional_tenders,
    load_raw_suppliers,
    run_pipeline,
)
from src.optimization_model import CenabastOptimizer, ProcurementOptimizer

__version__ = "1.0.0"

__all__ = [
    "CATALOG_ITEMS",
    "CLP_PER_USD",
    "CatalogItem",
    "CenabastOptimizer",
    "ProcurementOptimizer",
    "clean_and_standardize_data",
    "filter_clinical_suppliers",
    "generate_transactional_tenders",
    "load_raw_suppliers",
    "run_pipeline",
]

