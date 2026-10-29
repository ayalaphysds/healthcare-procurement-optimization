"""
Unit tests for ETL data ingestion, clinical entity filtering, schema standardization,
and IQR outlier winsorization (`src/data_processing.py`).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config import CLP_PER_USD
from src.data_processing import (
    clean_and_standardize_data,
    filter_clinical_suppliers,
    generate_transactional_tenders,
    load_raw_suppliers,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_SUPPLIERS_CSV = REPO_ROOT / "data" / "raw" / "cenabast_suppliers_2026_raw.csv"


def test_load_raw_suppliers_parses_chilean_ledger() -> None:
    df = load_raw_suppliers(RAW_SUPPLIERS_CSV)
    assert len(df) >= 240
    expected_cols = {
        "supplier_name",
        "supplier_tax_id",
        "transacted_amount_clp",
        "purchase_order_count",
    }
    assert expected_cols.issubset(set(df.columns))
    assert (df["transacted_amount_clp"] >= 0).all()
    assert df["transacted_amount_clp"].sum() > 400_000_000_000  # > 400B CLP (~$420M+ USD)


def test_load_raw_suppliers_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_suppliers("nonexistent_suppliers.csv")


def test_filter_clinical_suppliers_excludes_non_clinical_vendors() -> None:
    raw_df = load_raw_suppliers(RAW_SUPPLIERS_CSV)
    clinical_df = filter_clinical_suppliers(raw_df)

    assert len(clinical_df) > 100
    assert len(clinical_df) < len(raw_df)
    assert pytest.approx(clinical_df["market_share_weight"].sum(), rel=1e-6) == 1.0

    # Non-healthcare administrative vendors should be excluded
    names_joined = " | ".join(clinical_df["supplier_name"].tolist())
    assert "CORREOS DE CHILE" not in names_joined
    assert "SALA CUNA" not in names_joined
    assert "LATAM AIRLINES" not in names_joined


def test_generate_and_clean_transactional_pipeline_is_deterministic() -> None:
    raw_df = load_raw_suppliers(RAW_SUPPLIERS_CSV)
    tx_a = generate_transactional_tenders(raw_df, seed=123, num_orders=200)
    tx_b = generate_transactional_tenders(raw_df, seed=123, num_orders=200)

    pd.testing.assert_frame_equal(tx_a, tx_b)

    clean_df = clean_and_standardize_data(tx_a)
    assert len(clean_df) == 200
    assert (clean_df["delay_days"] >= 0).all()
    assert (clean_df["unit_price_usd"] > 0).all()
    assert pytest.approx(
        clean_df["unit_price_clp"].iloc[0] / CLP_PER_USD, abs=1e-3
    ) == clean_df["unit_price_usd"].iloc[0]


def test_iqr_winsorization_caps_extreme_price_spike() -> None:
    base_rows = [
        {
            "tender_id": f"TND-{i}",
            "purchase_order_id": f"PO-{i}",
            "award_date": "2025-06-01",
            "active_ingredient": "LOSARTAN 50 MG",
            "therapeutic_category": "Cardiovascular & Hypertension",
            "dosage_form": "Oral Tablet",
            "quantity": 1000,
            "unit_price_clp": 20.0 + (i % 5),
            "total_amount_clp": 1000 * (20.0 + (i % 5)),
            "supplier_name": "TEST LAB SPA",
            "supplier_tax_id": "76.000.000-1",
            "agreed_lead_time_days": 30,
            "actual_lead_time_days": 28,
            "on_time_delivery": 1,
            "reliability_score": 90.0,
        }
        for i in range(20)
    ]
    # Inject a 100x data-entry error outlier
    outlier_row = dict(base_rows[0])
    outlier_row["purchase_order_id"] = "PO-OUTLIER"
    outlier_row["unit_price_clp"] = 2500.0
    outlier_row["total_amount_clp"] = 2_500_000.0

    dirty_df = pd.DataFrame(base_rows + [outlier_row])
    cleaned = clean_and_standardize_data(dirty_df)

    assert len(cleaned) == 21
    assert cleaned["unit_price_clp"].max() < 50.0  # Extreme 2500 CLP spike capped via 2.5x IQR

