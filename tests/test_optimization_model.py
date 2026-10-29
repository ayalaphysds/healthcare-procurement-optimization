"""
Mathematical & invariance unit tests for `ProcurementOptimizer` (`src/optimization_model.py`).

Verifies:
- Baseline metrics & HHI bounds [0, 10000]
- Fair-Price Band quantile monotonicity (P25 <= P50 <= P75)
- HiGHS Linear Programming convergence
- Exact 100% demand conservation per medication
- Strict adherence to anti-monopoly supplier share caps (x_{i,j} <= alpha * D_i)
- Category-scoped consistency
"""

from pathlib import Path

import pytest

from src.data_processing import (
    clean_and_standardize_data,
    generate_transactional_tenders,
    load_raw_suppliers,
)
from src.optimization_model import ProcurementOptimizer

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_SUPPLIERS_CSV = REPO_ROOT / "data" / "raw" / "cenabast_suppliers_2026_raw.csv"


@pytest.fixture(scope="module")
def sample_clean_df():
    suppliers = load_raw_suppliers(RAW_SUPPLIERS_CSV)
    raw_tx = generate_transactional_tenders(suppliers, seed=42, num_orders=600)
    return clean_and_standardize_data(raw_tx)


def test_baseline_metrics_and_hhi_bounds(sample_clean_df) -> None:
    opt = ProcurementOptimizer(df=sample_clean_df)
    base = opt.compute_baseline()

    assert base["total_spend_clp"] > 0
    assert base["total_spend_usd"] > 0
    assert 0.0 < base["hhi_market_concentration"] <= 10_000.0
    assert 0.0 < base["avg_product_hhi"] <= 10_000.0
    assert 0.0 <= base["on_time_delivery_pct"] <= 100.0
    assert base["avg_max_min_ratio"] >= 1.0


def test_fair_price_bands_monotonicity(sample_clean_df) -> None:
    opt = ProcurementOptimizer(df=sample_clean_df)
    bands = opt.compute_fair_price_bands()

    assert len(bands) == sample_clean_df["active_ingredient"].nunique()
    assert (bands["fair_price_min_clp"] <= bands["fair_price_median_clp"]).all()
    assert (bands["fair_price_median_clp"] <= bands["fair_price_max_clp"]).all()
    assert (bands["historical_overprice_clp"] >= 0.0).all()


@pytest.mark.parametrize("max_share", [0.30, 0.45, 0.60])
def test_highs_lp_demand_conservation_and_share_caps(sample_clean_df, max_share: float) -> None:
    opt = ProcurementOptimizer(df=sample_clean_df)
    res = opt.solve_procurement_optimization(
        max_supplier_share=max_share,
        min_reliability_score=70.0,
    )

    assert res["success"] is True
    assert res["optimized_spend_clp"] <= res["historical_spend_clp"]
    assert res["savings_pct"] > 0.0
    assert 0.0 <= res["resilience_score"] <= 100.0

    df_alloc = res["df_allocations"]
    assert not df_alloc.empty

    # 1. Exact Demand Conservation: allocated units per drug must match historical demand
    expected_demand = sample_clean_df.groupby("active_ingredient")["quantity"].sum()
    actual_allocated = df_alloc.groupby("active_ingredient")["allocated_units"].sum()
    for drug, req_qty in expected_demand.items():
        assert pytest.approx(actual_allocated.loc[drug], rel=1e-4) == req_qty

    # 2. Anti-Monopoly Share Cap: no single supplier exceeds max_share * 100% (+ tiny rounding epsilon)
    assert (df_alloc["product_share_pct"] <= (max_share * 100.0 + 0.05)).all()

    # 3. Category comparison sums exactly to global LP objective
    df_cat = res["df_category_comparison"]
    assert pytest.approx(df_cat["optimized_spend_clp"].sum(), rel=1e-5) == res["optimized_spend_clp"]


def test_category_filter_scoping(sample_clean_df) -> None:
    opt = ProcurementOptimizer(df=sample_clean_df)
    target_cat = "Antibiotics & Anti-Infectives"
    res = opt.solve_procurement_optimization(
        max_supplier_share=0.45,
        min_reliability_score=70.0,
        category_filter=target_cat,
    )

    assert res["success"] is True
    df_alloc = res["df_allocations"]
    assert set(df_alloc["therapeutic_category"].unique()) == {target_cat}


def test_invalid_parameters_raise_value_error(sample_clean_df) -> None:
    opt = ProcurementOptimizer(df=sample_clean_df)
    with pytest.raises(ValueError):
        opt.solve_procurement_optimization(max_supplier_share=0.0)
    with pytest.raises(ValueError):
        opt.solve_procurement_optimization(min_reliability_score=120.0)

