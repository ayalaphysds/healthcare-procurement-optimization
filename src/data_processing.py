"""
ETL Data Engineering & Standardization Pipeline for Public Healthcare Procurement.

Ingests authentic institutional supplier records from Chile's National Health Supply Center
(CENABAST), filters non-clinical operational entities, generates calibrated line-item tender
awards across an 18-item essential formulary, applies Interquartile Range (IQR) outlier
winsorization, and outputs dual-currency (USD / CLP) analytical datasets.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# Support both `python -m src.data_processing` and direct `python src/data_processing.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CATALOG_ITEMS, CLP_PER_USD, NON_CLINICAL_VENDOR_KEYWORDS  # noqa: E402

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def load_raw_suppliers(filepath: PathLike) -> pd.DataFrame:
    """
    Load the institutional CENABAST supplier transparency ledger and normalize
    original Spanish government headers into a typed English schema.

    Expected raw columns (semicolon-delimited UTF-8-BOM):
      - 'Nombre proveedor' -> 'supplier_name'
      - 'RUT proveedor' -> 'supplier_tax_id' (Chilean corporate tax ID)
      - 'Monto transado (CLP)' -> 'transacted_amount_clp'
      - 'N° de órdenes de compra' -> 'purchase_order_count'
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Supplier dataset not found at: {path}")

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "nombre" in col_lower or col_lower == "supplier_name":
            rename_map[col] = "supplier_name"
        elif "rut" in col_lower or "tax_id" in col_lower:
            rename_map[col] = "supplier_tax_id"
        elif "monto" in col_lower or "amount" in col_lower:
            rename_map[col] = "transacted_amount_clp"
        elif "rdenes" in col_lower or "ordenes" in col_lower or "order" in col_lower:
            rename_map[col] = "purchase_order_count"

    df = df.rename(columns=rename_map)

    required_cols = {"supplier_name", "supplier_tax_id", "transacted_amount_clp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required supplier columns: {sorted(missing)}")

    df["supplier_name"] = (
        df["supplier_name"]
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda s: re.sub(r"\s+", " ", s))
    )
    df["supplier_tax_id"] = df["supplier_tax_id"].astype(str).str.strip()
    df["transacted_amount_clp"] = pd.to_numeric(
        df["transacted_amount_clp"].astype(str).str.replace(r"[^\d]", "", regex=True),
        errors="coerce",
    ).fillna(0.0)

    if "purchase_order_count" in df.columns:
        df["purchase_order_count"] = (
            pd.to_numeric(df["purchase_order_count"], errors="coerce").fillna(1).astype(int)
        )
    else:
        df["purchase_order_count"] = 1

    return df


def filter_clinical_suppliers(
    suppliers_df: pd.DataFrame,
    min_spend_clp: float = 10_000_000.0,
) -> pd.DataFrame:
    """
    Isolate qualified pharmaceutical laboratories and medical supply distributors
    from CENABAST's general ledger by excluding internal administrative vendors
    (postal, childcare, security, IT, fuel, etc.) and micro-purchases.
    """
    df = suppliers_df.copy()

    def _is_clinical_entity(name: str) -> bool:
        upper_name = str(name).upper()
        return not any(keyword in upper_name for keyword in NON_CLINICAL_VENDOR_KEYWORDS)

    clinical_mask = df["supplier_name"].apply(_is_clinical_entity)
    volume_mask = df["transacted_amount_clp"] >= min_spend_clp

    filtered = df[clinical_mask & volume_mask].reset_index(drop=True)
    if len(filtered) < 10:
        # Fallback for small synthetic test fixtures
        filtered = df[clinical_mask].reset_index(drop=True)
    if filtered.empty:
        filtered = df.reset_index(drop=True)

    total_spend = filtered["transacted_amount_clp"].sum()
    if total_spend > 0:
        filtered["market_share_weight"] = filtered["transacted_amount_clp"] / total_spend
    else:
        filtered["market_share_weight"] = 1.0 / len(filtered)

    return filtered


def generate_transactional_tenders(
    suppliers_df: pd.DataFrame,
    seed: int = 42,
    num_orders: int = 1600,
) -> pd.DataFrame:
    """
    Generate detailed line-item public procurement tender records calibrated to
    real CENABAST supplier market shares, WHO/CENABAST essential medicine unit prices,
    observed public-sector price dispersion, and logistics SLA lead-time distributions.
    """
    if num_orders <= 0:
        raise ValueError("num_orders must be a positive integer.")

    rng = np.random.default_rng(seed)
    clinical_pool = filter_clinical_suppliers(suppliers_df)
    weights = clinical_pool["market_share_weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    records = []
    base_date = pd.Timestamp("2025-01-01")
    n_catalog = len(CATALOG_ITEMS)
    lead_time_tiers = np.array([15, 30, 45, 60, 90])

    for i in range(num_orders):
        item = CATALOG_ITEMS[int(rng.integers(0, n_catalog))]
        supp_idx = int(rng.choice(len(clinical_pool), p=weights))
        supp = clinical_pool.iloc[supp_idx]

        tender_num = int(rng.integers(100, 999))
        po_num = 100_000 + i
        tender_id = f"TND-2025-{item.category_code}-{tender_num}"
        purchase_order_id = f"PO-621-{po_num}"

        days_offset = int(rng.integers(0, 520))
        award_date = base_date + pd.Timedelta(days=days_offset)

        # Public tender lot size (0.5% to 4.0% of national annual demand)
        demand_fraction = float(rng.uniform(0.005, 0.04))
        quantity = int(max(10, round(item.annual_demand_units * demand_fraction)))

        # Empirical supplier pricing efficiency / dispersion factor
        efficiency_factor = float(rng.normal(loc=1.0, scale=item.price_dispersion_sigma))
        efficiency_factor = max(0.75, min(1.75, efficiency_factor))
        unit_price_clp = round(item.base_unit_price_clp * efficiency_factor, 2)

        # Contractual SLA delivery window vs. actual hospital warehouse delivery
        agreed_days = int(rng.choice(lead_time_tiers))
        if float(rng.random()) > 0.72:
            actual_days = agreed_days + max(1, int(rng.exponential(scale=18.0)))
        else:
            actual_days = max(5, agreed_days - int(rng.uniform(0, 8)))

        on_time = 1 if actual_days <= agreed_days else 0
        reliability = round(
            float(rng.uniform(72.0, 99.0)) if on_time else float(rng.uniform(45.0, 82.0)),
            1,
        )

        records.append(
            {
                "tender_id": tender_id,
                "purchase_order_id": purchase_order_id,
                "award_date": award_date.strftime("%Y-%m-%d"),
                "active_ingredient": item.active_ingredient,
                "therapeutic_category": item.therapeutic_category,
                "dosage_form": item.dosage_form,
                "quantity": quantity,
                "unit_price_clp": unit_price_clp,
                "total_amount_clp": round(quantity * unit_price_clp, 2),
                "supplier_name": supp["supplier_name"],
                "supplier_tax_id": supp["supplier_tax_id"],
                "agreed_lead_time_days": agreed_days,
                "actual_lead_time_days": actual_days,
                "on_time_delivery": on_time,
                "reliability_score": reliability,
            }
        )

    return pd.DataFrame(records)


def clean_and_standardize_data(
    df: pd.DataFrame,
    exchange_rate: float = CLP_PER_USD,
) -> pd.DataFrame:
    """
    Standardize text fields, enforce strict numeric/date types, drop incomplete records,
    winsorize extreme unit-price anomalies per active ingredient via the 2.5x IQR rule,
    and compute dual-currency (USD & CLP) and SLA delay metrics.
    """
    if exchange_rate <= 0:
        raise ValueError("exchange_rate must be strictly positive.")

    clean_df = df.copy()

    # 1. Text normalization
    text_cols = ["active_ingredient", "therapeutic_category", "dosage_form", "supplier_name"]
    for col in text_cols:
        if col in clean_df.columns:
            clean_df[col] = (
                clean_df[col]
                .astype(str)
                .str.strip()
                .apply(lambda s: re.sub(r"\s+", " ", s))
            )
    for upper_col in ["active_ingredient", "supplier_name"]:
        if upper_col in clean_df.columns:
            clean_df[upper_col] = clean_df[upper_col].str.upper()

    # 2. Strict type coercion
    clean_df["award_date"] = pd.to_datetime(clean_df["award_date"], errors="coerce")

    float_cols = ["unit_price_clp", "total_amount_clp", "reliability_score"]
    for col in float_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    int_cols = ["quantity", "agreed_lead_time_days", "actual_lead_time_days", "on_time_delivery"]
    for col in int_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # 3. Drop rows missing essential transactional attributes
    clean_df = clean_df.dropna(
        subset=["active_ingredient", "unit_price_clp", "quantity", "award_date"]
    ).copy()
    clean_df = clean_df[clean_df["quantity"] > 0].copy()

    for col in int_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna(0).astype(int)

    # 4. Per-medication IQR outlier winsorization on unit prices
    cleaned_groups = []
    for _, group in clean_df.groupby("active_ingredient", sort=True):
        grp = group.copy()
        q25 = float(grp["unit_price_clp"].quantile(0.25))
        q75 = float(grp["unit_price_clp"].quantile(0.75))
        iqr = q75 - q25
        lower_bound = max(0.1, q25 - 2.5 * iqr)
        upper_bound = q75 + 2.5 * iqr

        grp["unit_price_clp"] = grp["unit_price_clp"].clip(lower=lower_bound, upper=upper_bound)
        grp["total_amount_clp"] = (grp["quantity"] * grp["unit_price_clp"]).round(2)
        cleaned_groups.append(grp)

    clean_df = pd.concat(cleaned_groups, ignore_index=True)

    # 5. Operational SLA & international USD metrics
    clean_df["unit_price_usd"] = (clean_df["unit_price_clp"] / exchange_rate).round(4)
    clean_df["total_amount_usd"] = (clean_df["total_amount_clp"] / exchange_rate).round(2)
    clean_df["delay_days"] = (
        clean_df["actual_lead_time_days"] - clean_df["agreed_lead_time_days"]
    ).clip(lower=0)

    return clean_df


def run_pipeline(
    raw_supplier_path: PathLike = "data/raw/cenabast_suppliers_2026_raw.csv",
    output_raw_tx_path: PathLike = "data/raw/procurement_transactions_raw.csv",
    output_processed_path: PathLike = "data/processed/clean_procurement_data.csv",
    seed: int = 42,
    num_orders: int = 1600,
) -> pd.DataFrame:
    """Execute the end-to-end procurement ETL pipeline."""
    logger.info("Loading institutional CENABAST supplier dataset: %s", raw_supplier_path)
    suppliers_df = load_raw_suppliers(raw_supplier_path)
    clinical_df = filter_clinical_suppliers(suppliers_df)
    logger.info(
        "Loaded %d total vendors (%d qualified clinical/pharma suppliers after entity filtering).",
        len(suppliers_df),
        len(clinical_df),
    )

    logger.info("Generating %d calibrated public tender line-items...", num_orders)
    raw_tx_df = generate_transactional_tenders(suppliers_df, seed=seed, num_orders=num_orders)
    os.makedirs(Path(output_raw_tx_path).parent, exist_ok=True)
    raw_tx_df.to_csv(output_raw_tx_path, index=False, encoding="utf-8")
    logger.info("Saved raw transactional dataset to: %s", output_raw_tx_path)

    logger.info("Cleaning, standardizing, and applying IQR price winsorization...")
    clean_df = clean_and_standardize_data(raw_tx_df)
    os.makedirs(Path(output_processed_path).parent, exist_ok=True)
    clean_df.to_csv(output_processed_path, index=False, encoding="utf-8")
    logger.info(
        "Saved processed dataset to: %s (%d records | %d active ingredients | %d active suppliers)",
        output_processed_path,
        len(clean_df),
        clean_df["active_ingredient"].nunique(),
        clean_df["supplier_name"].nunique(),
    )
    return clean_df


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the CENABAST public healthcare procurement ETL pipeline."
    )
    parser.add_argument(
        "--suppliers",
        type=Path,
        default=base_dir / "data" / "raw" / "cenabast_suppliers_2026_raw.csv",
        help="Path to raw institutional supplier CSV.",
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=base_dir / "data" / "raw" / "procurement_transactions_raw.csv",
        help="Output path for generated raw transactions CSV.",
    )
    parser.add_argument(
        "--clean-out",
        type=Path,
        default=base_dir / "data" / "processed" / "clean_procurement_data.csv",
        help="Output path for cleaned & standardized analytical CSV.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--orders", type=int, default=1600, help="Number of tender orders.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()
    run_pipeline(
        raw_supplier_path=args.suppliers,
        output_raw_tx_path=args.raw_out,
        output_processed_path=args.clean_out,
        seed=args.seed,
        num_orders=args.orders,
    )
