"""
Mathematical Procurement Optimization & Supply Chain Risk Analytics Module.

Implements:
1. Historical Baseline Diagnostics (As-Is): Unit-price dispersion (CV, Max/Min ratio),
   Herfindahl-Hirschman Index (HHI) market concentration, and logistics SLA compliance.
2. Empirical Fair-Price Bands (IQR P25-P75): Automated detection of public tender overpricing.
3. Constrained Linear Programming Engine (To-Be): Solved via `scipy.optimize.linprog`
   using the high-performance C++ HiGHS dual-simplex / interior-point solver to minimize
   public healthcare expenditure subject to 100% demand fulfillment, anti-monopoly
   multi-supplier split caps, SLA reliability screening, and supplier capacity bounds.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Support both `python -m src.optimization_model` and direct `python src/optimization_model.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CLP_PER_USD  # noqa: E402

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class ProcurementOptimizer:
    """
    Analytical optimizer for centralized public healthcare procurement and
    multi-supplier supply chain resilience (CENABAST Chile Case Study).
    """

    def __init__(
        self,
        data_path: Optional[PathLike] = None,
        df: Optional[pd.DataFrame] = None,
        exchange_rate: float = CLP_PER_USD,
    ) -> None:
        if exchange_rate <= 0:
            raise ValueError("exchange_rate must be strictly positive.")
        self.exchange_rate = exchange_rate

        if df is not None:
            self.df = df.copy()
        elif data_path is not None and Path(data_path).exists():
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide either an in-memory DataFrame or a valid data_path.")

        if "award_date" in self.df.columns:
            self.df["award_date"] = pd.to_datetime(self.df["award_date"], errors="coerce")

        if "unit_price_usd" not in self.df.columns and "unit_price_clp" in self.df.columns:
            self.df["unit_price_usd"] = self.df["unit_price_clp"] / self.exchange_rate
        if "total_amount_usd" not in self.df.columns and "total_amount_clp" in self.df.columns:
            self.df["total_amount_usd"] = self.df["total_amount_clp"] / self.exchange_rate

        self.baseline_metrics: Dict[str, Any] = {}
        self.optimization_results: Dict[str, Any] = {}
        self.price_bands: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _compute_hhi(spend_by_entity: pd.Series) -> float:
        """
        Compute the Herfindahl-Hirschman Index (HHI) on a 0 to 10,000 scale:
            HHI = sum_j (market_share_pct_j)^2
        where market_share_pct_j in [0, 100].
        """
        total = float(spend_by_entity.sum())
        if total <= 0:
            return 0.0
        shares_pct = (spend_by_entity / total) * 100.0
        return float((shares_pct**2).sum())

    def compute_baseline(self, df_scope: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Compute historical baseline (As-Is) KPIs:
        - Total evaluated spend (CLP & USD)
        - Unit-price dispersion per active ingredient (CV, Max/Min ratio)
        - Logistics SLA compliance & delivery delay statistics
        - Portfolio-level and product-level Herfindahl-Hirschman Index (HHI)
        """
        df = self.df if df_scope is None else df_scope
        total_spend_clp = float(df["total_amount_clp"].sum())
        total_spend_usd = total_spend_clp / self.exchange_rate
        total_units = int(df["quantity"].sum())
        total_orders = len(df)

        # Unit-price dispersion per active ingredient
        price_dispersion = (
            df.groupby(["active_ingredient", "therapeutic_category"], as_index=False)["unit_price_clp"]
            .agg(
                min_price_clp="min",
                max_price_clp="max",
                median_price_clp="median",
                mean_price_clp="mean",
                std_price_clp="std",
                order_count="count",
            )
        )
        price_dispersion["std_price_clp"] = price_dispersion["std_price_clp"].fillna(0.0)
        price_dispersion["coef_variation"] = (
            price_dispersion["std_price_clp"] / price_dispersion["mean_price_clp"]
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        price_dispersion["max_min_ratio"] = (
            price_dispersion["max_price_clp"] / price_dispersion["min_price_clp"]
        ).replace([np.inf, -np.inf], 1.0).fillna(1.0)

        for col in ["min_price_clp", "max_price_clp", "median_price_clp", "mean_price_clp"]:
            usd_col = col.replace("_clp", "_usd")
            price_dispersion[usd_col] = price_dispersion[col] / self.exchange_rate

        avg_cv = float(price_dispersion["coef_variation"].mean()) if not price_dispersion.empty else 0.0
        avg_price_ratio = (
            float(price_dispersion["max_min_ratio"].mean()) if not price_dispersion.empty else 1.0
        )

        # Logistics & SLA compliance metrics
        on_time_rate = float(df["on_time_delivery"].mean() * 100.0) if total_orders > 0 else 0.0
        avg_delay_days = float(df["delay_days"].mean()) if total_orders > 0 else 0.0
        avg_agreed_lead = float(df["agreed_lead_time_days"].mean()) if total_orders > 0 else 0.0
        avg_actual_lead = float(df["actual_lead_time_days"].mean()) if total_orders > 0 else 0.0
        avg_reliability = float(df["reliability_score"].mean()) if total_orders > 0 else 0.0

        # Portfolio-level & per-medication HHI concentration
        supplier_spend = df.groupby("supplier_name")["total_amount_clp"].sum()
        portfolio_hhi = self._compute_hhi(supplier_spend)
        top_5_share = (
            float((supplier_spend.sort_values(ascending=False).head(5).sum() / total_spend_clp) * 100.0)
            if total_spend_clp > 0
            else 0.0
        )

        product_hhis = []
        for _, grp in df.groupby("active_ingredient"):
            prod_supp_spend = grp.groupby("supplier_name")["total_amount_clp"].sum()
            product_hhis.append(self._compute_hhi(prod_supp_spend))
        avg_product_hhi = float(np.mean(product_hhis)) if product_hhis else 0.0

        metrics = {
            "total_spend_clp": total_spend_clp,
            "total_spend_millions_clp": total_spend_clp / 1e6,
            "total_spend_usd": total_spend_usd,
            "total_spend_millions_usd": total_spend_usd / 1e6,
            "total_units": total_units,
            "total_orders": total_orders,
            "num_active_ingredients": int(df["active_ingredient"].nunique()),
            "num_suppliers": int(df["supplier_name"].nunique()),
            "avg_coef_variation": avg_cv,
            "avg_max_min_ratio": avg_price_ratio,
            "on_time_delivery_pct": on_time_rate,
            "avg_delay_days": avg_delay_days,
            "avg_agreed_lead_time": avg_agreed_lead,
            "avg_actual_lead_time": avg_actual_lead,
            "avg_reliability_score": avg_reliability,
            "hhi_market_concentration": portfolio_hhi,
            "avg_product_hhi": avg_product_hhi,
            "top_5_market_share_pct": top_5_share,
            "price_dispersion_table": price_dispersion,
        }
        if df_scope is None:
            self.baseline_metrics = metrics
        return metrics

    def compute_fair_price_bands(self) -> pd.DataFrame:
        """
        Build empirical Interquartile Range (P25-P75) Fair-Price Bands per active ingredient
        and quantify historical overpricing excess above the median benchmark for orders > P75.
        """
        bands = []
        for drug, group in self.df.groupby("active_ingredient", sort=True):
            p25 = float(group["unit_price_clp"].quantile(0.25))
            p50 = float(group["unit_price_clp"].median())
            p75 = float(group["unit_price_clp"].quantile(0.75))
            category = str(group["therapeutic_category"].iloc[0])

            overpriced_orders = group[group["unit_price_clp"] > p75]
            overprice_spend_clp = float(
                ((overpriced_orders["unit_price_clp"] - p50) * overpriced_orders["quantity"]).sum()
            )
            overprice_spend_clp = max(0.0, overprice_spend_clp)

            bands.append(
                {
                    "active_ingredient": drug,
                    "therapeutic_category": category,
                    "fair_price_min_clp": round(p25, 2),
                    "fair_price_median_clp": round(p50, 2),
                    "fair_price_max_clp": round(p75, 2),
                    "fair_price_min_usd": round(p25 / self.exchange_rate, 4),
                    "fair_price_median_usd": round(p50 / self.exchange_rate, 4),
                    "fair_price_max_usd": round(p75 / self.exchange_rate, 4),
                    "historical_overprice_clp": round(overprice_spend_clp, 2),
                    "historical_overprice_usd": round(overprice_spend_clp / self.exchange_rate, 2),
                    "overpriced_orders_count": int(len(overpriced_orders)),
                    "overpriced_orders_pct": round((len(overpriced_orders) / len(group)) * 100.0, 2),
                }
            )

        self.price_bands = pd.DataFrame(bands)
        return self.price_bands

    def solve_procurement_optimization(
        self,
        max_supplier_share: float = 0.45,
        min_reliability_score: float = 70.0,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Solve the Constrained Linear Programming procurement allocation problem (HiGHS):

            minimize_{x >= 0}  sum_{i=1..N} sum_{j=1..M} c_{i,j} * x_{i,j}

        Subject to:
          1. Exact Demand Fulfillment:
                 sum_{j=1..M} x_{i,j} == D_i                     for all medications i
          2. Anti-Monopoly / Stockout Diversification Cap:
                 x_{i,j} <= max_supplier_share * D_i             for all i, j
          3. Supplier Operational Capacity Bound:
                 sum_{i=1..N} x_{i,j} <= Capacity_j              for all suppliers j
          4. Supplier SLA Pre-Qualification:
                 MeanReliability_j >= min_reliability_score
        """
        if not (0.05 <= max_supplier_share <= 1.0):
            raise ValueError("max_supplier_share must be in [0.05, 1.0].")
        if not (0.0 <= min_reliability_score <= 100.0):
            raise ValueError("min_reliability_score must be in [0.0, 100.0].")

        df = self.df.copy()
        if category_filter and category_filter != "ALL":
            filtered_cat = df[df["therapeutic_category"] == category_filter].copy()
            if not filtered_cat.empty:
                df = filtered_cat

        if not self.baseline_metrics:
            self.compute_baseline()
        scope_baseline = self.compute_baseline(df_scope=df)

        # Map each product to its therapeutic category
        product_to_category: Dict[str, str] = (
            df.groupby("active_ingredient")["therapeutic_category"].first().to_dict()
        )

        # Screen suppliers by historical SLA reliability
        supp_stats = (
            df.groupby("supplier_name", as_index=False)
            .agg(
                reliability=("reliability_score", "mean"),
                historical_volume=("quantity", "sum"),
                historical_spend_clp=("total_amount_clp", "sum"),
            )
        )

        # Minimum number of suppliers mathematically required to satisfy x_{ij} <= alpha * D_i
        min_suppliers_needed = max(3, int(math.ceil(1.0 / max_supplier_share)) + 1)
        valid_suppliers = supp_stats[
            supp_stats["reliability"] >= min_reliability_score
        ]["supplier_name"].tolist()

        if len(valid_suppliers) < min_suppliers_needed:
            valid_suppliers = (
                supp_stats.sort_values(by=["reliability", "historical_volume"], ascending=[False, False])
                .head(min_suppliers_needed)["supplier_name"]
                .tolist()
            )

        df_qualified = df[df["supplier_name"].isin(valid_suppliers)].copy()

        demand_series = df.groupby("active_ingredient")["quantity"].sum()
        products = demand_series.index.tolist()
        suppliers = sorted(valid_suppliers)

        n_prod = len(products)
        n_supp = len(suppliers)

        # Build unit cost matrix c_{i,j}: supplier's historical median bid for product i,
        # imputing unobserved (product, supplier) pairs with market median + 5% new-line friction
        prod_supp_price = (
            df_qualified.groupby(["active_ingredient", "supplier_name"])["unit_price_clp"]
            .median()
            .unstack(fill_value=np.nan)
        )
        global_prod_median = df.groupby("active_ingredient")["unit_price_clp"].median()

        for p in products:
            fallback_price = float(global_prod_median.get(p, 100.0)) * 1.05
            if p not in prod_supp_price.index:
                prod_supp_price.loc[p] = fallback_price
            else:
                prod_supp_price.loc[p] = prod_supp_price.loc[p].fillna(fallback_price)

        cost_matrix = prod_supp_price.reindex(index=products, columns=suppliers).to_numpy(dtype=float)
        c_obj = cost_matrix.flatten()

        # 1. Equality constraints: sum_j x_{i,j} == D_i
        a_eq = np.zeros((n_prod, n_prod * n_supp), dtype=float)
        b_eq = np.zeros(n_prod, dtype=float)
        for i, p in enumerate(products):
            d_i = float(demand_series.loc[p])
            b_eq[i] = d_i
            a_eq[i, i * n_supp : (i + 1) * n_supp] = 1.0

        # 2. Variable box bounds: 0 <= x_{i,j} <= max_supplier_share * D_i
        # Expressing per-variable share caps directly in `bounds` keeps the LP matrix compact & fast!
        bounds = []
        for i, p in enumerate(products):
            d_i = float(demand_series.loc[p])
            max_qty_ij = max_supplier_share * d_i
            for _ in range(n_supp):
                bounds.append((0.0, max_qty_ij))

        # 3. Inequality constraints: Supplier aggregate operational capacity sum_i x_{i,j} <= Cap_j
        hist_vol_map = supp_stats.set_index("supplier_name")["historical_volume"].to_dict()
        total_demand_units = float(demand_series.sum())
        a_ub = np.zeros((n_supp, n_prod * n_supp), dtype=float)
        b_ub = np.zeros(n_supp, dtype=float)

        for j, s in enumerate(suppliers):
            hist_cap = float(hist_vol_map.get(s, 0.0))
            cap_limit = max(hist_cap * 2.2, total_demand_units * (max_supplier_share * 0.8))
            for i in range(n_prod):
                a_ub[j, i * n_supp + j] = 1.0
            b_ub[j] = cap_limit

        res = linprog(
            c_obj,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            # Relax aggregate throughput bound while strictly preserving demand & anti-monopoly bounds
            res = linprog(
                c_obj,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

        historical_spend_clp = float(df["total_amount_clp"].sum())
        optimized_spend_clp = float(res.fun) if res.success else historical_spend_clp

        # Extract optimal contract allocations
        x_sol = res.x if res.success else np.zeros(n_prod * n_supp, dtype=float)
        allocations = []
        reliability_map = supp_stats.set_index("supplier_name")["reliability"].to_dict()

        for i, p in enumerate(products):
            d_i = float(demand_series.loc[p])
            for j, s in enumerate(suppliers):
                qty_alloc = float(x_sol[i * n_supp + j])
                if qty_alloc > 1e-3:
                    unit_p_clp = float(cost_matrix[i, j])
                    spend_clp = qty_alloc * unit_p_clp
                    allocations.append(
                        {
                            "active_ingredient": p,
                            "therapeutic_category": product_to_category[p],
                            "supplier_name": s,
                            "supplier_reliability": round(float(reliability_map.get(s, 0.0)), 1),
                            "allocated_units": round(qty_alloc, 2),
                            "product_share_pct": round((qty_alloc / d_i) * 100.0, 2) if d_i > 0 else 0.0,
                            "unit_price_clp": round(unit_p_clp, 2),
                            "unit_price_usd": round(unit_p_clp / self.exchange_rate, 4),
                            "allocated_spend_clp": round(spend_clp, 2),
                            "allocated_spend_usd": round(spend_clp / self.exchange_rate, 2),
                        }
                    )

        df_alloc = pd.DataFrame(allocations)

        # Exact deterministic category-level comparison (As-Is vs. To-Be)
        hist_by_cat = (
            df.groupby("therapeutic_category", as_index=False)["total_amount_clp"]
            .sum()
            .rename(columns={"total_amount_clp": "historical_spend_clp"})
        )
        if not df_alloc.empty:
            opt_by_cat = (
                df_alloc.groupby("therapeutic_category", as_index=False)["allocated_spend_clp"]
                .sum()
                .rename(columns={"allocated_spend_clp": "optimized_spend_clp"})
            )
            df_cat_comp = pd.merge(hist_by_cat, opt_by_cat, on="therapeutic_category", how="left")
            df_cat_comp["optimized_spend_clp"] = df_cat_comp["optimized_spend_clp"].fillna(0.0)
        else:
            df_cat_comp = hist_by_cat.copy()
            df_cat_comp["optimized_spend_clp"] = df_cat_comp["historical_spend_clp"]

        df_cat_comp["savings_clp"] = (
            df_cat_comp["historical_spend_clp"] - df_cat_comp["optimized_spend_clp"]
        ).clip(lower=0.0)
        df_cat_comp["savings_pct"] = np.where(
            df_cat_comp["historical_spend_clp"] > 0,
            (df_cat_comp["savings_clp"] / df_cat_comp["historical_spend_clp"]) * 100.0,
            0.0,
        )
        for col in ["historical_spend_clp", "optimized_spend_clp", "savings_clp"]:
            df_cat_comp[col.replace("_clp", "_usd")] = df_cat_comp[col] / self.exchange_rate

        # Savings & HHI metrics
        savings_clp = max(0.0, historical_spend_clp - optimized_spend_clp)
        savings_pct = (savings_clp / historical_spend_clp) * 100.0 if historical_spend_clp > 0 else 0.0

        hhi_as_is = scope_baseline["hhi_market_concentration"]
        avg_prod_hhi_as_is = scope_baseline["avg_product_hhi"]

        if not df_alloc.empty and optimized_spend_clp > 0:
            opt_supp_spend = df_alloc.groupby("supplier_name")["allocated_spend_clp"].sum()
            hhi_to_be = self._compute_hhi(opt_supp_spend)

            opt_prod_hhis = [
                self._compute_hhi(grp.groupby("supplier_name")["allocated_spend_clp"].sum())
                for _, grp in df_alloc.groupby("active_ingredient")
            ]
            avg_prod_hhi_to_be = float(np.mean(opt_prod_hhis)) if opt_prod_hhis else 0.0

            suppliers_per_drug = df_alloc.groupby("active_ingredient")["supplier_name"].nunique()
            avg_suppliers_per_drug = float(suppliers_per_drug.mean())

            # Post-optimization unit-price CV across awarded contracts
            opt_cvs = []
            for _, grp in df_alloc.groupby("active_ingredient"):
                mean_p = float(grp["unit_price_clp"].mean())
                std_p = float(grp["unit_price_clp"].std()) if len(grp) > 1 else 0.0
                opt_cvs.append((std_p / mean_p) if mean_p > 0 else 0.0)
            optimized_cv = float(np.mean(opt_cvs)) if opt_cvs else 0.0

            weighted_reliability = float(
                (df_alloc["supplier_reliability"] * df_alloc["allocated_spend_clp"]).sum()
                / df_alloc["allocated_spend_clp"].sum()
            )
        else:
            hhi_to_be = hhi_as_is
            avg_prod_hhi_to_be = avg_prod_hhi_as_is
            avg_suppliers_per_drug = 1.0
            optimized_cv = scope_baseline["avg_coef_variation"]
            weighted_reliability = min_reliability_score

        # Supply Chain Resilience Index (0-100): combines multi-sourcing depth & weighted SLA reliability
        diversification_score = min(100.0, (avg_suppliers_per_drug / 3.0) * 100.0)
        resilience_score = round(0.6 * diversification_score + 0.4 * weighted_reliability, 1)

        # Baseline resilience score for apples-to-apples comparison
        hist_supp_per_order_group = 1.0  # Single-winner tender baseline
        baseline_resilience_score = round(
            0.6 * ((hist_supp_per_order_group / 3.0) * 100.0)
            + 0.4 * scope_baseline["avg_reliability_score"],
            1,
        )

        self.optimization_results = {
            "success": bool(res.success),
            "solver_status": res.message,
            "historical_spend_clp": historical_spend_clp,
            "historical_spend_usd": historical_spend_clp / self.exchange_rate,
            "optimized_spend_clp": optimized_spend_clp,
            "optimized_spend_usd": optimized_spend_clp / self.exchange_rate,
            "total_savings_clp": savings_clp,
            "total_savings_millions_clp": savings_clp / 1e6,
            "total_savings_usd": savings_clp / self.exchange_rate,
            "total_savings_millions_usd": (savings_clp / self.exchange_rate) / 1e6,
            "savings_pct": savings_pct,
            "hhi_as_is": hhi_as_is,
            "hhi_to_be": hhi_to_be,
            "hhi_reduction": hhi_as_is - hhi_to_be,
            "avg_product_hhi_as_is": avg_prod_hhi_as_is,
            "avg_product_hhi_to_be": avg_prod_hhi_to_be,
            "avg_suppliers_per_drug": avg_suppliers_per_drug,
            "optimized_coef_variation": optimized_cv,
            "baseline_resilience_score": baseline_resilience_score,
            "resilience_score": resilience_score,
            "df_allocations": df_alloc,
            "df_category_comparison": df_cat_comp,
        }
        return self.optimization_results


# Backward-compatible alias
CenabastOptimizer = ProcurementOptimizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    base_dir = Path(__file__).resolve().parent.parent
    proc_file = base_dir / "data" / "processed" / "clean_procurement_data.csv"

    logger.info("Initializing ProcurementOptimizer from %s...", proc_file)
    optimizer = ProcurementOptimizer(data_path=proc_file)

    baseline = optimizer.compute_baseline()
    logger.info(
        "Historical Spend (As-Is): $%.2fM USD ($%.1fM CLP)",
        baseline["total_spend_millions_usd"],
        baseline["total_spend_millions_clp"],
    )
    logger.info("Average Unit-Price Dispersion (CV): %.2f%%", baseline["avg_coef_variation"] * 100)
    logger.info(
        "Historical Concentration: Portfolio HHI = %.1f | Avg Molecule HHI = %.1f",
        baseline["hhi_market_concentration"],
        baseline["avg_product_hhi"],
    )

    opt_res = optimizer.solve_procurement_optimization(
        max_supplier_share=0.45,
        min_reliability_score=70.0,
    )
    logger.info("HiGHS LP Convergence: %s (%s)", opt_res["success"], opt_res["solver_status"])
    logger.info(
        "Optimized Spend (To-Be): $%.2fM USD ($%.1fM CLP)",
        opt_res["optimized_spend_usd"] / 1e6,
        opt_res["optimized_spend_clp"] / 1e6,
    )
    logger.info(
        "Projected Net Fiscal Savings: $%.2fM USD / $%.1fM CLP (%.2f%%)",
        opt_res["total_savings_millions_usd"],
        opt_res["total_savings_millions_clp"],
        opt_res["savings_pct"],
    )
    logger.info(
        "Supply Chain Resilience Score: %.1f/100 (Baseline: %.1f/100 | Avg %.2f suppliers/drug)",
        opt_res["resilience_score"],
        opt_res["baseline_resilience_score"],
        opt_res["avg_suppliers_per_drug"],
    )
