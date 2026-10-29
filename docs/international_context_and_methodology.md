# 🌎 International Context, Domain Glossary & Mathematical Methodology

> **Audience:** International hiring managers, technical interviewers (e.g., US/global remote engineering teams), operations researchers, and healthcare systems engineers unfamiliar with Chilean public-sector institutions.

---

## 1. Why Does This Project Exist? (Real-World Problem & Social Impact)

### 1.1. Healthcare Structure in Chile
Chile operates a dual healthcare system serving approximately **20 million residents**:
- **FONASA (*Fondo Nacional de Salud*):** The public national health insurance fund covering **~80% of the population (~16M+ beneficiaries)** through a nationwide network of public hospitals, regional health services (*Servicios de Salud*), and municipal primary-care clinics (*farmacias comunales* / *CESFAM*).
- **ISAPREs:** Private health insurers serving the remaining ~15–18% of residents.

At the center of the entire public supply chain sits **CENABAST** (*Central de Abastecimiento del Sistema Nacional de Servicios de Salud*), a federal agency under Chile's Ministry of Health (**MINSAL**).

### 1.2. International Analogies for CENABAST
If you are based in the United States, Europe, or Asia, you can think of **CENABAST** as a hybrid of:
- **US Veterans Affairs National Acquisition Center (VA NAC)** + a massive hospital **Group Purchasing Organization (GPO)** like Vizient or Premier, combined with national cold-chain warehousing and distribution.
- **UK NHS Supply Chain / Commercial Medicines Unit (CMU)**, negotiating centralized framework agreements for public NHS trusts.

CENABAST intermediates over **$1.13B USD (~1.07 Trillion CLP)** in registered vendor procurement volume across chronic-care generics, antibiotics, vaccines, high-cost oncology biologics, and critical surgical consumables.

---

## 2. Three Structural Failures in Traditional Public Tenders

Despite massive buyer leverage, traditional public-sector pharmaceutical procurement suffers from three recurring failure modes—both in Latin America and globally:

1. **Severe Unit-Price Dispersion (Fiscal Leakage):**
   When hospitals purchase outside centralized schedules—or when emergency/spot tenders are awarded across fragmented lots—the state pays up to **2.1x–2.4x different unit prices** for the **exact same active ingredient, dosage, and clinical presentation**.
2. **Winner-Take-All Stockout Fragility (Patient Safety Risk):**
   Conventional public procurement rules often award **100% of an annual tender line to the single lowest-priced bidder**. If that manufacturer encounters an active pharmaceutical ingredient (API) shortage, customs hold, or quality recall, public hospitals experience immediate **medication stockouts**, delaying surgeries or interrupting chronic/oncology treatments.
3. **Excessive Supplier Concentration (Antitrust / HHI Risk):**
   High administrative barriers concentrate contract volume in a handful of incumbents, eroding long-term market contestability as measured by the **Herfindahl-Hirschman Index (HHI)**.

---

## 3. Institutional & Financial Glossary for International Readers

| Chilean Term / Acronym | English Translation & Equivalent | Role in This Codebase |
| :--- | :--- | :--- |
| **CENABAST** | National Health Supply Center | Centralized public healthcare GPO whose vendor ledger anchors the pipeline |
| **MINSAL** | Ministry of Health of Chile | Federal cabinet ministry overseeing public hospitals & CENABAST |
| **FONASA** | National Health Fund | Public payer insuring ~80% of Chile's population |
| **RUT** (*Rol Único Tributario*) | Chilean Corporate Tax ID (analogous to US **EIN** or EU **VAT ID**) | Mapped to `supplier_tax_id` in [`src/data_processing.py`](../src/data_processing.py) |
| **CLP** (*Peso Chileno*) | Chilean Peso (`1 USD ≈ 950 CLP`) | Raw government ledger currency; automatically paired with `USD` across the entire pipeline and UI |
| **Licitación Pública / OC** | Public Tender / Purchase Order (PO) | Modeled via `tender_id` and `purchase_order_id` |

---

## 4. Data Provenance & Hybrid Engineering Architecture

Real-world government open data is rarely delivered as a clean, ML-ready table. This repository demonstrates how to engineer a reproducible analytical system around imperfect public disclosures:

1. **Authentic Institutional Anchor (`data/raw/cenabast_suppliers_2026_raw.csv`):**
   Contains **243 real entities** registered in CENABAST's supplier ledger (`;`-delimited UTF-8-BOM CSV with Spanish headers), totaling **1,074.32 Billion CLP (~$1.13 Billion USD)** in transacted public spend (**166 clinical pharmaceutical & medical supply vendors** after entity screening), including global pharmaceutical manufacturers (*Pfizer, Roche, Novartis, Sanofi, AstraZeneca, AbbVie, Bayer, GSK, Merck, Sandoz, Fresenius Kabi, Baxter, Sinovac, Novo Nordisk*) and major Chilean laboratories (*Laboratorio Chile, Andrómaco, Saval, Biosano, Sanderson, Scienza, Novofarma*).
2. **Domain Entity Classification (`filter_clinical_suppliers`):**
   Raw government vendor ledgers mix **clinical pharmaceutical suppliers** with **internal agency operational vendors** (postal services, staff childcare centers, building security, fuel, IT services). [`src/data_processing.py`](../src/data_processing.py) applies deterministic domain filtering to isolate qualified clinical laboratories and medical distributors before computing market-share weights.
3. **Calibrated Line-Item Disaggregation & IQR Winsorization:**
   Because aggregate vendor disclosures do not expose unified hospital delivery SLA timestamps per molecule in a single file, the pipeline disaggregates supplier volume across an **18-item WHO/CENABAST essential formulary** preserving empirical vendor market shares, realistic bid dispersion ($\sigma \in [0.12, 0.40]$), and log-normal delivery delay shocks. A **2.5x Interquartile Range (IQR)** soft winsorizer caps extreme unit-price data-entry anomalies per drug without discarding valid high-spread tender awards.

---

## 5. Mathematical Optimization Formulation (`SciPy HiGHS`)

### 5.1. Sets & Parameters
- $i \in \{1, \dots, N\}$: Essential medications and hospital supplies ($N = 18$ priority formulary items).
- $j \in \{1, \dots, M\}$: Pre-qualified clinical suppliers satisfying $\text{Reliability}_j \ge \tau$ (default $\tau = 70.0 / 100$).
- $D_i > 0$: Aggregate clinical unit demand for medication $i$.
- $c_{ij} > 0$: Unit price offered by supplier $j$ for medication $i$ (historical median bid, or market median $\times 1.05$ switching-cost penalty for unbid pairs).
- $\alpha \in [0.25, 0.80]$: Regulatory anti-monopoly split-award cap (default $\alpha = 0.45$, forcing at least $\lceil 1 / 0.45 \rceil = 3$ suppliers per medication).
- $\text{Cap}_j > 0$: Maximum operational unit throughput capacity for supplier $j$.

### 5.2. Linear Program
Decision variable $x_{ij} \in \mathbb{R}_{\ge 0}$ represents the quantity of medication $i$ awarded to supplier $j$:

$$\min_{x} \sum_{i=1}^{N} \sum_{j=1}^{M} c_{ij} \, x_{ij}$$

Subject to:

$$\sum_{j=1}^{M} x_{ij} = D_i \qquad \forall i \in \{1, \dots, N\} \quad \text{(100\% Clinical Demand Coverage)}$$

$$0 \le x_{ij} \le \alpha \, D_i \qquad \forall i, j \quad \text{(Anti-Monopoly & Stockout Diversification Cap)}$$

$$\sum_{i=1}^{N} x_{ij} \le \text{Cap}_j \qquad \forall j \in \{1, \dots, M\} \quad \text{(Supplier Throughput Capacity)}$$

### 5.3. Solver Implementation Details
- Solved in [`ProcurementOptimizer.solve_procurement_optimization`](../src/optimization_model.py) via `scipy.optimize.linprog(..., method="highs")`, invoking the C++ **HiGHS** dual-simplex/interior-point engine.
- **Compact Matrix Construction:** Per-variable upper bounds $x_{ij} \le \alpha D_i$ are passed directly as variable box bounds `(0, alpha * D_i)` rather than inflating the dense inequality matrix $A_{\text{ub}}$, reducing constraint rows from $N \cdot M + M$ down to just $M$ inequality rows plus $N$ equality rows—enabling sub-50ms interactive re-optimization inside Streamlit as users drag policy sliders.
- **Mathematical Feasibility Guard:** For any split-award cap $\alpha$, a feasible solution requires at least $M \ge \lceil 1 / \alpha \rceil$ qualified suppliers (`min_suppliers_needed = max(3, ceil(1 / alpha) + 1)`). If an overly strict SLA cutoff leaves fewer suppliers, the engine automatically retains the top-ranked suppliers by reliability so the LP never crashes on user slider extremes.

---

## 6. Market Concentration (HHI) & Resilience Metrics

- **Herfindahl-Hirschman Index (HHI):**
  $$\text{HHI} = \sum_{j=1}^{M} s_j^2, \qquad s_j = 100 \times \frac{\text{Spend}_j}{\sum_{k=1}^{M} \text{Spend}_k} \in [0, 100]$$
  Ranges from near $0$ (atomistic competition) to $10,000$ (pure monopoly). Both portfolio-level HHI and average molecule-level HHI are tracked.
- **Supply Chain Resilience Index ($0\text{–}100$):**
  Combines multi-sourcing depth (average active suppliers per medication benchmarked against a 3-supplier target, weighted $60\%$) and spend-weighted supplier SLA delivery reliability (weighted $40\%$).

