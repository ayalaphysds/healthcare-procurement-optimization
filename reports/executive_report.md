# EXECUTIVE REPORT: Mathematical Optimization of Public Pharmaceutical Procurement & Supply Chain Risk Mitigation

**Case Study Institution:** *Central de Abastecimiento del Sistema Nacional de Servicios de Salud* (**CENABAST** — National Health Supply Center, Ministry of Health of Chile)  
**Author:** Sebastián Ayala (`ayala.phys.ds`) — Senior Python, Machine Learning & Operations Research Engineer  
**Reference Exchange Rate:** `1 USD ≈ 950 CLP` (Chilean Pesos)  
**System:** *Public Healthcare Procurement Optimization & Supply Chain Analytics*

---

## 1. Executive Summary & International Context

**CENABAST** serves as the centralized public healthcare **Group Purchasing Organization (GPO)** and cold-chain distribution agency for Chile's Ministry of Health (**MINSAL**). It supplies essential pharmaceuticals, vaccines, oncology biologics, and surgical consumables to Chile's nationwide public hospital network (covering **~80% of the national population** insured under **FONASA**) as well as municipal primary-care pharmacies—representing **$1.13 Billion USD (1,074.32 Billion CLP)** in registered vendor ledger volume across **243 institutional entities** (**166 qualified pharmaceutical & medical supply companies**).

Using empirical **Interquartile Range (IQR) Fair-Price Band auditing** and **Constrained Linear Programming (`SciPy HiGHS`)**, this study evaluated a priority formulary of **18 high-impact essential medications and surgical supplies** (`1,600` purchase orders totaling **$106.32M USD / $101,000M CLP**):

1. **Structural Price Dispersion Identified:** Uncoordinated tender lots pay an average **2.05x max/min unit-price spread** (up to **2.33x** on cardiovascular statins and **>2.2x** on essential analgesics) for the exact same active ingredient and dosage, with an average Coefficient of Variation (CV) of **20.44%**.
2. **Verified Net Fiscal Savings (+22.97%):** Reallocating annual tender volumes via the constrained HiGHS Linear Programming solver reduces evaluated expenditure from **$106.32M USD ($101,000M CLP)** down to **$81.89M USD ($77,796M CLP)**—delivering **$24.43M USD ($23,204M CLP) in net fiscal savings** without reducing clinical unit volume by a single dose.
3. **Elimination of Single-Winner Stockout Fragility:** Enforcing a mandatory anti-monopoly split-award cap ($\le 45\%$ per supplier per medication) guarantees **3.00 qualified suppliers per drug**, slashing tender-level monopoly concentration by **-58.6%** (`10,000` single-winner HHI $\to$ `4,142` split-award HHI), keeping global portfolio concentration well inside the US DOJ/FTC competitive zone (**704.4 / 10,000**), and elevating the **Supply Chain Resilience Index from 51.9/100 to 91.8/100 (+76.9%)**.

---

## 2. Historical Baseline Diagnosis (`As-Is`)

### 2.1. Unit-Price Dispersion Across Identical Molecules
Auditing historical awards across competing laboratories revealed substantial price variation for identical generic and biologic specifications:

| Active Ingredient | Therapeutic Category | Min Price (USD / CLP) | Median Price (USD / CLP) | Max Price (USD / CLP) | Price CV (%) | Max / Min Spread |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ATORVASTATIN 20 MG** | Cardiovascular & Hypertension | $0.0355 / $33.75 | $0.0465 / $44.21 | $0.0829 / $78.75 | 24.8% | **2.33x** |
| **ACETAMINOPHEN 500 MG** | Analgesics & Antipyretics | $0.0178 / $16.88 | $0.0239 / $22.72 | $0.0397 / $37.67 | 22.9% | **2.23x** |
| **ENALAPRIL 10 MG** | Cardiovascular & Hypertension | $0.0146 / $13.88 | $0.0188 / $17.83 | $0.0316 / $30.05 | 25.0% | **2.16x** |
| **AMOXICILLIN 500 MG** | Antibiotics & Anti-Infectives | $0.0434 / $41.25 | $0.0588 / $55.85 | $0.0886 / $84.19 | 20.2% | **2.04x** |
| **AZITHROMYCIN 500 MG** | Antibiotics & Anti-Infectives | $0.1658 / $157.50 | $0.2231 / $211.96 | $0.3318 / $315.25 | 21.2% | **2.00x** |
| **IBUPROFEN 400 MG** | Analgesics & Antipyretics | $0.0268 / $25.50 | $0.0330 / $31.34 | $0.0521 / $49.51 | 21.9% | **1.94x** |
| **CEFTRIAXONE 1 G** | Antibiotics & Anti-Infectives | $0.9079 / $862.50 | $1.1952 / $1,135.46 | $1.6452 / $1,562.94 | 17.4% | **1.81x** |

Across the entire 18-item formulary, purchase orders awarded above the empirical 75th percentile (`P75`) generated **$6.16M USD ($5,852M CLP)** in excess overpricing relative to benchmark median bids alone.

### 2.2. Hospital SLA Delivery Delays & Single-Winner Vulnerability
- **SLA Delivery Breaches:** **25.7%** of historical purchase orders arrived past their contractual hospital delivery window (mean delay across all orders: **4.52 days**, with tail delays exceeding 30+ days).
- **Single-Supplier Exposure:** Under winner-take-all lot adjudication (`HHI = 10,000` per tender), any manufacturing halt or customs delay by the sole winning vendor immediately triggers regional hospital stockouts.

---

## 3. Constrained Linear Programming Architecture (`To-Be`)

### 3.1. Mathematical Formulation
Implemented in [`src/optimization_model.py`](../src/optimization_model.py) using `scipy.optimize.linprog` (`method="highs"`):

$$\min_{x \ge 0} \sum_{i=1}^{N} \sum_{j=1}^{M} c_{ij} \cdot x_{ij}$$

**Subject to:**
1. **100% Clinical Demand Fulfillment:** $\sum_{j=1}^{M} x_{ij} = D_i \quad \forall i$ (zero rationing of hospital demand).
2. **Anti-Monopoly & Anti-Stockout Diversification:** $0 \le x_{ij} \le \alpha \cdot D_i \quad \forall i, j$ (with $\alpha = 0.45$, guaranteeing at least $\lceil 1 / 0.45 \rceil = 3$ active suppliers per essential medication).
3. **Supplier Throughput Capacity:** $\sum_{i=1}^{N} x_{ij} \le \text{Cap}_j \quad \forall j$.
4. **Logistics SLA Pre-Qualification:** Exclusion of vendors with mean reliability score $< 70.0 / 100$.

---

## 4. Comparative Impact Summary (`As-Is` vs. `To-Be`)

| Performance Indicator | Historical Baseline (`As-Is`) | HiGHS LP Optimal (`To-Be`) | Verified Impact |
| :--- | :---: | :---: | :---: |
| **Evaluated Portfolio Spend (USD)** | $106.32M USD | **$81.89M USD** | **-$24.43M USD Saved** |
| **Evaluated Portfolio Spend (CLP)** | $101,000M CLP | **$77,796M CLP** | **-$23,204M CLP Saved** |
| **Net Fiscal Efficiency** | 0.00% (Baseline) | **22.97% Reduction** | **+22.97% Cost Efficiency** |
| **Max Supplier Share per Drug** | Up to 100% (Winner-Take-All) | **Capped at $\le 45.0\%$** | **3.00 Suppliers / Drug Guaranteed** |
| **Tender-Level Concentration (HHI)** | 10,000 pts (Single-Winner) | **4,142 pts (Split Award)** | **-58.6% Monopoly Risk** |
| **Global Portfolio HHI** | 519.5 pts | **704.4 pts** | **Competitive (< 1,500 DOJ/FTC)** |
| **Awarded Unit-Price Dispersion (CV)** | 20.44% | **1.09%** | **Price Standardization** |
| **Supply Chain Resilience Score** | 51.9 / 100 | **91.8 / 100** | **+76.9% Stockout Resilience** |

---

## 5. Strategic Policy & Engineering Recommendations

1. **Institutionalize Combinatorial Split-Award Tenders:** Replace winner-take-all lot adjudication with LP-cleared split contracts (`45% / 45% / 10%` or `45% / 35% / 20%`). Maintaining three active suppliers per molecule preserves competitive pricing pressure while providing immediate backup capacity if any vendor defaults.
2. **Enforce Automated P25–P75 Reference Price Guardrails:** Embed empirical IQR fair-price bands directly into tender evaluation APIs to require automatic Secondary Review whenever a bid exceeds the 75th percentile ceiling.
3. **Integrate SLA Reliability into Procurement Clearing:** Dynamically update supplier eligibility and effective bid weights based on rolling hospital warehouse on-time delivery performance.

---
*Prepared as part of the open-view engineering & operations research portfolio repository `healthcare-procurement-optimization`.*
