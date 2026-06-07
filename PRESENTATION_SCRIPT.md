# Pharma Demand Forecasting & Intelligent Replenishment System
## 10-Minute Demo Presentation Script

---

## Slide 1: Opening / Introduction (1 min)

**[On screen: Title slide]**

"Good morning/afternoon everyone. Thank you for being here today.

My name is [Your Name], and I'm presenting our team's work on **Case 2: Pharmaceutical Demand Forecasting and Intelligent Replenishment System**.

**Context:** We are building a prototype for a large pharmaceutical B2B e-commerce platform in China. The platform manages **500 SKUs** across multiple distribution channels — hospitals, chain pharmacies, and independent pharmacies.

**The challenge:** Pharmaceutical demand is highly volatile. It's influenced by seasonal flu outbreaks, national drug procurement policies, holidays, and channel-specific dynamics. Traditional inventory management struggles to keep up.

**Our solution:** A data-driven system that combines **demand forecasting** with **intelligent replenishment recommendations**, all packaged in an interactive Streamlit dashboard.

Today I'll walk you through the system in about 10 minutes. Let's get started."

---

## Slide 2: System Architecture (1.5 min)

**[On screen: Architecture diagram]**

"Before diving into the demo, let me quickly explain our system architecture.

**At the core**, we have **four predictive models** that are automatically routed based on each SKU's demand pattern:

1. **ETS (Exponential Smoothing)** — for fast-moving, stable SKUs. About 8% of our portfolio.
2. **Prophet** — for seasonal SKUs with clear cyclical patterns. About 9%.
3. **Croston / SBA** — for long-tail, intermittent-demand SKUs. This covers the majority, about 80%.
4. **XGBoost** — for policy-shocked SKUs affected by government procurement programs. About 3%.

**The demand engine** combines base demand with trend, seasonality, flu activity, VBP policy shocks, and holiday effects.

**The output** feeds into our Streamlit dashboard with five functional modules: Demand Forecasting, Intelligent Replenishment, Inventory Health, Policy Simulation, and Alerts.

Let's see it in action."

---

## Slide 3: Module 1 — Demand Forecasting Dashboard (2 min)

**[Switch to Streamlit app — Page 1: Demand Forecasting]**

"First, our **Demand Forecasting Dashboard**.

**[Point to SKU selector]** Users can select any of the 500 SKUs and choose a channel — Total, Hospital, Chain Pharmacy, or Independent Pharmacy.

**[Point to the line chart]** The system displays historical sales as a solid blue line and our 90-day forecast as a dashed red line, with a clear vertical cutoff between history and prediction. The historical data spans 12 months, and the forecast covers June through August 2026.

**[Point to channel split chart]** Below that, we show a stacked area chart breaking down the forecast by channel — hospital in blue, chain pharmacy in green, and independent pharmacy in orange. This helps managers understand where demand is coming from.

**[Point to accuracy metrics]** For each SKU, we display forecast accuracy metrics — MAPE and RMSE — so users can judge prediction reliability.

**[Point to feature importance]** And here's something especially valuable for stakeholders who need explainability: the **Prediction Factor Analysis**. Using XGBoost feature importance, we show which factors drive each SKU's demand — things like flu activity index, holiday effects, 7-day lagged sales, and VBP policy markers. This builds trust in the model's recommendations."

---

## Slide 4: Module 2 — Intelligent Replenishment (2 min)

**[Switch to Streamlit app — Page 2: Replenishment]**

"Next, the **Intelligent Replenishment** module. This is where predictions turn into actionable decisions.

**[Point to channel selector]** A key feature: users can view replenishment recommendations not just for total demand, but **channel by channel**. When you switch from 'Total' to 'Hospital,' the system recalculates everything — current inventory split by channel share, safety stock, reorder points, and suggested order quantities.

**[Point to the three metric cards]** At the top, three status cards show the count of SKUs in each priority tier: Urgent, Need Replenish, and Sufficient. This gives an instant health snapshot.

**[Scroll to the priority list table]** The priority list table shows each SKU's current inventory, safety stock, days of availability, suggested order quantity, and order value. Users can filter by priority, therapy area, or search by drug name.

**[Point to Best Order Timing column]** Here's a feature I'm particularly proud of: **Best Order Timing**. The system calculates the estimated stockout date and the suggested order date. If a drug needs to be ordered immediately, it shows 'Order Now' in red. Otherwise, it displays the exact date and days remaining. This helps procurement teams plan their purchase schedules efficiently."

---

## Slide 5: Module 3 — Inventory Health Diagnosis (2 min)

**[Switch to Streamlit app — Page 3: Inventory Health]**

"Now let's look at **Inventory Health Diagnosis**. This module helps identify structural problems in the inventory portfolio.

**[Point to inventory structure cards and pie charts]** At the top, we show inventory structure analytics — total inventory value, total quantity, healthy SKU count, and slow-moving SKU count. Two donut charts break this down by ABC classification and therapy area, so managers can see where capital is tied up.

**[Point to ABC/XYZ scatter plot]** The ABC/XYZ scatter plot is a classic inventory management tool. Each dot is one SKU. The X-axis is revenue contribution, the Y-axis is demand variability. AX items are high-value, stable — these are your priority auto-replenishment candidates. CZ items are low-value, highly variable — candidates for elimination or consolidation.

**[Scroll to Fast-moving Items]** Below that, we automatically identify **fast-moving items** — SKUs with high daily sales and fast turnover. The system recommends 'Priority Replenish' and 'Increase Safety Stock' for these.

**[Scroll to Slow-moving Items]** And **slow-moving items** — SKUs with very low sales and turnover over 180 days. For items over 360 days, we recommend 'Promote / Clear.' For others, 'Reduce Order / Return.'

**[Point to Reallocation section]** Most importantly, the system quantifies **excess inventory** from slow-movers and **stock shortages** from fast-movers, then proposes reallocation suggestions. For example, it might say: 'Reallocate 1,500 boxes from 12 slow-moving SKUs to 8 fast-moving SKUs, covering 85% of the shortage.' This turns inventory analysis into concrete transfer recommendations."

---

## Slide 6: Modules 4 & 5 — Policy Simulator & Alerts (1 min)

**[Switch to Streamlit app — Page 4: Policy Simulator]**

"The **Policy Impact Simulator** allows users to model the effects of government drug procurement policies.

**[Point to scenario selector and sliders]** Users select a VBP batch, adjust price reduction and volume uplift assumptions, and instantly see the before-and-after revenue impact. This supports strategic planning when negotiating procurement contracts.

**[Switch to Page 5: Alerts]**

"Finally, the **Alerts** module provides at-a-glance risk notifications.

**[Point to stockout alerts]** Red alert cards flag SKUs at risk of stockout within days, with suggested reorder quantities.

**[Point to overstock alerts]** Yellow cards highlight inventory overstock — SKUs with more than 3 months of supply."

---

## Slide 7: Closing / Value Summary (0.5 min)

**[On screen: Key takeaways]**

"To summarize, our system delivers **three core values**:

**First**, **accurate demand forecasting** across 500 SKUs and four channels, using model routing tailored to each SKU's demand pattern.

**Second**, **actionable replenishment recommendations** with channel-level granularity and precise timing guidance.

**Third**, **inventory structural optimization** through automated identification of fast-movers, slow-movers, and near-expiry items, with quantified reallocation proposals.

The entire system runs in a lightweight Streamlit web application, making it accessible to inventory managers without requiring technical expertise.

Thank you very much. I'd be happy to take any questions."

---

## Appendix: Timing Checklist

| Section | Time | Cumulative |
|---------|------|------------|
| Opening / Introduction | 1:00 | 1:00 |
| System Architecture | 1:30 | 2:30 |
| Demand Forecasting | 2:00 | 4:30 |
| Intelligent Replenishment | 2:00 | 6:30 |
| Inventory Health | 2:00 | 8:30 |
| Policy Simulator & Alerts | 1:00 | 9:30 |
| Closing & Q&A buffer | 0:30 | 10:00 |

## Demo Navigation Tips

During the live demo, keep the Streamlit app open and practice these transitions:
1. **Page 1 → Page 2:** Click sidebar "Replenishment"
2. **Page 2:** Switch channel dropdown from "Total" → "Hospital" to show dynamic recalculation
3. **Page 2 → Page 3:** Click sidebar "Inventory Health"
4. **Page 3:** Scroll down to show Fast-moving → Slow-moving → Reallocation sections
5. **Page 3 → Page 4:** Click sidebar "Policy Simulator"
6. **Page 4 → Page 5:** Click sidebar "Alerts"
7. **Return to Page 1** for closing if needed

## Suggested Q&A Preparation

**Q: Why four models instead of one?**
A: Different demand patterns require different approaches. ETS works well for stable demand, but fails on intermittent long-tail items where Croston is better. XGBoost captures complex policy interactions that time-series models miss.

**Q: How is the model accuracy?**
A: Average MAPE varies by SKU type. Fast-moving items typically achieve under 15% MAPE. The dashboard displays per-SKU accuracy metrics transparently.

**Q: Can this system integrate with an ERP?**
A: Yes. The system reads standard CSV files and outputs replenishment recommendations in CSV format. Integration would require API connectors to the specific ERP system.

**Q: How often should forecasts be refreshed?**
A: Daily for high-velocity SKUs, weekly for long-tail items. The system can be scheduled to rerun the model pipeline automatically.
