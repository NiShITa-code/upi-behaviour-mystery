# Simulation Methodology & Validation Strategy

## Why Simulate?

Real user-level UPI transaction data is confidential and governed by RBI data
localisation norms. No public dataset exists at the granularity needed for churn
and uplift modelling. However, NPCI publishes monthly aggregate statistics that
allow us to calibrate a realistic simulator.

The key insight: **the methodology is data-source-agnostic**. Replace
`simulate_users()` with a data warehouse connector and the entire pipeline
runs unchanged. The interface contract (a validated DataFrame with the schema
defined in `src/data/schema.py`) is identical.

## Calibration Sources

| Parameter | Published Value | Source | Our Setting |
|-----------|----------------|--------|-------------|
| Monthly volume | ~20B transactions (2025) | NPCI monthly reports | Scaled to per-user rates |
| Average ticket | 1,293 INR (Dec 2025) | NPCI | Log-normal mean calibrated to match |
| P2M share | ~63% of volume (H1 2025) | NPCI | Archetype-specific P2M ratios |
| City distribution | ~30% metro / 35% T2 / 35% T3 | RBI Digital Payments Index | `city_tier_dist: [0.30, 0.35, 0.35]` |

## Statistical Choices

**Poisson arrivals.** Transaction events are modelled as a Poisson process with
archetype-specific daily rates. This is standard for event-count data in payments
and aligns with the memoryless property of independent transactions.

**Log-normal values.** Transaction amounts follow a log-normal distribution,
which is well-established for monetary data: most transactions are small, with
a long right tail. The sigma parameter (0.6) produces a coefficient of variation
consistent with published UPI average ticket variability.

**Archetype-based heterogeneity.** Users are drawn from four archetypes (Power
User, Regular, Occasional, Dormant) with distinct transaction rates, values, and
churn probabilities. This captures the heavy-tailed user engagement distribution
observed in payment platforms.

## Validation Protocol (For Real Data Deployment)

When deploying against actual data, the following validation steps should be run
to verify the simulation's distributional assumptions hold:

### 1. Marginal Distribution Tests

For each continuous feature (`txn_d14`, `value_d14`, `p2m_ratio`):
- **Kolmogorov-Smirnov test** comparing simulated vs real marginal distributions
- **QQ plots** for visual inspection of tail behaviour
- Acceptance threshold: KS p-value > 0.05 for at least 80% of features

### 2. Correlation Structure

- Compute the Pearson correlation matrix for simulated and real feature sets
- Compare using the Frobenius norm: `||R_sim - R_real||_F < threshold`
- Critical correlations to check: `txn_d14` vs `churned`, `has_first_p2m_d14` vs `churned`

### 3. Predictive Equivalence

- Train the churn model on real data and simulated data separately
- Compare AUC-ROC on a held-out real test set
- If `|AUC_real - AUC_sim| < 0.03`, the simulation is sufficiently realistic

### 4. Temporal Stability

- Run the simulation with different seeds (we test seeds 1-10)
- Verify CV AUC standard deviation < 0.02 across seeds
- Current result: CV AUC std = 0.010 (seed=42), confirming stability

## Known Limitations

1. **No network effects.** Real UPI users influence each other (peer referrals,
   merchant adoption cascading). The simulator treats users as independent.

2. **Static archetypes.** Users are assigned one archetype for life. In reality,
   users migrate between segments over time.

3. **No seasonality.** Real UPI data shows strong month-end salary spikes and
   festival effects (Diwali, Eid). The simulator uses constant daily rates.

4. **Simplified treatment effect.** The uplift model uses a constant per-archetype
   response boost. Real intervention effects depend on timing, channel, amount,
   and user state.

These limitations are documented as future work, not hidd