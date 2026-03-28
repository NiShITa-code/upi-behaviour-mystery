# SQL Queries

Production BigQuery equivalents of the pandas operations in `src/analysis/`.

In a real deployment, these queries run against the data warehouse.
The pandas code in `src/` is used for local development and testing —
the interfaces are identical.

## Query Index

| # | Query | Pandas Equivalent | Purpose |
|---|-------|-------------------|---------|
| 1 | Retention curves | `cohorts._retention_curves()` | Archetype drop-off over time |
| 2 | Day-14 gap | `cohorts._day14_gap()` | Retained vs churned engagement |
| 3 | Category breakdown | `cohorts._category_breakdown()` | Volume and value by payment type |
| 4 | City tier analysis | `cohorts._city_tier_breakdown()` | Geographic behaviour differences |
| 5 | Age group analysis | `cohorts._age_group_breakdown()` | Demographic segmentation |
| 6 | Feature table | `features/engineer.py` | Model training input with derived features |
| 7 | Uplift segments | `models/uplift.py` | ROI summary by treatment segment |
