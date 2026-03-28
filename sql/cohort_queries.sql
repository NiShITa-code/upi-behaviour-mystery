-- ============================================================
-- UPI Behaviour Mystery — BigQuery Cohort Queries
-- ============================================================
-- These are the production BigQuery equivalents of the pandas
-- operations in src/analysis/cohorts.py. In a real deployment,
-- these run against the data warehouse; the pandas code is
-- used for local development and testing.
-- ============================================================


-- 1. RETENTION CURVES BY ARCHETYPE
-- How quickly does each archetype drop off?
-- Used to set early-warning thresholds per segment.

WITH retention_windows AS (
    SELECT day_cutoff
    FROM UNNEST([14, 30, 60, 90, 180, 365]) AS day_cutoff
),
active_users AS (
    SELECT
        u.archetype,
        rw.day_cutoff,
        COUNT(DISTINCT t.user_id) AS active_users
    FROM `project.dataset.users` u
    CROSS JOIN retention_windows rw
    LEFT JOIN `project.dataset.transactions` t
        ON u.user_id = t.user_id
        AND t.day < rw.day_cutoff
    GROUP BY u.archetype, rw.day_cutoff
),
total_users AS (
    SELECT
        archetype,
        COUNT(*) AS n_users
    FROM `project.dataset.users`
    GROUP BY archetype
)
SELECT
    a.archetype,
    a.day_cutoff,
    ROUND(a.active_users * 100.0 / t.n_users, 2) AS retention_pct,
    t.n_users
FROM active_users a
JOIN total_users t USING (archetype)
ORDER BY a.archetype, a.day_cutoff;


-- 2. DAY-14 ENGAGEMENT GAP (RETAINED vs CHURNED)
-- The core finding: users who transact heavily in first 14 days
-- retain at dramatically higher rates.

SELECT
    CASE WHEN churned = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    COUNT(*)                                                   AS n_users,
    APPROX_QUANTILES(txn_d14, 100)[OFFSET(50)]                AS median_txn_d14,
    APPROX_QUANTILES(value_d14, 100)[OFFSET(50)]              AS median_value_d14,
    ROUND(AVG(has_first_p2m_d14) * 100, 1)                    AS pct_with_p2m_d14,
    APPROX_QUANTILES(cat_diversity, 100)[OFFSET(50)]           AS median_cat_diversity
FROM `project.dataset.users`
GROUP BY status;


-- 3. TRANSACTION CATEGORY BREAKDOWN
-- Which payment categories drive volume vs value?

SELECT
    category,
    COUNT(*)                                AS transaction_count,
    ROUND(SUM(value) / 1e7, 2)             AS total_value_crore,
    ROUND(AVG(value), 0)                   AS avg_ticket,
    ROUND(AVG(is_p2m), 3)                  AS p2m_share
FROM `project.dataset.transactions`
GROUP BY category
ORDER BY transaction_count DESC;


-- 4. CHURN RATE BY CITY TIER
-- Do metro users behave differently from tier-2/3?

SELECT
    CASE city_tier
        WHEN 1 THEN 'Tier 1 (Metro)'
        WHEN 2 THEN 'Tier 2'
        WHEN 3 THEN 'Tier 3'
    END                                     AS city_tier_label,
    COUNT(*)                                AS n_users,
    ROUND(AVG(churned), 4)                  AS churn_rate,
    ROUND(AVG(txn_d14), 2)                  AS avg_txn_d14,
    ROUND(AVG(p2m_ratio), 4)               AS avg_p2m_ratio,
    ROUND(AVG(total_value), 2)             AS avg_ltv
FROM `project.dataset.users`
GROUP BY city_tier
ORDER BY city_tier;


-- 5. AGE GROUP ANALYSIS
-- Behavioural differences across age cohorts.

SELECT
    age_group,
    COUNT(*)                                AS n_users,
    ROUND(AVG(churned), 4)                  AS churn_rate,
    ROUND(AVG(txn_d14), 2)                  AS avg_txn_d14,
    ROUND(AVG(p2m_ratio), 4)               AS avg_p2m_ratio
FROM `project.dataset.users`
GROUP BY age_group
ORDER BY age_group;


-- 6. FEATURE TABLE FOR MODEL TRAINING
-- This is the query that feeds the churn model.
-- Window functions compute early-engagement features.

SELECT
    user_id,
    txn_d7,
    value_d7,
    txn_d14,
    value_d14,
    has_first_p2m_d14,
    first_p2m_day,
    cat_diversity,
    p2m_ratio,
    city_tier,

    -- Derived features (same as src/features/engineer.py)
    ROUND(txn_d14 / 14.0, 4)                       AS txn_d14_per_day,
    IF(txn_d14 > 0, ROUND(value_d14 / txn_d14, 2), 0.0)
                                                     AS value_d14_per_txn,
    IF(txn_d7 > 0, ROUND(txn_d14 / txn_d7, 3), 1.0)
                                                     AS txn_acceleration,
    ROUND(LN(1 + value_d7), 4)                      AS log_value_d7,
    ROUND(LN(1 + value_d14), 4)                     AS log_value_d14,

    churned  -- target label
FROM `project.dataset.users`;


-- 7. UPLIFT MODEL — TREATMENT EFFECT BY SEGMENT
-- After the uplift model scores users, this query summarises
-- the ROI of targeting Persuadables vs random.

SELECT
    segment,
    COUNT(*)                                AS n_users,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)
                                            AS pct_of_total,
    ROUND(AVG(uplift), 4)                   AS avg_uplift,
    ROUND(AVG(p0), 4)                       AS avg_p0,
    ROUND(AVG(p1), 4)                       AS avg_p1
FROM `project.dataset.users_segmented`
GROUP BY segment
ORDER BY avg_uplift DESC;
