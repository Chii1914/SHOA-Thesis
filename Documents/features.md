
# SHO + LIME Strategy Specification

## 1. Observable Features for LIME

The following features are extracted **after each fitness evaluation** (`post-fitness evaluation`) for selected agents.

| Feature | Phase | Equation Relation | How It Is Computed | Role in Dynamics |
|---|---|---:|---|---|
| `r1` | Movement | Eq.(4) vs Eq.(7) | Random selector generated during movement (`r1(i)`) | Selects movement branch (helicoidal/Levy vs perturbation/Brownian). |
| `mag_browniano` | Movement | Eq.(7) | Magnitude of the Brownian perturbation term. Example: `abs(l * beta(i,j) * (Sea_horses(i,j) - beta(i,j) * Elite(i,j)))` | Controls perturbation intensity in Brownian movement. |
| `mag_levy` | Movement | Eq.(4) | Magnitude of Levy jump (`abs(Step_length(i,j))`) | Controls exploratory jump intensity. |
| `theta` | Movement | Eq.(4) | `theta = r * 2π` | Controls helicoidal geometry and movement angle. |
| `r2` | Predation | Eq.(10) | Random selector generated in predation (`r2(i)`) | Chooses predation branch (A vs B). |
| `mag_predacion` | Predation | Eq.(10) | Magnitude of stochastic predation term. Example: `abs(rand() * Sea_horses_new1(i,j))` | Controls predation scaling intensity. |
| `alpha` | Predation | Eq.(10) | Temporal coefficient: `alpha = (1 - t/Max_iter)^(2*t/Max_iter)` | Exploration–exploitation balancing factor. |
| `distance_to_elite` | Global Population State | Elite guidance | Euclidean distance to elite agent | Measures exploration distance relative to best solution. |

### Distance to Elite

The elite distance is computed as:

$$
d_i = ||X_i - X_{elite}||_2
$$

Where:

- `d_i`: distance of agent `i` to the elite.
- `X_i`: position vector of agent `i`.
- `X_elite`: elite (best solution found).
- `||·||_2`: Euclidean norm.

Interpretation:

- Small value → exploitation behavior.
- Large value → exploratory behavior.

---

## 2. Agent Types Selected for LIME

LIME is **not applied to the full population**. A stratified subset of agents is selected.

Recommended distribution:

- **Elite agents / high-impact:** 40%
- **Diverse agents:** 30%
- **Outliers / anomalies:** 20%
- **Random agents:** 10%

A minimum of **one agent per category** is guaranteed.

---

### 2.1 Elite / High-impact Agents

#### Purpose

Represent agents driving optimization progress.

#### Acceptance Criteria

Agents are selected according to:

1. Low fitness value.
2. Large fitness improvement.

Fitness improvement:

$$
\Delta f_i = f(x_{new}) - f(x_{old})
$$

Where:

- `Δf_i`: fitness change of agent `i`.
- `f(x_new)`: current fitness.
- `f(x_old)`: previous fitness.

Interpretation:

- Large negative value → strong improvement.
- Positive value → worsening.

Possible ranking score:

$$
S_i = w_1 rank(f_i) + w_2 rank(-\Delta f_i)
$$

Where:

- `S_i`: impact score.
- `w_1`, `w_2`: weighting coefficients.
- `rank(f_i)`: fitness ranking.
- `rank(-Δf_i)`: improvement ranking.

Interpretation:

Agents with lower `S_i` are considered high-impact.

---

### 2.2 Diverse Agents

#### Purpose

Represent exploratory behavior.

#### Acceptance Criteria

Agents with the largest distance to elite.

Computed as:

$$
d_i = ||X_i - X_{elite}||_2
$$

Where:

- `d_i`: diversity distance.
- `X_i`: agent position.
- `X_elite`: elite position.

Interpretation:

- High distance → exploration.
- Low distance → exploitation.

---

### 2.3 Outlier / Anomalous Agents

#### Purpose

Detect unusual optimization behavior.

#### Acceptance Criteria

Outliers are selected using a standardized improvement score:

$$
z_i = \frac{\Delta f_i - \mu}{\sigma}
$$

Where:

- `z_i`: anomaly score of agent `i`.
- `Δf_i`: fitness improvement of agent `i`.
- `μ` (mu): mean fitness improvement of the population.
- `σ` (sigma): standard deviation of fitness improvements.

Interpretation:

- `|z_i|` high → unusual behavior.
- Large negative value → unexpectedly strong improvement.
- Large positive value → unexpectedly poor movement.

Typical criterion:

$$
|z_i| > threshold
$$

or top-k largest absolute z-scores.

---

### 2.4 Random Agents

#### Purpose

Prevent selection bias.

#### Acceptance Criteria

Uniform random sampling from remaining population.

Interpretation:

Provides neutral population coverage and prevents overfitting explanations to elites or outliers.

---

## 3. Global Feature Aggregation

LIME explanations are aggregated globally using feature importances rather than raw feature values.

### Mean Feature Importance

$$
I_f = \frac{1}{N}\sum_i |w_{i,f}|
$$

Where:

- `I_f`: global importance of feature `f`.
- `N`: number of LIME explanations.
- `w_(i,f)`: importance weight of feature `f` in explanation `i`.

Interpretation:

Measures how globally important a feature is.

---

### Signed Feature Importance

$$
S_f = \frac{1}{N}\sum_i w_{i,f}
$$

Where:

- `S_f`: directional importance.
- Positive value → feature tends to help improvement.
- Negative value → feature tends to harm improvement.

---

### Temporal Aggregation

A sliding window of explanations may be used:

Example:

- last 50 explanations
- last 100 explanations

This enables monitoring:

- exploration → exploitation transition
- feature dominance
- behavioral collapse

---

## 4. Current SHOA-LIME Implementation Snapshot (May 2026)

This section documents the behavior currently implemented in code.

### LIME Trigger Condition

LIME is triggered only when both conditions hold:

1. Iteration condition: `t % lime_every == 0`
2. Data condition: `len(agent_samples) >= lime_min_samples`

Default values in the benchmark runner:

- `lime_every = 10`
- `lime_min_samples = 1000`

### Selected Candidate Pool Per Diagnosis

The candidate pool is still selected with the existing stratified policy:

- Elite/high-impact: 4% of population
- Diverse: 3% of population
- Outliers: 2% of population
- Random: 1% of population

With min one agent per category.

### Targets Kept in Production

Two target channels are kept active:

- `classification_improved`
- `regression_y_reg`

### Run Artifacts Used in Analysis

Core artifacts remain:

- `config_used.json`
- `runs_raw.csv`
- `full_output.csv`
- `lime_contributions.csv`
- `global_feature_explanations.csv`
- `summary_by_function.csv`

---

## 5. Selection Mode Decision and New Flag

### Decision

Default execution mode is now `medoid` (faster), while preserving `selected_agents` for methodological comparison.

### New CLI Contract

`--lime-selection-mode {medoid, selected_agents}`

- `medoid` (default): explain only one representative medoid vector from the selected pool.
- `selected_agents`: explain all selected agents in the pool.

### Propagation Path

The mode is propagated through the full pipeline:

`run_cec2022_benchmark.py` -> `SHO_LIME_Controller.py` -> `lime_diagnostic.py`

### Traceability Fields in Contributions

Each row in `lime_contributions.csv` now includes:

- `selection_mode`
- `selected_pool_size`
- `n_agents`

Interpretation:

- In `selected_agents`, `n_agents` is typically equal to `selected_pool_size`.
- In `medoid`, `n_agents = 1` and `selected_pool_size` keeps the original selected-group size.

### Cost Rationale

Let $k$ be selected agents per diagnosis and $m$ the number of LIME perturbation samples.

- `selected_agents` scales approximately with $O(k \cdot m)$ explanation calls per target.
- `medoid` scales approximately with $O(m)$ explanation calls per target.

For the current SHOA-LIME workflow, this reduces explanation overhead while preserving diagnosis cadence and feature schema.

---

## 6. SHOA-COMBINED Online Stage (LIME + Stagnation)

This section documents the new online combined stage currently implemented.

### Scope

- LIME diagnostics and stagnation detection run in the same optimization loop.
- Both components are active from the beginning of the run.
- `rescue_mode` is intentionally out of scope for this stage.

### Main module path

`Initial Implementations/SHOA-COMBINED/`

Primary files:

- `SHO_HYBRID_Controller.py`
- `run_cec2022_combined.py`

### Online flow per iteration

1. Run movement + predation updates and evaluate population.
2. Build per-agent feature rows used by LIME.
3. Update global stagnation detector with current `best_fitness` and `FE`.
4. Update LIME dataset only when the run is outside stagnation.
5. Trigger LIME only on event `stagnation_start` (if minimum sample size is met).
6. Record unified telemetry row in `full_output.csv`.

### Combined output contract

The combined runner writes both LIME and stagnation legacy artifacts:

- `config_used.json`
- `runs_raw.csv`
- `full_output.csv`
- `lime_contributions.csv`
- `global_feature_explanations.csv`
- `stagnation_history.csv`
- `stagnation_events.csv`
- `summary_by_function.csv`

### LIME policy in combined runner

- Trigger policy: `stagnation_start_only`.
- During stagnation (`stagnated=1`) no new samples are added to LIME dataset.
- Outside stagnation, sampling resumes automatically.
- Explanation window is accumulated from global run start, excluding stagnated intervals.

### LIME cadence field (legacy compatibility)

- If `--lime-every` is omitted, effective cadence is computed once at startup:

$$
lime\_every = \max(1, \lceil 0.05 \cdot max\_iter \rceil)
$$

- If `--lime-every` is provided, it is stored for traceability but does not activate periodic explanations in this stage.

### Traceability additions in full output

`full_output.csv` includes both channels in one row schema, including:

- LIME fields: `lime_selection_mode`, `lime_triggered`, `lime_trigger_source`, `diagnosis_id`, `lime_dataset_updated`, `lime_buffer_size`
- Stagnation fields: `fe`, `max_fes`, `sfes`, `min_sfes`, `stagnated`, `event`

This allows existing convergence and explanation pipelines to consume one unified run output.
