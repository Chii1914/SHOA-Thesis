"""Agent selection and LIME explanations for SHOA-LIME."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass
class SelectionResult:
    category_indices: dict[str, list[int]]
    selected_unique: list[int]


def _rank_ascending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(values))
    return rank


def selection_counts_from_population(pop_size: int) -> dict[str, int]:
    return {
        "elite_high_impact": max(1, int(round(pop_size * 0.04))),
        "diverse": max(1, int(round(pop_size * 0.03))),
        "outliers": max(1, int(round(pop_size * 0.02))),
        "random": max(1, int(round(pop_size * 0.01))),
    }


def _pick_indices(
    ordered_candidates: np.ndarray,
    count: int,
    used: set[int],
    allow_overlap_when_needed: bool,
) -> list[int]:
    selected: list[int] = []

    for idx in ordered_candidates:
        idx_int = int(idx)
        if idx_int not in used:
            selected.append(idx_int)
        if len(selected) >= count:
            return selected

    if not allow_overlap_when_needed:
        return selected

    for idx in ordered_candidates:
        idx_int = int(idx)
        if idx_int not in selected:
            selected.append(idx_int)
        if len(selected) >= count:
            return selected

    return selected


def select_agents_for_lime(
    fitness_new: np.ndarray,
    delta_f: np.ndarray,
    distance_to_elite: np.ndarray,
    rng: np.random.Generator,
) -> SelectionResult:
    n = len(fitness_new)
    if n == 0:
        return SelectionResult(category_indices={}, selected_unique=[])

    counts = selection_counts_from_population(n)
    allow_overlap = n < 4

    fitness_rank = _rank_ascending(fitness_new)
    improvement_rank = _rank_ascending(-delta_f)
    impact_score = fitness_rank + improvement_rank

    elite_order = np.argsort(impact_score)
    diverse_order = np.argsort(-distance_to_elite)

    std_delta = float(np.std(delta_f))
    if std_delta > 0:
        z_scores = (delta_f - float(np.mean(delta_f))) / std_delta
    else:
        z_scores = np.zeros_like(delta_f)
    outlier_order = np.argsort(-np.abs(z_scores))

    random_order = rng.permutation(n)

    used: set[int] = set()
    categories: dict[str, list[int]] = {}

    categories["elite_high_impact"] = _pick_indices(elite_order, counts["elite_high_impact"], used, allow_overlap)
    used.update(categories["elite_high_impact"])

    categories["diverse"] = _pick_indices(diverse_order, counts["diverse"], used, allow_overlap)
    used.update(categories["diverse"])

    categories["outliers"] = _pick_indices(outlier_order, counts["outliers"], used, allow_overlap)
    used.update(categories["outliers"])

    categories["random"] = _pick_indices(random_order, counts["random"], used, allow_overlap)
    used.update(categories["random"])

    selected_unique = sorted({idx for values in categories.values() for idx in values})
    return SelectionResult(category_indices=categories, selected_unique=selected_unique)


def _weights_to_dense(feature_count: int, sparse_pairs: list[tuple[int, float]]) -> np.ndarray:
    dense = np.zeros(feature_count, dtype=float)
    for feature_idx, weight in sparse_pairs:
        dense[int(feature_idx)] = float(weight)
    return dense


def _compute_medoid_matrix(x_selected: np.ndarray) -> np.ndarray:
    """Return a single-row matrix containing the medoid of x_selected."""
    if x_selected.shape[0] <= 1:
        return x_selected

    diff = x_selected[:, np.newaxis, :] - x_selected[np.newaxis, :, :]
    distance_sum = np.linalg.norm(diff, axis=2).sum(axis=1)
    medoid_index = int(np.argmin(distance_sum))
    return x_selected[[medoid_index], :]


def _build_lime_rows_for_target(
    target_type: str,
    rows_per_agent: list[np.ndarray],
    feature_columns: list[str],
    run_metadata: dict,
    diagnosis_id: int,
    iteration: int,
    n_agents: int,
    selection_mode: str,
    selected_pool_size: int,
) -> list[dict]:
    if not rows_per_agent:
        return []

    stacked = np.vstack(rows_per_agent)
    mean_signed = np.mean(stacked, axis=0)
    mean_abs = np.mean(np.abs(stacked), axis=0)

    output: list[dict] = []
    for idx, feature in enumerate(feature_columns):
        output.append(
            {
                "run_id": run_metadata["run_id"],
                "timestamp": run_metadata["timestamp"],
                "function_name": run_metadata["function_name"],
                "dimension": run_metadata["dimension"],
                "iteration": iteration,
                "diagnosis_id": diagnosis_id,
                "target_type": target_type,
                "feature": feature,
                "mean_weight": float(mean_signed[idx]),
                "mean_abs_weight": float(mean_abs[idx]),
                "n_agents": n_agents,
                "selection_mode": selection_mode,
                "selected_pool_size": selected_pool_size,
            }
        )
    return output


def explain_selected_agents(
    all_samples: list[dict],
    current_samples: list[dict],
    selected_indices: list[int],
    feature_columns: list[str],
    run_metadata: dict,
    diagnosis_id: int,
    iteration: int,
    random_state: int,
    selection_mode: str = "selected_agents",
) -> list[dict]:
    if not selected_indices:
        return []

    x_train = np.array([[float(row[c]) for c in feature_columns] for row in all_samples], dtype=float)
    if x_train.shape[0] < 2:
        return []

    current_by_agent = {int(row["agent_id"]): row for row in current_samples}
    selected_rows = [current_by_agent[idx] for idx in selected_indices if idx in current_by_agent]
    if not selected_rows:
        return []

    x_selected = np.array([[float(row[c]) for c in feature_columns] for row in selected_rows], dtype=float)

    normalized_mode = str(selection_mode).strip().lower()
    if normalized_mode == "medoid":
        x_explain = _compute_medoid_matrix(x_selected)
    elif normalized_mode == "selected_agents":
        x_explain = x_selected
    else:
        raise ValueError("selection_mode must be 'selected_agents' or 'medoid'")

    selected_pool_size = int(x_selected.shape[0])
    explained_agents = int(x_explain.shape[0])

    all_rows: list[dict] = []

    # Classification target: improved (1 if delta_f < 0)
    y_class = np.array([int(row["improved"]) for row in all_samples], dtype=int)
    if np.unique(y_class).size > 1:
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
        clf.fit(x_train, y_class)

        explainer_cls = LimeTabularExplainer(
            x_train,
            feature_names=feature_columns,
            class_names=["not_improved", "improved"],
            mode="classification",
            random_state=random_state,
            discretize_continuous=True,
        )

        rows_per_agent_cls: list[np.ndarray] = []
        for row_x in x_explain:
            explanation = explainer_cls.explain_instance(
                row_x,
                clf.predict_proba,
                num_features=len(feature_columns),
                labels=[1],
            )
            sparse_pairs = explanation.local_exp.get(1, [])
            rows_per_agent_cls.append(_weights_to_dense(len(feature_columns), sparse_pairs))

        all_rows.extend(
            _build_lime_rows_for_target(
                target_type="classification_improved",
                rows_per_agent=rows_per_agent_cls,
                feature_columns=feature_columns,
                run_metadata=run_metadata,
                diagnosis_id=diagnosis_id,
                iteration=iteration,
                n_agents=explained_agents,
                selection_mode=normalized_mode,
                selected_pool_size=selected_pool_size,
            )
        )

    # Regression target: y = -delta_f
    y_reg = np.array([float(row["y_reg"]) for row in all_samples], dtype=float)
    if np.unique(y_reg).size > 1:
        reg = RandomForestRegressor(n_estimators=200, random_state=random_state)
        reg.fit(x_train, y_reg)

        explainer_reg = LimeTabularExplainer(
            x_train,
            feature_names=feature_columns,
            mode="regression",
            random_state=random_state,
            discretize_continuous=True,
        )

        rows_per_agent_reg: list[np.ndarray] = []
        for row_x in x_explain:
            explanation = explainer_reg.explain_instance(
                row_x,
                reg.predict,
                num_features=len(feature_columns),
            )
            if explanation.local_exp:
                first_key = sorted(explanation.local_exp.keys())[0]
                sparse_pairs = explanation.local_exp[first_key]
            else:
                sparse_pairs = []
            rows_per_agent_reg.append(_weights_to_dense(len(feature_columns), sparse_pairs))

        all_rows.extend(
            _build_lime_rows_for_target(
                target_type="regression_y_reg",
                rows_per_agent=rows_per_agent_reg,
                feature_columns=feature_columns,
                run_metadata=run_metadata,
                diagnosis_id=diagnosis_id,
                iteration=iteration,
                n_agents=explained_agents,
                selection_mode=normalized_mode,
                selected_pool_size=selected_pool_size,
            )
        )

    return all_rows
