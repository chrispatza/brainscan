import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS,
    FILTER_METRICS_CSV,
    KNN_MODELS_DIR,
    KNN_RESULTS_DIR,
    KNN_SUMMARY_CSV,
    KNN_TEST_PREDS_DIR,
    RANDOM_STATE,
    REMOVED_FILTERS_CSV,
)


K_VALUES = list(range(10, 201))
N_TRAIN_PER_CLASS = 450
N_TEST_PER_CLASS = 50
SAVE_MODELS = True
SAVE_TEST_PREDICTIONS = True


def split_images_per_class(
    df_in: pd.DataFrame,
    cls_name: str,
    n_train: int,
    n_test: int,
    random_state: int,
) -> tuple[set[str], set[str]]:
    images = sorted(df_in.loc[df_in["class"] == cls_name, "image"].unique())
    required = n_train + n_test
    if len(images) < required:
        raise ValueError(
            f"Class '{cls_name}' has only {len(images)} unique images, but {required} are required."
        )

    rng = np.random.RandomState(random_state)
    shuffled = np.array(images)
    rng.shuffle(shuffled)

    train_imgs = shuffled[:n_train]
    test_imgs = shuffled[n_train : n_train + n_test]
    return set(train_imgs), set(test_imgs)


def main() -> None:
    KNN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    KNN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    KNN_TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FILTER_METRICS_CSV)
    removed_df = pd.read_csv(REMOVED_FILTERS_CSV)

    required_cols = {"image", "class", "layer", "filter", *FEATURE_COLUMNS}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in metrics CSV: {missing_cols}")

    removed_required = {"layer", "filter"}
    missing_removed_cols = removed_required - set(removed_df.columns)
    if missing_removed_cols:
        raise ValueError(f"Missing columns in removed_filters CSV: {missing_removed_cols}")

    df = df[df["class"].isin(["cat", "dog"])].copy()
    df["filter"] = df["filter"].astype(int)
    removed_df["filter"] = removed_df["filter"].astype(int)

    removed_keys = set(zip(removed_df["layer"], removed_df["filter"]))
    df["filter_key"] = list(zip(df["layer"], df["filter"]))
    df_active = df[~df["filter_key"].isin(removed_keys)].copy()
    df_active.drop(columns=["filter_key"], inplace=True)

    label_map = {"cat": 0, "dog": 1}
    df_active["y"] = df_active["class"].map(label_map)
    df_active[FEATURE_COLUMNS] = df_active[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    df_active = df_active.dropna(subset=FEATURE_COLUMNS)

    cat_train_imgs, cat_test_imgs = split_images_per_class(
        df_active, "cat", N_TRAIN_PER_CLASS, N_TEST_PER_CLASS, RANDOM_STATE
    )
    dog_train_imgs, dog_test_imgs = split_images_per_class(
        df_active, "dog", N_TRAIN_PER_CLASS, N_TEST_PER_CLASS, RANDOM_STATE
    )

    train_images = cat_train_imgs.union(dog_train_imgs)
    test_images = cat_test_imgs.union(dog_test_imgs)

    if train_images.intersection(test_images):
        raise RuntimeError("Train/test overlap detected.")

    df_train = df_active[df_active["image"].isin(train_images)].copy()
    df_test = df_active[df_active["image"].isin(test_images)].copy()

    active_filters = (
        df_active[["layer", "filter"]]
        .drop_duplicates()
        .sort_values(["layer", "filter"])
        .reset_index(drop=True)
    )

    results = []

    for idx, row in active_filters.iterrows():
        layer_name = row["layer"]
        filter_idx = int(row["filter"])
        print(f"[{idx + 1}/{len(active_filters)}] layer={layer_name}, filter={filter_idx}")

        train_sub = df_train[(df_train["layer"] == layer_name) & (df_train["filter"] == filter_idx)].copy()
        test_sub = df_test[(df_test["layer"] == layer_name) & (df_test["filter"] == filter_idx)].copy()

        if train_sub.empty or test_sub.empty:
            continue

        X_train = train_sub[FEATURE_COLUMNS].values
        y_train = train_sub["y"].values
        X_test = test_sub[FEATURE_COLUMNS].values
        y_test = test_sub["y"].values

        valid_k_values = [k for k in K_VALUES if k <= len(X_train)]
        if not valid_k_values:
            continue

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(weights="distance")),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid={"knn__n_neighbors": valid_k_values},
            scoring="accuracy",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1,
            refit=True,
            verbose=0,
            return_train_score=True,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        probas = best_model.predict_proba(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        cat_mask = y_test == 0
        dog_mask = y_test == 1

        results.append(
            {
                "layer": layer_name,
                "filter": filter_idx,
                "n_train": len(train_sub),
                "n_test": len(test_sub),
                "best_k": grid.best_params_["knn__n_neighbors"],
                "best_cv_accuracy": grid.best_score_,
                "test_accuracy": test_acc,
                "cat_test_accuracy": accuracy_score(y_test[cat_mask], y_pred[cat_mask]) if np.sum(cat_mask) else np.nan,
                "dog_test_accuracy": accuracy_score(y_test[dog_mask], y_pred[dog_mask]) if np.sum(dog_mask) else np.nan,
                "tn_cat_as_cat": tn,
                "fp_cat_as_dog": fp,
                "fn_dog_as_cat": fn,
                "tp_dog_as_dog": tp,
            }
        )

        if SAVE_MODELS:
            model_name = f"knn_distance_{layer_name}_filter_{filter_idx}.joblib"
            joblib.dump(best_model, KNN_MODELS_DIR / model_name)

        if SAVE_TEST_PREDICTIONS:
            pred_df = test_sub[["image", "class", "layer", "filter"]].copy()
            pred_df["y_true"] = y_test
            pred_df["y_pred"] = y_pred
            pred_df["pred_class"] = np.where(y_pred == 0, "cat", "dog")
            pred_df["prob_cat"] = probas[:, 0]
            pred_df["prob_dog"] = probas[:, 1]
            pred_df.to_csv(
                KNN_TEST_PREDS_DIR / f"preds_{layer_name}_filter_{filter_idx}.csv",
                index=False,
            )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No filter models were trained. Check your input files.")

    results_df = results_df.sort_values(
        ["test_accuracy", "best_cv_accuracy", "layer", "filter"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    results_df.to_csv(KNN_SUMMARY_CSV, index=False)
    print(f"Saved summary to: {KNN_SUMMARY_CSV}")
    print(results_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
