from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, load_model

from config import (
    DASHBOARD_DIR,
    FEATURE_COLUMNS,
    IMAGE_SIZE,
    KNN_MODELS_DIR,
    KNN_SUMMARY_CSV,
    MODEL_PATH,
    REMOVED_FILTERS_CSV,
    TRANSFORMATIONS_DIR,
    UPDATED_DASHBOARD_DIR,
)
from utils import compute_activation_metrics, image_stem, load_and_preprocess_image


OVERWRITE_EXISTING_CSVS = False
RESTRICT_TO_LAYERS_USED_BY_KNN = True


def extract_features_for_folder(
    folder_path: Path,
    conv_layers: list,
    feature_extractor: Model,
    removed_keys: set[tuple[str, int]],
    active_filter_keys: set[tuple[str, int]],
    knn_models: dict[tuple[str, int], object],
    target_size: tuple[int, int],
) -> pd.DataFrame:
    records = []
    image_files = sorted(
        [p for p in folder_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

    for idx, img_path in enumerate(image_files, start=1):
        arr = load_and_preprocess_image(img_path, target_size=target_size)
        feature_maps = feature_extractor.predict(arr, verbose=0)

        for layer_obj, fmap in zip(conv_layers, feature_maps):
            layer_name = layer_obj.name
            fmap = fmap[0]

            for filter_idx in range(fmap.shape[-1]):
                key = (layer_name, int(filter_idx))
                if key in removed_keys or key not in active_filter_keys or key not in knn_models:
                    continue

                records.append(
                    {
                        "image": img_path.name,
                        "image_stem": image_stem(img_path.name),
                        "layer": layer_name,
                        "filter": int(filter_idx),
                        **compute_activation_metrics(fmap[:, :, filter_idx]),
                    }
                )

        if idx % 10 == 0 or idx == len(image_files):
            print(f"  processed {idx}/{len(image_files)} images")

    return pd.DataFrame(records)


def apply_filter_knns(features_df: pd.DataFrame, knn_models: dict[tuple[str, int], object]) -> pd.DataFrame:
    prediction_rows = []

    for (layer_name, filter_idx), sub in features_df.groupby(["layer", "filter"], sort=False):
        key = (layer_name, int(filter_idx))
        if key not in knn_models:
            continue

        model = knn_models[key]
        X = sub[FEATURE_COLUMNS].values
        pred_num = model.predict(X)
        probas = model.predict_proba(X)

        out = sub[["image", "image_stem", "layer", "filter"]].copy()
        out["prediction"] = np.where(pred_num == 0, "cat", "dog")
        out["confidence"] = probas.max(axis=1)
        prediction_rows.append(out)

    if not prediction_rows:
        return pd.DataFrame(columns=["image", "image_stem", "layer", "filter", "prediction", "confidence"])

    return pd.concat(prediction_rows, ignore_index=True)


def update_dashboard_csv(csv_path: Path, pred_df: pd.DataFrame, output_csv_path: Path) -> None:
    dash_df = pd.read_csv(csv_path)

    required_cols = {"image", "layer", "filter"}
    missing = required_cols - set(dash_df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    dash_df["filter"] = dash_df["filter"].astype(int)
    dash_df["image_stem"] = dash_df["image"].apply(image_stem)

    merged = dash_df.merge(
        pred_df[["image_stem", "layer", "filter", "prediction", "confidence"]],
        on=["image_stem", "layer", "filter"],
        how="left",
        suffixes=("", "_new"),
    )

    if "prediction" in merged.columns and "prediction_new" in merged.columns:
        merged["prediction"] = merged["prediction_new"].combine_first(merged["prediction"])
    elif "prediction_new" in merged.columns:
        merged["prediction"] = merged["prediction_new"]

    if "confidence" in merged.columns and "confidence_new" in merged.columns:
        merged["confidence"] = merged["confidence_new"].combine_first(merged["confidence"])
    elif "confidence_new" in merged.columns:
        merged["confidence"] = merged["confidence_new"]

    merged = merged.drop(columns=[c for c in ["prediction_new", "confidence_new", "image_stem"] if c in merged.columns])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv_path, index=False)
    print(f"Saved updated CSV: {output_csv_path}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing CNN model: {MODEL_PATH}")

    cnn_model = load_model(MODEL_PATH)

    removed_df = pd.read_csv(REMOVED_FILTERS_CSV)
    removed_df["filter"] = removed_df["filter"].astype(int)
    removed_keys = set(zip(removed_df["layer"], removed_df["filter"]))

    knn_summary = pd.read_csv(KNN_SUMMARY_CSV)
    knn_summary["filter"] = knn_summary["filter"].astype(int)

    knn_layers = knn_summary["layer"].unique().tolist() if RESTRICT_TO_LAYERS_USED_BY_KNN else None
    active_knn_filters = knn_summary[["layer", "filter", "best_k"]].drop_duplicates().copy()
    active_knn_filters = active_knn_filters[
        ~active_knn_filters.apply(lambda r: (r["layer"], int(r["filter"])) in removed_keys, axis=1)
    ].reset_index(drop=True)

    active_filter_keys = set(zip(active_knn_filters["layer"], active_knn_filters["filter"]))

    all_conv_layers = [layer for layer in cnn_model.layers if isinstance(layer, Conv2D)]
    conv_layers = [layer for layer in all_conv_layers if layer.name in knn_layers] if knn_layers else all_conv_layers
    if not conv_layers:
        raise RuntimeError("No Conv2D layers matched the KNN summary layer names.")

    feature_extractor = Model(cnn_model.inputs, outputs=[layer.output for layer in conv_layers])

    img_height = cnn_model.input_shape[1] or IMAGE_SIZE[0]
    img_width = cnn_model.input_shape[2] or IMAGE_SIZE[1]

    knn_models = {}
    for _, row in active_knn_filters.iterrows():
        key = (row["layer"], int(row["filter"]))
        model_path = KNN_MODELS_DIR / f"knn_distance_{key[0]}_filter_{key[1]}.joblib"
        if model_path.exists():
            knn_models[key] = joblib.load(model_path)

    UPDATED_DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    for t in range(1, 13):
        folder_name = f"T{t}"
        folder_path = TRANSFORMATIONS_DIR / folder_name
        dashboard_csv = DASHBOARD_DIR / f"{folder_name}_knn_predictions.csv"

        if not folder_path.is_dir():
            print(f"Skipping missing folder: {folder_path}")
            continue
        if not dashboard_csv.exists():
            print(f"Skipping missing dashboard CSV: {dashboard_csv}")
            continue

        print(f"Working on {folder_name}")
        features_df = extract_features_for_folder(
            folder_path=folder_path,
            conv_layers=conv_layers,
            feature_extractor=feature_extractor,
            removed_keys=removed_keys,
            active_filter_keys=active_filter_keys,
            knn_models=knn_models,
            target_size=(img_height, img_width),
        )

        features_df.to_csv(UPDATED_DASHBOARD_DIR / f"{folder_name}_extracted_features.csv", index=False)

        pred_df = apply_filter_knns(features_df, knn_models)
        pred_df.to_csv(UPDATED_DASHBOARD_DIR / f"{folder_name}_filter_predictions_only.csv", index=False)

        output_csv = dashboard_csv if OVERWRITE_EXISTING_CSVS else UPDATED_DASHBOARD_DIR / f"{folder_name}_knn_predictions_updated.csv"
        update_dashboard_csv(dashboard_csv, pred_df, output_csv)

    print("All done.")


if __name__ == "__main__":
    main()
