from pathlib import Path

import pandas as pd
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image

from config import FILTER_METRICS_CSV, IMAGE_SIZE, MODEL_PATH, TEST_DIR, TRAIN_DIR, VAL_DIR
from utils import compute_activation_metrics


def collect_image_paths() -> list[tuple[Path, str, str]]:
    records = []
    split_dirs = [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]

    for split_name, split_dir in split_dirs:
        for class_name in ["cat", "dog"]:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    records.append((img_path, class_name, split_name))
    return records


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]
    feature_extractor = Model(model.inputs, outputs=[layer.output for layer in conv_layers])

    img_height = model.input_shape[1] or IMAGE_SIZE[0]
    img_width = model.input_shape[2] or IMAGE_SIZE[1]

    records = []
    image_records = collect_image_paths()

    for idx, (img_path, class_name, split_name) in enumerate(image_records, start=1):
        arr = image.load_img(img_path, target_size=(img_height, img_width))
        arr = image.img_to_array(arr) / 255.0
        arr = arr[None, ...]

        feature_maps = feature_extractor.predict(arr, verbose=0)

        for layer_obj, fmap in zip(conv_layers, feature_maps):
            layer_name = layer_obj.name
            fmap = fmap[0]

            for filter_idx in range(fmap.shape[-1]):
                metrics = compute_activation_metrics(fmap[:, :, filter_idx])
                records.append(
                    {
                        "image": img_path.name,
                        "class": class_name,
                        "split": split_name,
                        "layer": layer_name,
                        "filter": filter_idx,
                        **metrics,
                    }
                )

        if idx % 25 == 0 or idx == len(image_records):
            print(f"Processed {idx}/{len(image_records)} images")

    df = pd.DataFrame(records)
    FILTER_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILTER_METRICS_CSV, index=False)
    print(f"Saved metrics to: {FILTER_METRICS_CSV}")


if __name__ == "__main__":
    main()
