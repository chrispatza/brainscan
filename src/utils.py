from pathlib import Path

import numpy as np
from scipy.stats import entropy
from tensorflow.keras.preprocessing import image


def morans_i(activation_map: np.ndarray) -> float:
    x = activation_map.flatten()
    n = len(x)
    mean_x = np.mean(x)

    h, w = activation_map.shape
    weight_sum = 0
    numerator = 0.0

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * w + j)
            if i < h - 1:
                neighbors.append((i + 1) * w + j)
            if j > 0:
                neighbors.append(i * w + (j - 1))
            if j < w - 1:
                neighbors.append(i * w + (j + 1))

            for neighbor_idx in neighbors:
                numerator += (x[idx] - mean_x) * (x[neighbor_idx] - mean_x)
                weight_sum += 1

    denominator = np.sum((x - mean_x) ** 2)
    if denominator == 0 or weight_sum == 0:
        return 0.0

    return float((n / weight_sum) * (numerator / denominator))


def compute_activation_metrics(activation_map: np.ndarray) -> dict:
    flat = activation_map.flatten()
    non_zero = flat[flat > 0]

    hist, _ = np.histogram(flat, bins=30, density=True)
    hist = hist[hist > 0]

    return {
        "avg_activation": float(np.mean(flat)),
        "avg_nonzero": float(np.mean(non_zero)) if non_zero.size else 0.0,
        "median_nonzero": float(np.median(non_zero)) if non_zero.size else 0.0,
        "num_nonzero": int(non_zero.size),
        "variance": float(np.var(flat)),
        "variance_nonzero": float(np.var(non_zero)) if non_zero.size else 0.0,
        "max_activation": float(np.max(flat)),
        "entropy": float(entropy(hist)) if hist.size else 0.0,
        "morans_I": float(morans_i(activation_map)),
    }


def load_and_preprocess_image(img_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def image_stem(filename: str) -> str:
    return Path(str(filename)).stem
