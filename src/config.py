from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

TRANSFORMATIONS_DIR = PROJECT_ROOT / "Transformations"
DASHBOARD_DIR = PROJECT_ROOT / "Morphing_Dashboard"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
KNN_RESULTS_DIR = RESULTS_DIR / "knn_per_filter_results_distance_weighted"
KNN_MODELS_DIR = KNN_RESULTS_DIR / "saved_models"
KNN_TEST_PREDS_DIR = KNN_RESULTS_DIR / "test_predictions"
UPDATED_DASHBOARD_DIR = RESULTS_DIR / "updated_predictions"

MODEL_PATH = MODELS_DIR / "cat_dog_model_128.h5"
FILTER_METRICS_CSV = RESULTS_DIR / "filter_metrics_full.csv"
ACTIVE_FILTER_METRICS_CSV = RESULTS_DIR / "filter_metrics_active.csv"
REMOVED_FILTERS_CSV = RESULTS_DIR / "removed_filters.csv"
KNN_SUMMARY_CSV = KNN_RESULTS_DIR / "knn_filter_summary_distance_weighted.csv"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "avg_activation",
    "avg_nonzero",
    "median_nonzero",
    "num_nonzero",
    "variance",
    "variance_nonzero",
    "max_activation",
    "entropy",
    "morans_I",
]
