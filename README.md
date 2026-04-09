# Cat–Dog CNN Filter Analysis

Cleaned research repository reconstructed from the notebook you provided.

## Repository layout

```text
.
├── Data/
│   ├── train/{cat,dog}
│   ├── val/{cat,dog}
│   └── test/{cat,dog}
├── Transformations/
│   ├── T1/ ... T12/
├── Morphing_Dashboard/
├── models/
├── results/
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── train_cnn.py
│   ├── extract_filter_metrics.py
│   ├── remove_dead_filters.py
│   ├── extend_removed_filters.py
│   ├── train_weighted_knn_per_filter.py
│   └── update_dashboard_predictions.py
```

## What each script does

- `train_cnn.py`  
  Train the 4-layer cat/dog CNN and optionally evaluate it on the test split.

- `extract_filter_metrics.py`  
  Extract per-filter activation statistics for every image in a directory tree.

- `remove_dead_filters.py`  
  Remove filters whose mean activation is exactly zero across the dataset.

- `extend_removed_filters.py`  
  Add low-performing filters to the removed filter list based on KNN test accuracy.

- `train_weighted_knn_per_filter.py`  
  Train one distance-weighted KNN per active filter using the 9 spatial metrics.

- `update_dashboard_predictions.py`  
  Apply the saved KNN models to `Transformations/T1` ... `T12` and update dashboard CSVs.

## Suggested workflow

```bash
python src/train_cnn.py
python src/extract_filter_metrics.py
python src/remove_dead_filters.py
python src/train_weighted_knn_per_filter.py
python src/extend_removed_filters.py
python src/update_dashboard_predictions.py
```

## Notes

- Put your real cat/dog image data into `Data/`.
- Put your transformation frames into `Transformations/T1` ... `Transformations/T12`.
- Put your dashboard CSVs into `Morphing_Dashboard/` using names like `T1_knn_predictions.csv`.
- The code now uses project-relative paths instead of machine-specific absolute paths.
