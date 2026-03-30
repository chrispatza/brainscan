import pandas as pd

from config import ACTIVE_FILTER_METRICS_CSV, FILTER_METRICS_CSV, REMOVED_FILTERS_CSV


def main() -> None:
    df = pd.read_csv(FILTER_METRICS_CSV)

    mean_activation = (
        df.groupby(["layer", "filter"], as_index=False)["avg_activation"]
        .mean()
        .rename(columns={"avg_activation": "mean_avg_activation"})
    )

    active_filters = mean_activation[mean_activation["mean_avg_activation"] > 0][["layer", "filter"]]
    removed_filters = mean_activation[mean_activation["mean_avg_activation"] == 0][["layer", "filter"]]

    df_active = df.merge(active_filters, on=["layer", "filter"], how="inner")

    ACTIVE_FILTER_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_active.to_csv(ACTIVE_FILTER_METRICS_CSV, index=False)
    removed_filters.to_csv(REMOVED_FILTERS_CSV, index=False)

    print(f"Saved active metrics to: {ACTIVE_FILTER_METRICS_CSV}")
    print(f"Saved removed filters to: {REMOVED_FILTERS_CSV}")
    print(f"Original filters: {len(mean_activation)}")
    print(f"Removed filters: {len(removed_filters)}")
    print(f"Kept filters: {len(active_filters)}")


if __name__ == "__main__":
    main()
