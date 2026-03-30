import pandas as pd

from config import KNN_SUMMARY_CSV, REMOVED_FILTERS_CSV


THRESHOLD = 0.60
REASON_TEXT = "low_test_accuracy"


def main() -> None:
    removed_df = pd.read_csv(REMOVED_FILTERS_CSV)
    summary_df = pd.read_csv(KNN_SUMMARY_CSV)

    removed_df["layer"] = removed_df["layer"].astype(str).str.strip()
    summary_df["layer"] = summary_df["layer"].astype(str).str.strip()

    removed_df["filter"] = pd.to_numeric(removed_df["filter"], errors="coerce").astype("Int64")
    summary_df["filter"] = pd.to_numeric(summary_df["filter"], errors="coerce").astype("Int64")

    low_perf = summary_df.loc[
        summary_df["test_accuracy"] < THRESHOLD,
        ["layer", "filter"],
    ].copy()
    low_perf["reason"] = REASON_TEXT

    if "reason" not in removed_df.columns:
        removed_df["reason"] = "zero_mean_activation"

    combined = pd.concat([removed_df, low_perf], ignore_index=True)
    combined = combined.drop_duplicates(subset=["layer", "filter"], keep="first")
    combined = combined.sort_values(["layer", "filter"]).reset_index(drop=True)

    combined.to_csv(REMOVED_FILTERS_CSV, index=False)
    print(f"Updated removed filter list: {REMOVED_FILTERS_CSV}")
    print(f"Total removed filters: {len(combined)}")


if __name__ == "__main__":
    main()
