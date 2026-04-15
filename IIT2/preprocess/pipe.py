import glob
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Mumbai bounding box used across the original scripts.
MIN_LAT = 18.8900
MAX_LAT = 19.3000
MIN_LON = 72.7500
MAX_LON = 73.0000


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_raw_csvs(raw_input_folder: str, formatted_output_folder: str) -> int:
    """
    Step 1 (from pre.py): read tab-separated raw CSV files and write comma CSV files.
    """
    ensure_dir(formatted_output_folder)

    csv_files = sorted(glob.glob(os.path.join(raw_input_folder, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in '{raw_input_folder}'.")
        return 0

    print(f"Found {len(csv_files)} raw files. Formatting to comma-separated CSV...")

    success_count = 0
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(formatted_output_folder, f"formatted_{filename}")

        try:
            df = pd.read_csv(file_path, sep="\t")
            df.to_csv(output_path, index=False, sep=",")
            success_count += 1
            print(f"  [SUCCESS] formatted -> {filename}")
        except Exception as exc:
            print(f"  [ERROR] failed formatting '{filename}': {exc}")

    print(f"Formatting complete. {success_count}/{len(csv_files)} files processed successfully.")
    return success_count


def filter_and_add_ist(formatted_input_folder: str, time_output_folder: str) -> Tuple[int, int]:
    """
    Step 2 (combined mumbai.py + time.py):
    - filter points to Mumbai bounding box
    - convert unix timestamp to IST date/time columns
    - save per-file Mumbai+IST output
    """
    ensure_dir(time_output_folder)

    csv_files = sorted(glob.glob(os.path.join(formatted_input_folder, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in '{formatted_input_folder}'.")
        return 0, 0

    print(f"Found {len(csv_files)} formatted files. Filtering Mumbai + converting IST...")

    total_scanned = 0
    total_saved = 0

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(time_output_folder, f"mumbai_ist_{filename}")

        try:
            df = pd.read_csv(file_path)
            total_scanned += len(df)

            mumbai_df = df[
                (df["latitude"] >= MIN_LAT)
                & (df["latitude"] <= MAX_LAT)
                & (df["longitude"] >= MIN_LON)
                & (df["longitude"] <= MAX_LON)
            ].copy()

            if mumbai_df.empty:
                print(f"  [SKIP] no Mumbai points in {filename}")
                continue

            mumbai_df["datetime_utc"] = pd.to_datetime(
                mumbai_df["timestamp"], unit="s", errors="coerce"
            )
            mumbai_df["datetime_ist"] = mumbai_df["datetime_utc"] + pd.Timedelta(
                hours=5, minutes=30
            )
            mumbai_df["date_ist"] = mumbai_df["datetime_ist"].dt.strftime("%Y-%m-%d")
            mumbai_df["time_ist"] = mumbai_df["datetime_ist"].dt.strftime("%I:%M:%S %p")

            mumbai_df = mumbai_df.drop(columns=["datetime_utc", "datetime_ist"])
            mumbai_df.to_csv(output_path, index=False)

            total_saved += len(mumbai_df)
            print(f"  [SUCCESS] saved {len(mumbai_df):,} rows -> {output_path}")
        except Exception as exc:
            print(f"  [ERROR] failed processing '{filename}': {exc}")

    print("Mumbai + IST step complete.")
    print(f"  Total rows scanned: {total_scanned:,}")
    print(f"  Total rows saved:   {total_saved:,}")
    return total_scanned, total_saved


def build_active_devices_output(
    time_input_folder: str,
    final_output_folder: str,
    min_points_threshold: int,
    plot_name: str = "device_ping_distribution.png",
) -> str:
    """
    Step 3 (from count.py): combine IST CSVs and keep only highly active devices.
    """
    ensure_dir(final_output_folder)

    csv_files = sorted(glob.glob(os.path.join(time_input_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{time_input_folder}'.")

    print(f"Loading {len(csv_files)} IST files for device activity analysis...")

    all_data = []
    for path in csv_files:
        all_data.append(pd.read_csv(path))

    master_df = pd.concat(all_data, ignore_index=True)
    if master_df.empty:
        raise ValueError("No rows available after loading time-step output.")

    if "device_aid" not in master_df.columns:
        raise KeyError("Required column 'device_aid' was not found.")
    if "timestamp" not in master_df.columns:
        raise KeyError("Required column 'timestamp' was not found.")

    device_counts = master_df["device_aid"].value_counts()

    print("\n" + "=" * 40)
    print("DEVICE ACTIVITY STATISTICS")
    print("=" * 40)
    print(f"Total Unique Devices: {len(device_counts):,}")
    print(f"Average points per device: {device_counts.mean():.2f}")
    print(f"Median points per device:  {device_counts.median():.0f}")
    print(f"Max points for one device: {device_counts.max():,}")
    print("=" * 40)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    capped_data = device_counts[device_counts <= 500]
    sns.histplot(capped_data, bins=50, ax=axes[0], color="blue", kde=False)
    axes[0].set_title("Distribution of Pings per Device (0 - 500 pings)")
    axes[0].set_xlabel("Number of Pings")
    axes[0].set_ylabel("Number of Devices")

    sns.ecdfplot(data=device_counts, ax=axes[1], color="red")
    axes[1].set_title("Cumulative Percentage of Devices")
    axes[1].set_xlabel("Number of Pings")
    axes[1].set_ylabel("Percentage of Devices")
    axes[1].set_xlim(0, 100)
    axes[1].axvline(
        min_points_threshold,
        color="black",
        linestyle="--",
        label=f"Threshold: {min_points_threshold} pings",
    )
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(final_output_folder, plot_name)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved graph -> {plot_path}")

    valid_devices = device_counts[device_counts >= min_points_threshold].index
    filtered_df = master_df[master_df["device_aid"].isin(valid_devices)].copy()
    filtered_df = filtered_df.sort_values(["device_aid", "timestamp"])

    output_file = os.path.join(final_output_folder, "mumbai_highly_active_devices.csv")
    filtered_df.to_csv(output_file, index=False)

    print(
        f"Saved final dataset: {len(filtered_df):,} rows, {len(valid_devices):,} devices -> {output_file}"
    )
    return output_file


def run_pipeline(
    raw_input_folder: str,
    formatted_output_folder: str,
    time_output_folder: str,
    final_output_folder: str,
    min_points_threshold: int,
) -> None:
    print("\n" + "=" * 60)
    print("STARTING COMBINED PIPELINE")
    print("=" * 60)

    format_raw_csvs(raw_input_folder, formatted_output_folder)
    filter_and_add_ist(formatted_output_folder, time_output_folder)
    build_active_devices_output(time_output_folder, final_output_folder, min_points_threshold)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Edit these paths directly as needed.
    RAW_INPUT_FOLDER = "E:/gps_data/01"
    FORMATTED_OUTPUT_FOLDER = "E:/clean_data/01_clean"
    TIME_OUTPUT_FOLDER = "../mumbai_time/01_time"
    FINAL_OUTPUT_FOLDER = "../mumbai_active_data/01_active"
    MINIMUM_PINGS_REQUIRED = 20

    run_pipeline(
        raw_input_folder=RAW_INPUT_FOLDER,
        formatted_output_folder=FORMATTED_OUTPUT_FOLDER,
        time_output_folder=TIME_OUTPUT_FOLDER,
        final_output_folder=FINAL_OUTPUT_FOLDER,
        min_points_threshold=MINIMUM_PINGS_REQUIRED,
    )