from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("target_function_workload_d01_d14.csv")
DEFAULT_OUTPUT_DIR = Path("burst_datasets")
DEFAULT_TOTAL_POINTS = 8640
DEFAULT_RANDOM_SEED = 42

REQUEST_COL = "request_count"

REQUESTS_PER_POD_PER_20S = 80
POD_COUNT = 4
CLUSTER_CAPACITY_PER_20S = REQUESTS_PER_POD_PER_20S * POD_COUNT

NEAR_CAPACITY_MIN_UTILIZATION = 0.73
NEAR_CAPACITY_MAX_UTILIZATION = 1
HIGH_LOAD_MIN_REQUESTS = int(CLUSTER_CAPACITY_PER_20S * NEAR_CAPACITY_MIN_UTILIZATION)
HIGH_LOAD_MAX_REQUESTS = int(CLUSTER_CAPACITY_PER_20S * NEAR_CAPACITY_MAX_UTILIZATION)

SUDDEN_SPIKE_PEAK_RANGE = (
    int(CLUSTER_CAPACITY_PER_20S * 0.70),
    HIGH_LOAD_MAX_REQUESTS,
)
PERIODIC_PEAK_RANGE = (
    int(CLUSTER_CAPACITY_PER_20S * 0.65),
    HIGH_LOAD_MAX_REQUESTS,
)
PERIODIC_BASELINE_RANGE = (
    int(CLUSTER_CAPACITY_PER_20S * 0.20),
    int(CLUSTER_CAPACITY_PER_20S * 0.45),
)
IRREGULAR_BURST_HIGH_RANGE = (
    int(CLUSTER_CAPACITY_PER_20S * 0.50),
    HIGH_LOAD_MAX_REQUESTS,
)
IRREGULAR_BURST_MID_RANGE = (
    int(CLUSTER_CAPACITY_PER_20S * 0.25),
    int(CLUSTER_CAPACITY_PER_20S * 0.60),
)
PATTERN_SHIFT_LEVEL_RANGE = (
    int(CLUSTER_CAPACITY_PER_20S * 0.50),
    int(CLUSTER_CAPACITY_PER_20S * 0.75),
)
PATTERN_SHIFT_CLIP_RANGE = (
    int(CLUSTER_CAPACITY_PER_20S * 0.35),
    HIGH_LOAD_MAX_REQUESTS,
)
PATTERN_SHIFT_RAMP_RATIO_RANGE = (0.15, 0.25)

BURST_COUNTS_BY_DATASET = {
    "train": {
        "Sudden spike": 45,
        "Periodic peaks": 45,
        "Irregular bursts": 45,
        "Pattern shift": 42,
    },
    "val": {
        "Sudden spike": 22,
        "Periodic peaks": 22,
        "Irregular bursts": 22,
        "Pattern shift": 20,
    },
    "test": {
        "Sudden spike": 22,
        "Periodic peaks": 22,
        "Irregular bursts": 22,
        "Pattern shift": 20,
    },
}

BURST_DURATION_RANGE = {
    "Sudden spike": (4, 8),
    "Periodic peaks": (9, 18),
    "Irregular bursts": (8, 16),
    "Pattern shift": (18, 36),
}

MIN_NORMAL_GAP_WINDOWS = 3


def split_dataset(df: pd.DataFrame, total_points: int = DEFAULT_TOTAL_POINTS) -> dict[str, pd.DataFrame]:
    if len(df) < total_points:
        raise ValueError(f"Expected at least {total_points} rows, found {len(df)}")

    selected = df.iloc[:total_points].copy().reset_index(drop=True)
    train_end = int(total_points * 0.5)
    val_end = int(total_points * 0.75)

    return {
        "train": selected.iloc[:train_end].copy().reset_index(drop=True),
        "val": selected.iloc[train_end:val_end].copy().reset_index(drop=True),
        "test": selected.iloc[val_end:].copy().reset_index(drop=True),
    }


def divide_request_counts(df: pd.DataFrame, divide_by: float) -> pd.DataFrame:
    if divide_by <= 0:
        raise ValueError(f"divide_by must be positive, got {divide_by}")
    if REQUEST_COL not in df.columns:
        raise ValueError(f"Missing required column: {REQUEST_COL}")

    converted = df.copy()
    converted[REQUEST_COL] = converted[REQUEST_COL] / divide_by
    return converted


def scale_datasets_to_capacity(
    datasets: dict[str, pd.DataFrame],
    capacity_limit: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    if capacity_limit <= 0:
        raise ValueError(f"capacity_limit must be positive, got {capacity_limit}")

    pre_scale_max = max(float(df[REQUEST_COL].max()) for df in datasets.values())
    scale_factor = capacity_limit / pre_scale_max if pre_scale_max > 0 else 1.0

    scaled: dict[str, pd.DataFrame] = {}
    for dataset_name, df in datasets.items():
        scaled_df = df.copy()
        scaled_df[REQUEST_COL] = (
            (scaled_df[REQUEST_COL] * scale_factor)
            .round()
            .clip(lower=0, upper=capacity_limit)
            .astype(int)
        )
        scaled[dataset_name] = scaled_df

    metadata = {
        "pre_scale_max_request_count": pre_scale_max,
        "capacity_limit": float(capacity_limit),
        "scale_factor": scale_factor,
    }
    return scaled, metadata


def _rng(seed: int | None, rng: random.Random | None) -> random.Random:
    if rng is not None:
        return rng
    if seed is not None:
        return random.Random(seed)
    return random.Random()


def apply_sudden_spike(base_values: pd.Series, rng: random.Random) -> pd.Series:
    new_values = base_values.copy()
    length = len(base_values)

    if length <= 1:
        return new_values.round().astype(int)

    peak_pos = rng.randint(length // 3, max(length // 3, length * 2 // 3))
    peak_value = rng.randint(*SUDDEN_SPIKE_PEAK_RANGE)
    width = rng.randint(1, max(1, length // 3))

    for i in range(length):
        base = int(round(base_values.iloc[i]))
        distance = abs(i - peak_pos)

        if distance <= width:
            strength = 1 - distance / (width + 1)
            value = int(round(base + strength * (peak_value - base)))
            new_values.iloc[i] = min(HIGH_LOAD_MAX_REQUESTS, max(base, value))
        else:
            new_values.iloc[i] = base

    return new_values.round().astype(int)


def apply_periodic_peaks(base_values: pd.Series, rng: random.Random) -> pd.Series:
    new_values = base_values.copy()
    length = len(base_values)

    peak_interval = rng.randint(2, 4)
    start_offset = rng.randint(0, peak_interval - 1)

    for i in range(length):
        base = int(round(base_values.iloc[i]))

        if (i - start_offset) % peak_interval == 0:
            new_values.iloc[i] = rng.randint(*PERIODIC_PEAK_RANGE)
        else:
            baseline_value = rng.randint(*PERIODIC_BASELINE_RANGE)
            new_values.iloc[i] = max(base, baseline_value)

    return new_values.round().astype(int)


def apply_irregular_bursts(base_values: pd.Series, rng: random.Random) -> pd.Series:
    new_values = base_values.copy()
    length = len(base_values)

    for i in range(length):
        base = int(round(base_values.iloc[i]))
        burst_probability = rng.random()

        if burst_probability < 0.65:
            new_values.iloc[i] = rng.randint(*IRREGULAR_BURST_HIGH_RANGE)
        elif burst_probability < 0.85:
            new_values.iloc[i] = rng.randint(*IRREGULAR_BURST_MID_RANGE)
        else:
            new_values.iloc[i] = base

    return new_values.round().astype(int)


def apply_pattern_shift(base_values: pd.Series, rng: random.Random) -> pd.Series:
    new_values = base_values.copy()
    length = len(base_values)

    if length <= 1:
        return new_values.round().astype(int)

    base_mean = base_values.mean()
    target_level = rng.randint(*PATTERN_SHIFT_LEVEL_RANGE)
    max_shape_delta = int(CLUSTER_CAPACITY_PER_20S * 0.10)
    jitter_range = int(CLUSTER_CAPACITY_PER_20S * 0.08)
    ramp_len = max(1, int(length * rng.uniform(*PATTERN_SHIFT_RAMP_RATIO_RANGE)))
    ramp_len = min(ramp_len, max(1, length // 3))

    for i in range(length):
        base = int(round(base_values.iloc[i]))
        shape_delta = int(round((base_values.iloc[i] - base_mean) * 0.2))
        shape_delta = max(-max_shape_delta, min(max_shape_delta, shape_delta))
        jitter = rng.randint(-jitter_range, jitter_range)
        plateau_value = target_level + shape_delta + jitter
        plateau_value = min(PATTERN_SHIFT_CLIP_RANGE[1], max(PATTERN_SHIFT_CLIP_RANGE[0], plateau_value))

        if i < ramp_len:
            phase = (i + 1) / (ramp_len + 1)
        elif i >= length - ramp_len:
            phase = (length - i) / (ramp_len + 1)
        else:
            phase = 1.0

        value = int(round(base + phase * (plateau_value - base)))
        new_values.iloc[i] = max(0, min(PATTERN_SHIFT_CLIP_RANGE[1], value))

    return new_values.round().astype(int)


def apply_burst_requests(base_values: pd.Series, burst_type: str, rng: random.Random) -> pd.Series:
    if burst_type == "Sudden spike":
        return apply_sudden_spike(base_values, rng)
    if burst_type == "Periodic peaks":
        return apply_periodic_peaks(base_values, rng)
    if burst_type == "Irregular bursts":
        return apply_irregular_bursts(base_values, rng)
    if burst_type == "Pattern shift":
        return apply_pattern_shift(base_values, rng)
    return base_values.round().astype(int)


def add_random_burst_labels(
    df: pd.DataFrame,
    dataset_name: str,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if dataset_name not in BURST_COUNTS_BY_DATASET:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    if REQUEST_COL not in df.columns:
        raise ValueError(f"Missing required column: {REQUEST_COL}")

    local_rng = _rng(seed, rng)
    df = df.copy().reset_index(drop=True)

    if "workload_type" not in df.columns:
        df["workload_type"] = "normal"
    else:
        df["workload_type"] = df["workload_type"].fillna("normal")

    total_len = len(df)
    burst_counts = BURST_COUNTS_BY_DATASET[dataset_name]
    burst_tasks: list[dict[str, int | str]] = []

    for burst_type, count in burst_counts.items():
        min_duration, max_duration = BURST_DURATION_RANGE[burst_type]
        for _ in range(count):
            burst_tasks.append(
                {
                    "burst_type": burst_type,
                    "duration": local_rng.randint(min_duration, max_duration),
                }
            )

    local_rng.shuffle(burst_tasks)
    burst_tasks.sort(key=lambda task: int(task["duration"]), reverse=True)

    occupied = [False] * total_len
    inserted_records: list[dict[str, Any]] = []

    def has_space_with_gap(start_idx: int, duration: int) -> bool:
        guard_start = max(0, start_idx - MIN_NORMAL_GAP_WINDOWS)
        guard_end = min(total_len, start_idx + duration + MIN_NORMAL_GAP_WINDOWS)
        return not any(occupied[guard_start:guard_end])

    for task in burst_tasks:
        burst_type = str(task["burst_type"])
        duration = int(task["duration"])

        valid_starts = [
            start_idx
            for start_idx in range(0, total_len - duration + 1)
            if has_space_with_gap(start_idx, duration)
        ]

        if not valid_starts:
            print(f"[warning] {dataset_name}: unable to insert {burst_type}, duration={duration}")
            continue

        start_idx = local_rng.choice(valid_starts)
        end_idx = start_idx + duration

        df.loc[start_idx : end_idx - 1, "workload_type"] = burst_type

        base_values = df.loc[start_idx : end_idx - 1, REQUEST_COL]
        new_values = apply_burst_requests(base_values, burst_type, local_rng)
        df.loc[start_idx : end_idx - 1, REQUEST_COL] = new_values.values

        for i in range(start_idx, end_idx):
            occupied[i] = True

        inserted_records.append(
            {
                "dataset": dataset_name,
                "burst_type": burst_type,
                "start_idx": start_idx,
                "end_idx": end_idx - 1,
                "duration": duration,
            }
        )

    return df, inserted_records


def process_workload(
    df: pd.DataFrame,
    seed: int = DEFAULT_RANDOM_SEED,
    divide_by: float = 1.0,
    capacity_limit: float | None = None,
    return_metadata: bool = False,
) -> (
    tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]
    | tuple[dict[str, pd.DataFrame], list[dict[str, Any]], dict[str, float]]
):
    working_df = divide_request_counts(df, divide_by) if divide_by != 1 else df.copy()
    splits = split_dataset(working_df)
    rng = random.Random(seed)
    datasets: dict[str, pd.DataFrame] = {}
    all_records: list[dict[str, Any]] = []

    for dataset_name in ("train", "val", "test"):
        burst_df, records = add_random_burst_labels(splits[dataset_name], dataset_name, rng=rng)
        datasets[dataset_name] = burst_df
        all_records.extend(records)

    metadata = {
        "divide_by": float(divide_by),
        "pre_scale_max_request_count": max(float(split[REQUEST_COL].max()) for split in datasets.values()),
        "capacity_limit": float(capacity_limit) if capacity_limit is not None else 0.0,
        "scale_factor": 1.0,
    }
    if capacity_limit is not None:
        datasets, scale_metadata = scale_datasets_to_capacity(datasets, capacity_limit)
        metadata.update(scale_metadata)

    if return_metadata:
        return datasets, all_records, metadata
    return datasets, all_records


def write_outputs(
    datasets: dict[str, pd.DataFrame],
    records: list[dict[str, Any]],
    output_dir: str | Path,
    metadata: dict[str, float] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_window_id = 1
    for dataset_name in ("train", "val", "test"):
        output_df = format_dataset_for_replay(datasets[dataset_name], start_window_id)
        output_df.to_csv(output_path / f"{dataset_name}_workload_burst.csv", index=False)
        start_window_id += len(output_df)

    pd.DataFrame(records).to_csv(output_path / "burst_insert_records.csv", index=False)
    if metadata is not None:
        pd.DataFrame([metadata]).to_csv(output_path / "scaling_metadata.csv", index=False)


def format_dataset_for_replay(df: pd.DataFrame, start_window_id: int) -> pd.DataFrame:
    if REQUEST_COL not in df.columns:
        raise ValueError(f"Missing required column: {REQUEST_COL}")
    if "workload_type" not in df.columns:
        raise ValueError("Missing required column: workload_type")

    window_ids = range(start_window_id, start_window_id + len(df))
    output_df = pd.DataFrame(
        {
            "timestamp": [(window_id - 1) * 20 for window_id in window_ids],
            "window_id": list(range(start_window_id, start_window_id + len(df))),
            "requests_per_20s": df[REQUEST_COL].round().astype(int).tolist(),
            "workload_type": df["workload_type"].tolist(),
        }
    )
    return output_df


def plot_workload_with_burst_background(df: pd.DataFrame, dataset_name: str, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    df = df.copy().reset_index(drop=True)

    plt.figure(figsize=(16, 5))
    plt.plot(
        df.index,
        df[REQUEST_COL],
        color="black",
        linewidth=1.2,
        label=REQUEST_COL,
    )

    burst_colors = {
        "Sudden spike": "#ff9999",
        "Periodic peaks": "#99ccff",
        "Irregular bursts": "#ffcc99",
        "Pattern shift": "#cc99ff",
    }

    for burst_type, color in burst_colors.items():
        burst_indices = df.index[df["workload_type"] == burst_type].tolist()
        if not burst_indices:
            continue

        start = burst_indices[0]
        prev = burst_indices[0]
        for idx in burst_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                plt.axvspan(start, prev, color=color, alpha=0.35)
                start = idx
                prev = idx
        plt.axvspan(start, prev, color=color, alpha=0.35)

    plt.title(f"{dataset_name} Workload with Burst Background")
    plt.xlabel("Window Index")
    plt.ylabel("Requests per 20s")
    plt.grid(alpha=0.3)

    patches = [
        mpatches.Patch(color=color, alpha=0.35, label=burst_type)
        for burst_type, color in burst_colors.items()
    ]
    plt.legend(
        handles=[plt.Line2D([], [], color="black", label=REQUEST_COL)] + patches,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_plots(datasets: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    plot_workload_with_burst_background(datasets["train"], "Train", output_path / "train_workload_burst_plot.png")
    plot_workload_with_burst_background(datasets["val"], "Validation", output_path / "val_workload_burst_plot.png")
    plot_workload_with_burst_background(datasets["test"], "Test", output_path / "test_workload_burst_plot.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the first 8640 workload points into train/val/test and insert burst patterns."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--divide-by", type=float, default=1.0)
    parser.add_argument("--capacity-limit", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    datasets, records, metadata = process_workload(
        df,
        seed=args.seed,
        divide_by=args.divide_by,
        capacity_limit=args.capacity_limit,
        return_metadata=True,
    )
    write_outputs(datasets, records, args.output_dir, metadata=metadata)
    if not args.no_plots:
        write_plots(datasets, args.output_dir)

    print(f"Input rows used: {DEFAULT_TOTAL_POINTS}")
    print(f"divide_by: {metadata['divide_by']}")
    print(f"pre-scale max request_count: {metadata['pre_scale_max_request_count']}")
    print(f"capacity_limit: {metadata['capacity_limit']}")
    print(f"scale_factor: {metadata['scale_factor']}")
    for dataset_name in ("train", "val", "test"):
        counts = datasets[dataset_name]["workload_type"].value_counts().to_dict()
        max_request_count = datasets[dataset_name][REQUEST_COL].max()
        print(
            f"{dataset_name}: rows={len(datasets[dataset_name])}, "
            f"max_request_count={max_request_count}, workload_type_counts={counts}"
        )
    print(f"burst records: {len(records)}")
    print(f"output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
