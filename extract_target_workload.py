from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path(r"E:\bishe\azurefunctions-dataset2019")
DEFAULT_HASH_FUNCTION = "9ccb62facdfef7e77ee93544df32b1c59f002b8de1d3fd77b56958c2a3237739"
DEFAULT_OUTPUT = Path("target_function_workload_d01_d14.csv")


def day_file(input_dir: Path, day: int) -> Path:
    return input_dir / f"invocations_per_function_md.anon.d{day:02d}.csv"


def read_function_values(
    csv_path: Path,
    hash_function: str,
    start_col: int = 4,
    end_col: int = 1444,
    chunksize: int = 5000,
) -> list[Any]:
    matches: list[dict[str, Any]] = []

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        matched = chunk.loc[chunk["HashFunction"] == hash_function]
        if not matched.empty:
            matches.extend(matched.to_dict("records"))

    if not matches:
        raise ValueError(f"No row found for HashFunction={hash_function} in {csv_path}")
    if len(matches) > 1:
        raise ValueError(f"Found {len(matches)} rows for HashFunction={hash_function} in {csv_path}")

    row = pd.Series(matches[0])
    values = row.iloc[start_col:end_col].tolist()
    expected_count = end_col - start_col
    if len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} minute values in {csv_path}, found {len(values)}")
    return values


def build_workload(
    input_dir: str | Path,
    hash_function: str,
    start_day: int = 1,
    end_day: int = 14,
    start_col: int = 4,
    end_col: int = 1444,
    chunksize: int = 5000,
    workload_type: str = "normal",
) -> pd.DataFrame:
    input_path = Path(input_dir)
    values: list[Any] = []

    for day in range(start_day, end_day + 1):
        values.extend(
            read_function_values(
                day_file(input_path, day),
                hash_function,
                start_col=start_col,
                end_col=end_col,
                chunksize=chunksize,
            )
        )

    new_df = pd.DataFrame(
        {
            "minute": range(1, len(values) + 1),
            "request_count": values,
        }
    )
    new_df["workload_type"] = workload_type
    return new_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one Azure Functions HashFunction workload from d01 through d14."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--hash-function", default=DEFAULT_HASH_FUNCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=14)
    parser.add_argument("--start-col", type=int, default=4)
    parser.add_argument("--end-col", type=int, default=1444)
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument("--workload-type", default="normal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = build_workload(
        input_dir=args.input_dir,
        hash_function=args.hash_function,
        start_day=args.start_day,
        end_day=args.end_day,
        start_col=args.start_col,
        end_col=args.end_col,
        chunksize=args.chunksize,
        workload_type=args.workload_type,
    )
    workload.to_csv(args.output, index=False)
    print(f"Wrote {len(workload)} rows to {args.output}")


if __name__ == "__main__":
    main()
