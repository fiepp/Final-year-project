import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# Offline ETS tuning script
# ------------------------------------------------------------
# This script should be run on the host machine.
#
# Important:
# ETS is not trained and saved like a neural network model.
# This script tunes the ETS hyperparameters that will later be
# used by run_ets_baseline.py on the VM.
#
# The validation logic intentionally mirrors the online VM script:
#   1. keep a rolling history,
#   2. fit ETS on the most recent history window,
#   3. predict the next 20-second request count,
#   4. append the actual observed value,
#   5. repeat.
# ============================================================


DEFAULT_TRAIN_CSV = r"G:\PycharmProjects\WorkloadTransformerProject\data\final\train_workload.csv"
DEFAULT_VAL_CSV = r"G:\PycharmProjects\WorkloadTransformerProject\data\final\val_workload.csv"
DEFAULT_RESULTS_DIR = r"G:\PycharmProjects\WorkloadTransformerProject\baselines\ets\results"

REQUEST_COLUMN_CANDIDATES = [
    "requests_per_20s",
    "request_count",
    "requests",
    "expected_requests",
]


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def request_column(df):
    for col in REQUEST_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find request-count column. Expected one of: "
        + ", ".join(REQUEST_COLUMN_CANDIDATES)
    )


def load_series(csv_path):
    df = pd.read_csv(csv_path)
    col = request_column(df)
    series = df[col].astype(float).clip(lower=0).to_numpy()
    if len(series) == 0:
        raise ValueError(f"Empty workload file: {csv_path}")
    return df, series, col


def required_history_size(history_window_size, min_history_windows, seasonal, seasonal_periods):
    size = max(history_window_size, min_history_windows)
    if seasonal != "none":
        if seasonal_periods is None:
            raise ValueError("seasonal_periods is required when seasonal is enabled")
        size = max(size, 2 * int(seasonal_periods))
    return size


def fallback_prediction(history, history_window_size):
    recent = history[-history_window_size:]
    return float(sum(recent) / len(recent))


def predict_next_requests(
    history,
    *,
    trend,
    seasonal,
    seasonal_periods,
    history_window_size,
    min_history_windows,
):
    """Match the online controller's ETS prediction behavior."""
    if len(history) < required_history_size(
        history_window_size,
        min_history_windows,
        seasonal,
        seasonal_periods,
    ):
        return fallback_prediction(history, history_window_size)

    required_size = required_history_size(
        history_window_size,
        min_history_windows,
        seasonal,
        seasonal_periods,
    )
    model_history = history[-required_size:]

    trend_arg = None if trend == "none" else trend
    seasonal_arg = None if seasonal == "none" else seasonal
    period_arg = int(seasonal_periods) if seasonal_arg else None

    try:
        model = ExponentialSmoothing(
            model_history,
            trend=trend_arg,
            seasonal=seasonal_arg,
            seasonal_periods=period_arg,
            initialization_method="estimated",
        )
        fitted_model = model.fit(optimized=True)
        return max(0.0, float(fitted_model.forecast(1)[0]))
    except Exception as exc:
        print(
            "[Warning] ETS failed, fallback to moving average: "
            f"trend={trend}, seasonal={seasonal}, period={seasonal_periods}, "
            f"history_window={history_window_size}, error={exc}"
        )
        return fallback_prediction(history, history_window_size)


def requests_to_pods(predicted_requests, pod_capacity_per_window, min_pods, max_pods):
    desired_pods = math.ceil(float(predicted_requests) / float(pod_capacity_per_window))
    desired_pods = max(int(min_pods), desired_pods)
    desired_pods = min(int(max_pods), desired_pods)
    return desired_pods


def calculate_metrics(actual, predicted, desired_pods, args):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    desired_pods = np.asarray(desired_pods, dtype=float)

    under_errors = np.maximum(actual - predicted, 0.0)
    over_errors = np.maximum(predicted - actual, 0.0)
    weighted_errors = (args.under_prediction_weight * under_errors) + over_errors

    provided_capacity = desired_pods * args.pod_capacity_per_window
    capacity_shortfall = np.maximum(actual - provided_capacity, 0.0)
    capacity_waste = np.maximum(provided_capacity - actual, 0.0)

    ideal_pods = np.ceil(actual / args.pod_capacity_per_window)
    ideal_pods = np.clip(ideal_pods, args.min_pods, args.max_pods)
    pod_error = np.abs(desired_pods - ideal_pods)

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "UnderMAE": float(np.mean(under_errors)),
        "OverMAE": float(np.mean(over_errors)),
        "WeightedError": float(np.mean(weighted_errors)),
        "MeanDesiredPods": float(np.mean(desired_pods)),
        "PodMAE": float(np.mean(pod_error)),
        "MeanCapacityShortfall": float(np.mean(capacity_shortfall)),
        "MaxCapacityShortfall": float(np.max(capacity_shortfall)),
        "MeanCapacityWaste": float(np.mean(capacity_waste)),
    }


def evaluate_candidate(train_series, val_series, args, candidate):
    history = list(map(float, train_series))
    actual_values = []
    predicted_values = []
    desired_pods_values = []

    for i, actual in enumerate(val_series):
        yhat = predict_next_requests(
            history,
            trend=candidate["trend"],
            seasonal=candidate["seasonal"],
            seasonal_periods=candidate["seasonal_periods"],
            history_window_size=candidate["history_window_size"],
            min_history_windows=args.min_history_windows,
        )

        pods = requests_to_pods(
            yhat,
            args.pod_capacity_per_window,
            args.min_pods,
            args.max_pods,
        )

        actual_values.append(float(actual))
        predicted_values.append(float(yhat))
        desired_pods_values.append(int(pods))
        history.append(float(actual))

        if args.progress and i % args.progress_every == 0:
            print(f"    progress {i}/{len(val_series)}")

    metrics = calculate_metrics(
        actual_values,
        predicted_values,
        desired_pods_values,
        args,
    )

    prediction_df = pd.DataFrame({
        "val_index": np.arange(len(val_series)),
        "actual_requests": actual_values,
        "predicted_requests": predicted_values,
        "desired_pods": desired_pods_values,
        "provided_capacity": np.asarray(desired_pods_values) * args.pod_capacity_per_window,
    })

    return metrics, prediction_df


def make_candidates(args):
    candidates = []
    for trend in parse_str_list(args.trend_options):
        for seasonal in parse_str_list(args.seasonal_options):
            if seasonal == "none":
                period_list = [None]
            else:
                period_list = parse_int_list(args.seasonal_periods_options)

            for seasonal_periods in period_list:
                for history_window_size in parse_int_list(args.history_window_size_options):
                    candidates.append({
                        "trend": trend,
                        "seasonal": seasonal,
                        "seasonal_periods": seasonal_periods,
                        "history_window_size": history_window_size,
                    })
    return candidates


def save_best_config(best_row, args, results_dir):
    def none_if_nan(value):
        if pd.isna(value):
            return None
        return value

    best_config = {
        "trend": str(best_row["trend"]),
        "seasonal": str(best_row["seasonal"]),
        "seasonal_periods": (
            None
            if none_if_nan(best_row["seasonal_periods"]) is None
            else int(best_row["seasonal_periods"])
        ),
        "history_window_size": int(best_row["history_window_size"]),
        "min_history_windows": int(args.min_history_windows),
        "pod_capacity_per_window": float(args.pod_capacity_per_window),
        "min_pods": int(args.min_pods),
        "max_pods": int(args.max_pods),
        "smoothing_level": None,
        "smoothing_trend": None,
        "smoothing_seasonal": None,
    }

    payload = {
        "purpose": (
            "Best ETS hyperparameters selected offline. Upload this JSON to the VM "
            "and pass it to run_ets_baseline.py with --ets-config-json."
        ),
        "selection_metric": args.selection_metric,
        "request_column": args.request_column_used,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "validation_points": int(best_row["validation_points"]),
        "under_prediction_weight": float(args.under_prediction_weight),
        "best_config": best_config,
        "best_validation_metrics": {
            key: float(best_row[key])
            for key in [
                "MAE",
                "RMSE",
                "UnderMAE",
                "OverMAE",
                "WeightedError",
                "MeanDesiredPods",
                "PodMAE",
                "MeanCapacityShortfall",
                "MaxCapacityShortfall",
                "MeanCapacityWaste",
            ]
            if key in best_row
        },
    }

    config_path = results_dir / "ets_best_config.json"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    command_path = results_dir / "ets_vm_command_template.txt"
    command_path.write_text(
        "Example VM command after uploading ets_best_config.json:\n\n"
        "python3 run_ets_baseline.py \\\n"
        "  --ets-config-json ~/online_experiment/results/ets_best_config.json \\\n"
        "  --train-csv ~/online_experiment/data/train_workload.csv \\\n"
        "  --test-csv ~/online_experiment/data/test_workload.csv \\\n"
        "  --duration-minutes 5 \\\n"
        "  --start-row 0 \\\n"
        "  --run-id ets_test \\\n"
        "  --save-request-log \\\n"
        "  --collect-pod-metrics\n",
        encoding="utf-8",
    )

    return config_path, command_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune ETS hyperparameters offline using train/validation CSV files. "
            "The produced JSON can be loaded by run_ets_baseline.py on the VM."
        )
    )

    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", default=DEFAULT_VAL_CSV)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)

    parser.add_argument("--trend-options", default="none,add")
    parser.add_argument("--seasonal-options", default="add,none")
    parser.add_argument("--seasonal-periods-options", default="3,9,15,30")
    parser.add_argument("--history-window-size-options", default="30,60,120")

    parser.add_argument("--min-history-windows", type=int, default=5)
    parser.add_argument("--pod-capacity-per-window", type=float, default=80.0)
    parser.add_argument("--min-pods", type=int, default=1)
    parser.add_argument("--max-pods", type=int, default=6)

    parser.add_argument("--under-prediction-weight", type=float, default=3.0)
    parser.add_argument(
        "--selection-metric",
        choices=[
            "WeightedError",
            "RMSE",
            "MAE",
            "UnderMAE",
            "PodMAE",
            "MeanCapacityShortfall",
        ],
        default="WeightedError",
    )

    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    train_df, train_series, train_col = load_series(args.train_csv)
    val_df, val_series, val_col = load_series(args.val_csv)

    if train_col != val_col:
        print(f"[Warning] train request column={train_col}, val request column={val_col}")

    args.request_column_used = val_col

    print("========== ETS offline tuning ==========")
    print(f"train_csv={args.train_csv}")
    print(f"val_csv={args.val_csv}")
    print(f"request_column={val_col}")
    print(f"train_windows={len(train_series)}")
    print(f"val_windows={len(val_series)}")
    print(f"selection_metric={args.selection_metric}")
    print(f"under_prediction_weight={args.under_prediction_weight}")

    candidates = make_candidates(args)
    print(f"candidate_count={len(candidates)}")

    results = []
    best_prediction_df = None

    for idx, candidate in enumerate(candidates, start=1):
        print("=" * 70)
        print(f"Candidate {idx}/{len(candidates)}: {candidate}")

        try:
            metrics, prediction_df = evaluate_candidate(
                train_series,
                val_series,
                args,
                candidate,
            )
        except Exception as exc:
            print(f"[Failed] {candidate}, error={exc}")
            continue

        row = {
            **candidate,
            "validation_points": len(val_series),
            **metrics,
        }
        results.append(row)

        print(
            f"MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"UnderMAE={metrics['UnderMAE']:.4f}, "
            f"WeightedError={metrics['WeightedError']:.4f}, "
            f"PodMAE={metrics['PodMAE']:.4f}, "
            f"MeanCapacityShortfall={metrics['MeanCapacityShortfall']:.4f}"
        )

    if not results:
        raise RuntimeError("No ETS candidate finished successfully.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=[args.selection_metric, "RMSE", "MAE"],
        ascending=True,
    )

    tuning_path = results_dir / "ets_val_tuning_results.csv"
    results_df.to_csv(tuning_path, index=False)

    best_row = results_df.iloc[0]
    best_candidate = {
        "trend": str(best_row["trend"]),
        "seasonal": str(best_row["seasonal"]),
        "seasonal_periods": None
        if pd.isna(best_row["seasonal_periods"])
        else int(best_row["seasonal_periods"]),
        "history_window_size": int(best_row["history_window_size"]),
    }

    _, best_prediction_df = evaluate_candidate(
        train_series,
        val_series,
        args,
        best_candidate,
    )
    prediction_path = results_dir / "ets_best_val_predictions.csv"
    best_prediction_df.to_csv(prediction_path, index=False)

    config_path, command_path = save_best_config(best_row, args, results_dir)

    print("\nETS validation tuning finished.")
    print(f"Aggregate results saved to: {tuning_path}")
    print(f"Best validation predictions saved to: {prediction_path}")
    print(f"Best config JSON saved to: {config_path}")
    print(f"VM command template saved to: {command_path}")

    print("\nBest parameters:")
    print(best_row)


if __name__ == "__main__":
    main()
