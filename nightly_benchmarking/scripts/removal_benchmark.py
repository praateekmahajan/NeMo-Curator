#!/usr/bin/env python3
"""Removal logic benchmarking script for nightly benchmarking framework.

This script runs removal benchmarks with comprehensive metrics collection
using TaskPerfUtils and logs results to configured sinks.
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from ray_curator.pipeline import Pipeline
from ray_curator.stages.deduplication.removal import DuplicatesRemovalStage
from ray_curator.stages.text.io.reader import JsonlReader
from ray_curator.stages.text.io.writer import ParquetWriter
from ray_curator.tasks.utils import TaskPerfUtils
from ray_curator.utils.file_utils import get_all_file_paths_under


def create_removal_pipeline(  # noqa: PLR0913
    input_files: list[str],
    ids_to_remove_path: str,
    output_path: str,
    id_field: str = "_curator_dedup_id",
    duplicate_id_field: str = "id",
    files_per_partition: int = 1,
    max_files: int | None = None,
    use_id_generator: bool = False,
) -> Pipeline:
    """Create a removal pipeline with specified configuration."""
    if max_files:
        input_files = input_files[:max_files]

    stages = [
        JsonlReader(
            file_paths=input_files,
            files_per_partition=files_per_partition,
            _assign_ids=use_id_generator,
            fields=["adlr_id", "_curator_dedup_id", "text"],  # Include all needed columns
        ),
        DuplicatesRemovalStage(
            ids_to_remove_path=ids_to_remove_path,
            id_field=id_field,
            duplicate_id_field=duplicate_id_field,
        ),
        ParquetWriter(path=output_path),
    ]

    return Pipeline(name="removal_benchmark", stages=stages)


def collect_stage_metrics(output_tasks: list) -> dict[str, dict[str, np.ndarray]]:
    """Collect stage performance metrics using TaskPerfUtils."""
    return TaskPerfUtils.collect_stage_metrics(output_tasks)


def compute_metric_summary(stage_metrics: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Compute summary statistics for stage metrics."""
    summary = {}

    for stage_name, stage_data in stage_metrics.items():
        stage_summary = {}
        for metric_name, values in stage_data.items():
            if len(values) > 0:
                stage_summary[f"{metric_name}_mean"] = float(np.mean(values))
                stage_summary[f"{metric_name}_std"] = float(np.std(values))
                stage_summary[f"{metric_name}_min"] = float(np.min(values))
                stage_summary[f"{metric_name}_max"] = float(np.max(values))
                stage_summary[f"{metric_name}_sum"] = float(np.sum(values))
                stage_summary[f"{metric_name}_count"] = len(values)
            else:
                stage_summary[f"{metric_name}_mean"] = 0.0
                stage_summary[f"{metric_name}_std"] = 0.0
                stage_summary[f"{metric_name}_min"] = 0.0
                stage_summary[f"{metric_name}_max"] = 0.0
                stage_summary[f"{metric_name}_sum"] = 0.0
                stage_summary[f"{metric_name}_count"] = 0

        summary[stage_name] = stage_summary

    return summary


def run_removal_benchmark(  # noqa: PLR0913
    input_path: str,
    ids_to_remove_path: str,
    output_path: str,
    executor_name: str,
    id_field: str = "_curator_dedup_id",
    duplicate_id_field: str = "id",
    files_per_partition: int = 1,
    max_files: int | None = None,
    use_id_generator: bool = False,
) -> dict[str, Any]:
    """Run the removal benchmark and collect comprehensive metrics."""

    # Setup executor
    if executor_name == "ray_data":
        from ray_curator.backends.experimental.ray_data import RayDataExecutor

        executor = RayDataExecutor()
    elif executor_name == "xenna":
        from ray_curator.backends.xenna import XennaExecutor

        executor = XennaExecutor()
    else:
        msg = f"Executor {executor_name} not supported"
        raise ValueError(msg)

    # Setup ID generator if needed
    if use_id_generator:
        from ray_curator.stages.deduplication.id_generator import create_id_generator_actor

        # You may need to adjust this path or make it configurable
        id_generator_path = "/raid/prospector-lm/cleaned_exact_dedup_all_cc_sem_embeddings_80mn/xenna_1_cleaned_exact_dedup_all_cc_id_generator.json"
        create_id_generator_actor(id_generator_path)

    # Get input files
    input_files = get_all_file_paths_under(input_path)

    # Ensure output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    logger.info("Starting removal benchmark")
    run_start_time = time.perf_counter()

    try:
        # Create and run pipeline
        pipeline = create_removal_pipeline(
            input_files=input_files,
            ids_to_remove_path=ids_to_remove_path,
            output_path=output_path,
            id_field=id_field,
            duplicate_id_field=duplicate_id_field,
            files_per_partition=files_per_partition,
            max_files=max_files,
            use_id_generator=use_id_generator,
        )

        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        # Calculate removal statistics
        num_removed = sum(task._metadata.get("num_removed", 0) for task in output_tasks if hasattr(task, "_metadata"))  # noqa: SLF001
        logger.success(f"Benchmark completed in {run_time_taken:.2f}s, removed {num_removed} documents")
        success = True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Benchmark failed: {e}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_removed = 0
        success = False

    # Cleanup ID generator if used
    if use_id_generator:
        try:
            from ray_curator.stages.deduplication.id_generator import kill_id_generator_actor

            kill_id_generator_actor()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Warning: Failed to clean up ID generator: {e}")

    return {
        "params": {
            "executor": executor_name,
            "input_path": input_path,
            "ids_to_remove_path": ids_to_remove_path,
            "id_field": id_field,
            "duplicate_id_field": duplicate_id_field,
            "files_per_partition": files_per_partition,
            "max_files": max_files,
            "use_id_generator": use_id_generator,
        },
        "metrics": {
            "is_success": success,
            "time_taken": run_time_taken,
            "num_removed": num_removed,
            "num_output_tasks": len(output_tasks),
        },
        "tasks": output_tasks,
    }


def write_results(results: dict, output_path: str | None = None) -> None:
    """Write results to a file or stdout."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_path, "params.json"), "w") as f:
        json.dump(results["params"], f, indent=2)
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(results["metrics"], f, indent=2)
    with open(os.path.join(output_path, "tasks.pkl"), "wb") as f:
        pickle.dump(results["tasks"], f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Removal logic benchmark for nightly benchmarking")
    # Paths
    parser.add_argument(
        "--benchmark-results-path", required=True, help="Path to benchmark results"
    )  # we should write params.json / metrics.json / tasks.pkl here
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--ids-to-remove-path", required=True, help="Path to parquet file with IDs to remove")
    parser.add_argument("--output-path", required=True, help="Output directory for results")
    # Executor
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"], help="Executor to use")
    # Pipeline Specific
    parser.add_argument("--id-field", default="_curator_dedup_id", help="ID field in input data")
    parser.add_argument("--duplicate-id-field", default="id", help="ID field in removal file")
    parser.add_argument("--files-per-partition", type=int, default=1, help="Files per partition")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    parser.add_argument("--use-id-generator", action="store_true", help="Use ID generator")

    args = parser.parse_args()

    logger.info("=== Removal Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_removal_benchmark(
            input_path=args.input_path,
            ids_to_remove_path=args.ids_to_remove_path,
            output_path=args.output_path,
            executor_name=args.executor,
            id_field=args.id_field,
            duplicate_id_field=args.duplicate_id_field,
            files_per_partition=args.files_per_partition,
            max_files=args.max_files,
            use_id_generator=args.use_id_generator,
        )

    except Exception as e:  # noqa: BLE001
        print(f"Benchmark failed: {e}")
        results = {
            "params": vars(args),
            "metrics": {
                "is_success": False,
            },
            "tasks": [],
        }
    finally:
        write_results(results, args.benchmark_results_path)

    # Return proper exit code based on success
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
