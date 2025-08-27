# NeMo-Curator Nightly Benchmarking Framework

A robust benchmarking framework for running systematic performance tests on ray-curator with comprehensive logging, experiment tracking, and error handling.

## Quick Start

```bash
# Basic usage
python -m nightly_benchmarking.run \
    --matrix nightly_benchmarking/matrix.yaml \
    --datasets nightly_benchmarking/dataset_paths.json \
```
```
# With mlflow and a session name
MLFLOW_TRACKING_URI=http://your-mlflow:3569 python -m nightly_benchmarking.run \
    --matrix nightly_benchmarking/dedup_removal_benchmark_matrix.yaml \
    --datasets nightly_benchmarking/dataset_paths.json \
    --sink mlflow \
    --session-name dedup_removal
```

## Overview

The benchmarking framework consists of:

- **Driver** (`run.py`): Orchestrates benchmark execution with Ray isolation
- **Matrix Configuration** (YAML): Defines what benchmarks to run and their parameters
- **Benchmark Scripts**: Individual performance tests (e.g., `removal_benchmark.py`)
- **Sinks**: Result tracking (MLflow, W&B, or none)
- **Utilities**: Ray cluster management, dataset resolution, logging

## Architecture

```
Session (nightly-run-20231201-143022)
├── Entry 1 (removal_curator_dedup_id_xenna)
│   ├── scratch/           # Temporary data (auto-cleaned)
│   ├── ray_cluster/       # Ray cluster files
│   ├── logs/              # stdout/stderr logs
│   ├── artifacts/         # Environment snapshots
│   └── benchmark_results/ # Performance results (see format below)
└── Entry 2 (removal_curator_dedup_id_ray_data)
    ├── scratch/
    ├── ray_cluster/
    ├── logs/
    ├── artifacts/
    └── benchmark_results/
```

## Matrix Configuration

Create a YAML file defining your benchmark matrix:

```yaml
# Example: removal_benchmark_matrix.yaml
default_timeout_s: 3600
results_dir: /path/to/results
artifacts_dir: /path/to/artifacts

# MLflow configuration
mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI}
  experiment: ray-curator-removal

# Optional Slack notifications
slack:
  webhook_env: SLACK_WEBHOOK_URL

entries:
  - name: removal_curator_dedup_id_xenna
    script: removal_benchmark.py
    args: >-
      --input-path {dataset:cleaned_cc,jsonl}
      --ids-to-remove-path /path/to/duplicates
      --output-path {session}/scratch
      --executor xenna
      --id-field _curator_dedup_id
      --files-per-partition 1
      --max-files 100
    timeout_s: 1800
    ray:
      num_cpus: 128
      num_gpus: 0
      enable_object_spilling: false
```

### Template Placeholders

- `{dataset:name,format}`: Resolved from dataset_paths.json
- `{session}`: Replaced with entry directory path
- `${ENV_VAR}`: Environment variable substitution

## Dataset Configuration

Create `dataset_paths.json` with dataset locations:

```json
{
  "cleaned_cc": {
    "jsonl": "/path/to/cleaned_common_crawl/",
    "parquet": "/path/to/cleaned_common_crawl_parquet/"
  },
  "another_dataset": {
    "jsonl": "/path/to/other/data/"
  }
}
```

## Writing Benchmark Scripts

Your benchmark script must:

1. **Accept `--benchmark-results-path`** as a required argument
2. **Write results** in the expected format to that path
3. **Return proper exit codes**: 0 for success, non-zero for failure

### Required Result Format

Write these files to `benchmark_results_path`:

#### `params.json` - Input Parameters
```json
{
  "executor": "xenna",
  "input_path": "/path/to/input",
  "id_field": "_curator_dedup_id",
  "files_per_partition": 1,
  "max_files": 100
}
```

#### `metrics.json` - Performance Metrics
```json
{
  "is_success": true,
  "time_taken": 45.2,
  "num_removed": 87932,
  "num_output_tasks": 100,
  "stage_file_partitioning_process_time_mean": 2.3,
  "stage_file_partitioning_process_time_std": 0.5,
  "stage_jsonl_reader_process_time_sum": 120.4
}
```

#### `tasks.pkl` - Detailed Task Data
Binary pickle file containing task objects with detailed performance metrics. Used by `TaskPerfUtils.collect_stage_metrics()` for stage-level analysis.

### Example Benchmark Script Structure

```python
#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

def run_benchmark(args) -> dict:
    """Run your benchmark and return results."""
    try:
        # Your benchmark logic here
        success = True
        metrics = {"time_taken": 45.2, "num_removed": 1000}
        tasks = []  # Your task objects
    except Exception:
        success = False
        metrics = {}
        tasks = []

    return {
        "params": vars(args),
        "metrics": {"is_success": success, **metrics},
        "tasks": tasks
    }

def write_results(results: dict, output_path: str):
    """Write results in expected format."""
    Path(output_path).mkdir(parents=True, exist_ok=True)

    with open(f"{output_path}/params.json", "w") as f:
        json.dump(results["params"], f, indent=2)
    with open(f"{output_path}/metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)
    with open(f"{output_path}/tasks.pkl", "wb") as f:
        pickle.dump(results["tasks"], f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-results-path", required=True)
    # Add your other arguments...
    args = parser.parse_args()

    results = run_benchmark(args)
    write_results(results, args.benchmark_results_path)

    # Return proper exit code
    return 0 if results["metrics"]["is_success"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
```

## Logging Output

The framework provides clean hierarchical logging:

```
Started session nightly-run-20231201-143022...
	Running removal_curator_dedup_id_xenna
		Running command python nightly_benchmarking/scripts/removal_benchmark.py ...
		✅ Run Succeeded in 45.2 seconds
		Logs found in /path/to/logs
	Running removal_curator_dedup_id_ray_data
		Running command python nightly_benchmarking/scripts/removal_benchmark.py ...
		❌ Run Failed in 12.3 seconds
		Logs found in /path/to/logs
```

## Experiment Tracking

### MLflow Integration

Results are automatically tracked in MLflow with:
- **Parent Run**: Session-level tracking
- **Child Runs**: Individual entry results
- **Parameters**: All input parameters logged
- **Metrics**: Performance metrics logged
- **Artifacts**: Environment snapshots (pip freeze, conda list)

### Accessing Results

- View experiments at your MLflow UI
- Session and run URLs are printed during execution
- Results also saved locally in `results_dir`

## Error Handling

The framework handles errors at multiple levels:

1. **Script-level failures**: Script returns non-zero exit code
2. **Driver-level failures**: Infrastructure issues (Ray cluster, directories, etc.)
3. **Session resilience**: Continues processing after individual entry failures

All errors are logged with full context and tracebacks.

## Command Line Options

```bash
python -m nightly_benchmarking.run [OPTIONS]

Options:
  --matrix PATH          Path to YAML matrix config (required)
  --datasets PATH        Path to dataset_paths.json (required)
  --sink {mlflow,wandb,none}  Metrics sink (default: none)
  --session-name NAME    Custom session name (default: nightly-run-<timestamp>)
```

## Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow server URL
- `RAY_ADDRESS`: Automatically managed by framework

## Directory Structure

```
results_dir/
└── nightly-run-20231201-143022/          # Session directory
    ├── removal_curator_dedup_id_xenna/    # Entry directory
    │   ├── scratch/                       # Temporary workspace (cleaned)
    │   ├── ray_cluster/                   # Ray files (cleaned)
    │   ├── logs/                          # Execution logs
    │   │   ├── stdout.log
    │   │   ├── stderr.log
    │   │   └── ray_startup.log
    │   ├── artifacts/                     # Environment snapshots
    │   │   ├── pip-freeze.txt
    │   │   └── conda-explicit.txt
    │   ├── benchmark_results/             # Your benchmark output
    │   │   ├── params.json                # Input parameters
    │   │   ├── metrics.json               # Performance metrics
    │   │   └── tasks.pkl                  # Detailed task data
    │   └── results.jsonl                  # Normalized driver results
    └── removal_curator_dedup_id_ray_data/ # Another entry...
```

## Best Practices

1. **Resource Management**: Each entry gets isolated Ray cluster
2. **Timeout Configuration**: Set appropriate timeouts for long-running benchmarks
3. **Result Validation**: Always include `is_success` in metrics
4. **Error Reporting**: Use proper exit codes and descriptive error messages
5. **Data Cleanup**: Temporary data in `scratch/` is auto-cleaned
6. **Reproducibility**: All parameters and environment info captured

## Troubleshooting

### Common Issues

- **Ray connection errors**: Check that entries don't conflict on ports/resources
- **MLflow errors**: Ensure `MLFLOW_TRACKING_URI` is accessible
- **Directory permissions**: Verify write access to `results_dir`
- **Timeouts**: Increase `timeout_s` for long-running benchmarks

### Debug Information

- **Logs**: Check `logs/stderr.log` for detailed error messages
- **Ray logs**: Ray cluster logs in `logs/ray_startup.log`
- **Environment**: Captured in `artifacts/` for reproducibility
- **MLflow**: View experiment tracking for parameter/metric history

## Advanced Usage

### Custom Sinks

Extend the sink interface to integrate with other tracking systems:

```python
# nightly_benchmarking/sinks/custom_sink.py
class CustomSink:
    def start_session(self, session_name: str): pass
    def start_run(self, run_name: str): pass
    def log_params(self, params: dict): pass
    def log_metrics(self, metrics: dict): pass
    def log_artifact(self, path: str): pass
    def end_run(self, success: bool): pass
    def end_session(self, success: bool): pass
```

### Matrix Parameterization

Use YAML anchors and references for complex configurations:

```yaml
# Define reusable configs
ray_config: &default_ray
  num_cpus: 128
  num_gpus: 0

entries:
  - name: test_small
    ray: *default_ray
    args: --max-files 10

  - name: test_large
    ray: *default_ray
    args: --max-files 1000
```
