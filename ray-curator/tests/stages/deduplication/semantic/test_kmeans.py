"""End-to-end integration tests for KMeans semantic deduplication stage."""

# ruff: noqa: E402
import pytest

cudf = pytest.importorskip("cudf")
cuml = pytest.importorskip("cuml")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from ray_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.deduplication.semantic.kmeans import KMeansStage
from ray_curator.stages.deduplication.semantic.utils import get_array_from_df

N_CLUSTERS = 4
N_SAMPLES_PER_CLUSTER = 100
EMBEDDING_DIM = 32
RANDOM_STATE = 42


def create_clustered_dataset(  # noqa: PLR0913
    tmp_path: Path,
    n_clusters: int = N_CLUSTERS,
    n_samples_per_cluster: int = N_SAMPLES_PER_CLUSTER,
    embedding_dim: int = EMBEDDING_DIM,
    random_state: int = RANDOM_STATE,
    file_format: str = "parquet",
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Create a synthetic clustered dataset using sklearn make_blobs.

    Args:
        tmp_path: Temporary directory path
        n_clusters: Number of clusters to create
        n_samples_per_cluster: Number of samples per cluster
        embedding_dim: Dimensionality of embeddings
        random_state: Random seed for reproducibility
        file_format: Output file format ('parquet' or 'jsonl')

    Returns:
        Tuple of (input_dir_path, embeddings_array, true_labels_array)
    """
    # Create clustered data using sklearn
    X, y_true = make_blobs(  # noqa: N806
        n_samples=n_clusters * n_samples_per_cluster,
        centers=n_clusters,
        n_features=embedding_dim,
        random_state=random_state,
        cluster_std=0.5,  # Reduced cluster standard deviation for tighter clusters
    )

    # Normalize embeddings (same as KMeans stage will do)
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: N806

    # Create input directory
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Create dataframe with embeddings and IDs
    num_files = 20  # Create multiple files to test file partitioning
    samples_per_file = len(X_normalized) // num_files

    for file_idx in range(num_files):
        start_idx = file_idx * samples_per_file
        end_idx = (file_idx + 1) * samples_per_file if file_idx < num_files - 1 else len(X_normalized)
        df = pd.DataFrame(
            {
                "id": np.arange(start_idx, end_idx),
                "embeddings": X_normalized[start_idx:end_idx].tolist(),
                "true_cluster": y_true[start_idx:end_idx].tolist(),
            }
        )

        if file_format == "parquet":
            file_path = input_dir / f"data_part_{file_idx:02d}.parquet"
            df.to_parquet(file_path, index=False)
        elif file_format == "jsonl":
            file_path = input_dir / f"data_part_{file_idx:02d}.jsonl"
            df.to_json(file_path, orient="records", lines=True)
        else:
            msg = f"Unsupported file format: {file_format}"
            raise ValueError(msg)

    return input_dir, y_true


def create_kmeans_pipeline(
    input_dir: Path,
    output_dir: Path,
    n_clusters: int = N_CLUSTERS,
    embedding_dim: int = EMBEDDING_DIM,
    file_format: str = "parquet",
) -> Pipeline:
    """Create a KMeans pipeline for testing.

    Args:
        input_dir: Input directory containing test data
        output_dir: Output directory for results
        n_clusters: Number of clusters
        embedding_dim: Embedding dimensionality
        file_format: Input file format

    Returns:
        Configured pipeline
    """
    pipeline = Pipeline(name="kmeans_integration_test")

    kmeans_stage = KMeansStage(
        n_clusters=n_clusters,
        id_field="id",
        embedding_field="embeddings",
        input_path=str(input_dir),
        output_path=str(output_dir),
        embedding_dim=embedding_dim,
        input_filetype=file_format,
        distance_metric_to_use="cosine",
        verbose=True,
        random_state=RANDOM_STATE,
        max_iter=300,
        tol=1e-4,
    )

    pipeline.add_stage(kmeans_stage)
    return pipeline


def run_single_gpu_baseline(
    input_dir: Path,
    n_clusters: int = N_CLUSTERS,
    file_format: str = "parquet",
) -> np.ndarray:
    single_gpu_kmeans = cuml.KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=RANDOM_STATE,
        output_type="numpy",  # Use numpy output for easier comparison
    )

    # Read data based on file format
    if file_format == "parquet":
        df = cudf.read_parquet(input_dir)
    elif file_format == "jsonl":
        df = cudf.read_json(input_dir, lines=True)
    else:
        msg = f"Unsupported file format: {file_format}"
        raise ValueError(msg)

    embeddings = get_array_from_df(df, "embeddings")
    single_gpu_kmeans.fit(embeddings)
    df["centroid"] = single_gpu_kmeans.predict(embeddings)

    return df.sort_values("id", ignore_index=True)["centroid"].to_numpy()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "file_format",
    ["parquet", "jsonl"],
)
class TestKMeansStageIntegration:
    """Integration tests for KMeansStage comparing multi-GPU vs single-GPU results."""

    def test_multi_gpu_vs_single_gpu_consistency(
        self,
        tmp_path: Path,
        file_format: str,
    ) -> None:
        """Test that multi-GPU KMeans produces consistent results with single-GPU baseline.

        This test:
        1. Creates synthetic clustered data with known ground truth
        2. Runs single-GPU cuML KMeans as baseline
        3. Runs multi-GPU KMeansStage using RayActorPoolExecutor
        4. Verifies that both produce the same number of clusters
        5. Checks that clustering assignments are reasonable given the ground truth
        """
        # Generate synthetic clustered dataset
        input_dir, true_labels = create_clustered_dataset(tmp_path, file_format=file_format)

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Run multi-GPU pipeline first to get output data
        pipeline = create_kmeans_pipeline(input_dir, output_dir, file_format=file_format)
        executor = RayActorPoolExecutor()
        results = pipeline.run(executor)

        # Verify pipeline execution
        assert len(results) > 0, "Pipeline should produce results"

        # Step 2: Run single-GPU baseline on the input data
        single_gpu_assignments = run_single_gpu_baseline(input_dir, file_format=file_format)

        # Step 3: Compare results with multi-GPU baseline
        multi_gpu_assignments = (
            cudf.read_parquet(output_dir).sort_values("id", ignore_index=True)["centroid"].to_numpy()
        )

        from sklearn.metrics import adjusted_rand_score

        # Compare with ground truth
        multi_gpu_ari = adjusted_rand_score(multi_gpu_assignments, true_labels)
        single_gpu_ari = adjusted_rand_score(single_gpu_assignments, true_labels)

        # Both should produce reasonable clustering (not random)
        assert multi_gpu_ari > 0.99, f"Multi-GPU clustering should be better than random (got {multi_gpu_ari:.3f})"
        assert single_gpu_ari > 0.99, f"Single-GPU clustering should be better than random (got {single_gpu_ari:.3f})"

        # The key insight: both methods should produce similar quality results
        # since they're using the same algorithm on the same data
        quality_diff = abs(multi_gpu_ari - single_gpu_ari)
        assert quality_diff < 0.01, (
            f"Multi-GPU and single-GPU should produce similar quality results (difference: {quality_diff:.3f})"
        )
