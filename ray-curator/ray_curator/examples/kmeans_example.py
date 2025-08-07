#!/usr/bin/env python3
"""Example pipeline using FilePartitioningStage and KMeansStage."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from ray_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.deduplication.semantic import KMeansStage, PairwiseStage


def main() -> int:
    """Main function to run the KMeans pipeline."""
    logger.info("Starting KMeans clustering pipeline...")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"ID column: {args.id_col}")
    logger.info(f"Embedding column: {args.embedding_col}")
    logger.info(f"Number of clusters: {args.n_clusters}")

    kmeans_input_path = args.input_path
    kmeans_output_path = str(Path(args.output_path) / "kmeans")
    pairwise_input_path = str(Path(args.output_path) / "pairwise")

    # KMeans pipeline
    kmeans_pipeline = Pipeline(
        name="kmeans_clustering_pipeline",
        description="Pipeline for K-means clustering on document embeddings",
        stages=[
            KMeansStage(
                input_path=kmeans_input_path,
                id_col=args.id_col,
                embedding_col=args.embedding_col,
                output_path=kmeans_output_path,
                n_clusters=args.n_clusters,
                input_filetype=args.file_type,
                input_storage_options={},
                output_storage_options={},
                input_file_limit=None,
            )
        ]
    )
    kmeans_executor = RayActorPoolExecutor(
        config={
            "reserved_cpus": 1.0,  # Reserve some CPUs for system overhead
            "reserved_gpus": 0.0,
        }
    )
    kmeans_results = kmeans_pipeline.run(kmeans_executor)
    for task_out in kmeans_results:
        logger.info(task_out)

    # Pairwise pipeline
    pairwise_pipeline = Pipeline(
        name="pairwise_clustering_pipeline",
        description="Pipeline for pairwise clustering on document embeddings",
        stages=[
            PairwiseStage(
                id_col=args.id_col,
                embedding_col=args.embedding_col,
                input_path=kmeans_output_path,
                output_path=pairwise_input_path,
            )
        ]
    )
    pairwise_executor = RayActorPoolExecutor(
        config={
            "reserved_cpus": 1.0,  # Reserve some CPUs for system overhead
            "reserved_gpus": 0.0,
        }
    )
    pairwise_results = pairwise_pipeline.run(pairwise_executor)
    for task_out in pairwise_results:
        logger.info(task_out)

    return 0


if __name__ == "__main__":
    logger.info("Starting KMeans clustering pipeline...")
    parser = argparse.ArgumentParser(description="Run a pipeline with FilePartitioningStage and KMeansStage")

    # Required arguments
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input parquet files or directory containing parquet files",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to output directory for clustered results"
    )
    parser.add_argument("--id-col", type=str, default="id", help="Name of the ID column in the parquet files")
    parser.add_argument(
        "--embedding-col", type=str, default="embeddings", help="Name of the embedding column in the parquet files"
    )
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of clusters to create")

    parser.add_argument(
        "--file-type", type=str, default="parquet", help="File type to include (default: 'parquet')"
    )
    parser.add_argument("--executor", type=str, default="ray", help="Executor to use (default: 'ray_data') or 'xenna'")

    args = parser.parse_args()

    sys.exit(main())
