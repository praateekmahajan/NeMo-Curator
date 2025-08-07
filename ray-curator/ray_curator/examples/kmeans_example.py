#!/usr/bin/env python3
"""Example pipeline using FilePartitioningStage and KMeansStage."""

import argparse
import sys

from loguru import logger

from ray_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.deduplication.semantic.kmeans import KMeansStage


def main() -> int:
    """Main function to run the KMeans pipeline."""
    logger.info("Starting KMeans clustering pipeline...")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"ID column: {args.id_col}")
    logger.info(f"Embedding column: {args.embedding_col}")
    logger.info(f"Number of clusters: {args.n_clusters}")


    # Create the pipeline
    pipeline = Pipeline(
        name="kmeans_clustering_pipeline",
        description="Pipeline for K-means clustering on document embeddings",
        stages=[
            KMeansStage(
                id_col=args.id_col,
                embedding_col=args.embedding_col,
                output_path=args.output_path,
                n_clusters=args.n_clusters,
                input_path=args.input_path,
                input_filetype=args.file_type,
                input_storage_options={},
                output_storage_options={},
                input_file_limit=None,
            )
        ]
    )

    executor = RayActorPoolExecutor(
        config={
            "reserved_cpus": 1.0,  # Reserve some CPUs for system overhead
            "reserved_gpus": 0.0,
        }
    )

    # Create executor - KMeansStage requires RAFT, so we use RayActorPoolExecutor
    try:
        logger.info("Executing pipeline...")
        results = pipeline.run(executor)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error during pipeline execution: {e}")
        import traceback

        traceback.print_exc()
        return 1

    logger.info(f"Pipeline completed successfully! Results: {len(results) if results else 0} tasks")
    if results:
        for i, result in enumerate(results):
            logger.info(f"  Result {i}: {result}")

    logger.info("KMeans clustering pipeline completed successfully!")
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

    args = parser.parse_args()

    sys.exit(main())
