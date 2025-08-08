#!/usr/bin/env python3
"""Example pipeline using FilePartitioningStage and KMeansStage."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from ray_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.deduplication.semantic import IdentifySemanticDuplicatesStage, KMeansStage, PairwiseStage


def main() -> int:
    """Main function to run the KMeans pipeline."""
    logger.info("Starting KMeans clustering pipeline...")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"ID column: {args.id_col}")
    logger.info(f"Embedding column: {args.embedding_col}")
    logger.info(f"Number of clusters: {args.n_clusters}")
    logger.info(f"Cosine similarity threshold: {args.cosine_sim_threshold}")

    kmeans_input_path = args.input_path
    kmeans_output_path = str(Path(args.output_path) / "kmeans")
    pairwise_input_path = str(Path(args.output_path) / "pairwise")
    duplicates_output_path = str(Path(args.output_path) / "duplicates")

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
        ],
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
        ],
    )

    if args.cosine_sim_threshold is not None:
        from ray_curator.stages.resources import Resources
        duplicate_identify_stage =             IdentifySemanticDuplicatesStage(
                threshold=args.cosine_sim_threshold,
                output_path=duplicates_output_path,
                verbose=True,
            )

        if args.use_gpu_for_duplicates:
            duplicate_identify_stage.with_(resources=Resources(cpus=1.0, gpus=1.0))
        else:
            duplicate_identify_stage.with_(resources=Resources(cpus=1.0, gpus=0.0))
        pairwise_pipeline.add_stage(duplicate_identify_stage)

    if args.executor == "xenna":
        from ray_curator.backends.xenna import XennaExecutor

        pairwise_executor = XennaExecutor()
    else:
        from ray_curator.backends.experimental.ray_data import RayDataExecutor

        pairwise_executor = RayDataExecutor()

    pairwise_results = pairwise_pipeline.run(pairwise_executor)
    for task_out in pairwise_results:
        logger.info(task_out)

    for path in [kmeans_input_path, pairwise_input_path, duplicates_output_path]:
        df = pd.read_parquet(path)
        logger.info(f"{path=} {df.columns} {len(df)}")

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

    parser.add_argument("--file-type", type=str, default="parquet", help="File type to include (default: 'parquet')")
    parser.add_argument("--executor", type=str, default="ray", help="Executor to use (default: 'ray_data') or 'xenna'")

    # New argument for identify duplicates stage
    parser.add_argument(
        "--cosine-sim-threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for identifying duplicates (e.g., 0.8). If not provided, identify duplicates stage is skipped.",
    )
    parser.add_argument(
        "--use-gpu-for-duplicates",
        action="store_true",
        help="Use GPU for duplicates stage (default: False)",
    )

    args = parser.parse_args()

    sys.exit(main())
