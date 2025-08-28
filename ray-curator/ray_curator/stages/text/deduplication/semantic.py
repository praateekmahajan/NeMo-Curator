# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Monolithic Text Semantic Deduplication Workflow.

This module contains a complete end-to-end workflow for text semantic deduplication:
1. Embedding generation from text data
2. Semantic deduplication using clustering and pairwise similarity
3. Optional duplicate removal based on identified duplicates
"""

import os
import time
from typing import Any, Literal

from loguru import logger

# Ray Curator imports
from ray_curator.backends.base import BaseExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
    create_id_generator_actor,
    kill_id_generator_actor,
    write_id_generator_to_disk,
)
from ray_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow
from ray_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
from ray_curator.stages.text.embedders import EmbeddingCreatorStage
from ray_curator.stages.text.io.reader import JsonlReader, ParquetReader
from ray_curator.stages.text.io.writer import ParquetWriter
from ray_curator.utils.file_utils import create_or_overwrite_dir


class TextSemanticDeduplicationWorkflow:
    """
    Monolithic workflow for end-to-end text semantic deduplication.

    This workflow combines:
    1. Text embedding generation (configurable executor)
    2. Semantic deduplication (configurable executor for pairwise stage)
    3. Duplicate removal (configurable executor)

    Supports flexible executor configuration - can use a single executor for all stages
    or different executors for different phases.
    """

    def __init__(  # noqa: PLR0913
        self,
        # Input/Output configuration
        input_path: str | list[str],
        output_path: str,
        perform_removal: bool = True,
        # Embedding generation parameters
        text_field: str = "text",
        embedding_field: str = "embeddings",
        model_identifier: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length: int = 512,
        model_inference_batch_size: int = 256,
        hf_token: str | None = None,
        files_per_partition: int = 1,
        # Semantic deduplication parameters
        n_clusters: int = 100,
        id_field: str = CURATOR_DEDUP_ID_STR,
        embedding_dim: int | None = None,
        metadata_fields: list[str] | None = None,
        distance_metric: Literal["cosine", "l2"] = "cosine",
        which_to_keep: Literal["hard", "easy", "random"] = "hard",
        eps: float | None = 0.01,
        # ID generator parameters
        use_id_generator: bool = False,
        id_generator_state_file: str | None = None,
        # I/O parameters
        input_filetype: Literal["jsonl", "parquet"] = "parquet",
        input_file_extensions: list[str] | None = None,
        output_filetype: Literal["jsonl", "parquet"] = "parquet",
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        # Execution parameters
        verbose: bool = True,
        clear_output: bool = True,
    ):
        """
        Initialize the text semantic deduplication workflow.

        Args:
            input_path: Path(s) to input files containing text data
            output_path: Directory to write deduplicated output
            perform_removal: Whether to perform duplicate removal (True) or just identify duplicates (False)

            # Embedding generation parameters
            text_field: Name of the text field in input data
            embedding_field: Name of the embedding field to create
            model_identifier: HuggingFace model identifier for embeddings
            max_seq_length: Maximum sequence length for tokenization
            model_inference_batch_size: Batch size for model inference
            hf_token: HuggingFace token for private models
            files_per_partition: Number of files per partition for reading

            # Semantic deduplication parameters
            n_clusters: Number of clusters for K-means
            id_field: Name of the ID field in the data
            embedding_dim: Embedding dimension (for memory estimation)
            metadata_fields: List of metadata field names to preserve
            distance_metric: Distance metric for similarity ("cosine" or "l2")
            which_to_keep: Strategy for ranking within clusters ("hard", "easy", "random")
            eps: Epsilon value for duplicate identification (None to skip)

            # ID generator parameters
            use_id_generator: Whether to use ID generator for document IDs
            id_generator_state_file: Path to save/load ID generator state

            # Executor configuration
            executor: Single executor for all stages or tuple of (embedding, pairwise, removal) executors

            # I/O parameters
            input_filetype: Type of input files ("jsonl" or "parquet")
            input_file_extensions: List of file extensions to process
            output_filetype: Type of output files ("jsonl" or "parquet")
            read_kwargs: Keyword arguments for reading files
            write_kwargs: Keyword arguments for writing files

            # Execution parameters
            verbose: Enable verbose output
            clear_output: Clear output directory before running
        """
        # Core paths
        self.input_path = input_path
        self.output_path = output_path
        self.perform_removal = perform_removal

        # Intermediate paths
        self.embeddings_path = os.path.join(output_path, "embeddings")
        self.semantic_dedup_path = os.path.join(output_path, "semantic_dedup")
        self.final_output_path = os.path.join(output_path, "deduplicated")

        # Embedding generation parameters
        self.text_field = text_field
        self.embedding_field = embedding_field
        self.model_identifier = model_identifier
        self.max_seq_length = max_seq_length
        self.model_inference_batch_size = model_inference_batch_size
        self.hf_token = hf_token
        self.files_per_partition = files_per_partition

        # Semantic deduplication parameters
        self.n_clusters = n_clusters
        self.id_field = id_field
        self.embedding_dim = embedding_dim
        self.metadata_fields = metadata_fields or []
        self.distance_metric = distance_metric
        self.which_to_keep = which_to_keep
        self.eps = eps

        # ID generator parameters
        self.use_id_generator = use_id_generator
        self.id_generator_state_file = id_generator_state_file

        # I/O parameters
        self.input_filetype = input_filetype
        self.input_file_extensions = input_file_extensions
        self.output_filetype = output_filetype
        self.read_kwargs = read_kwargs or {}
        self.write_kwargs = write_kwargs or {}

        # Execution parameters
        self.verbose = verbose
        self.clear_output = clear_output

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate workflow configuration."""
        if self.perform_removal and self.eps is None:
            logger.warning(
                "perform_removal=True but eps=None. No duplicates will be identified for removal. "
                "Consider setting eps to a small value (e.g., 0.01) to enable duplicate identification."
            )

        if self.use_id_generator and self.id_generator_state_file is None:
            # Auto-generate ID generator state file path
            dataset_name = (
                os.path.basename(self.input_path[0])
                if isinstance(self.input_path, list)
                else os.path.basename(self.input_path)
            )
            self.id_generator_state_file = os.path.join(self.output_path, f"id_generator_{dataset_name}.json")

    def _setup_directories(self) -> None:
        """Setup output directories."""
        if self.clear_output:
            create_or_overwrite_dir(self.output_path, storage_options=self.write_kwargs.get("storage_options"))

        create_or_overwrite_dir(self.embeddings_path, storage_options=self.write_kwargs.get("storage_options"))
        create_or_overwrite_dir(self.semantic_dedup_path, storage_options=self.write_kwargs.get("storage_options"))

        if self.perform_removal:
            create_or_overwrite_dir(self.final_output_path, storage_options=self.write_kwargs.get("storage_options"))

    def _run_embedding_generation(self, executor: BaseExecutor) -> list[Any]:
        """Run embedding generation stage."""
        logger.info("Starting embedding generation stage...")

        pipeline = Pipeline(
            name="text_semantic_dedup_embedding",
            description="Generate embeddings from text data for semantic deduplication",
        )

        # Reader stage
        if self.input_filetype == "jsonl":
            reader = JsonlReader(
                file_paths=self.input_path,
                files_per_partition=self.files_per_partition,
                fields=(
                    ([self.id_field] if not self.use_id_generator else []) + [self.text_field, *self.metadata_fields]
                ),
                _generate_ids=self.use_id_generator,
                read_kwargs=self.read_kwargs,
            )
        elif self.input_filetype == "parquet":
            reader = ParquetReader(
                file_paths=self.input_path,
                files_per_partition=self.files_per_partition,
                fields=(
                    ([self.id_field] if not self.use_id_generator else []) + [self.text_field, *self.metadata_fields]
                ),
                read_kwargs=self.read_kwargs,
                _generate_ids=self.use_id_generator,
            )
        else:
            msg = f"Input filetype {self.input_filetype} not supported yet"
            raise NotImplementedError(msg)

        pipeline.add_stage(reader)

        # Embedding generation stage
        embedding_stage = EmbeddingCreatorStage(
            model_identifier=self.model_identifier,
            text_field=self.text_field,
            embedding_field=self.embedding_field,
            max_seq_length=self.max_seq_length,
            model_inference_batch_size=self.model_inference_batch_size,
            hf_token=self.hf_token,
        )
        pipeline.add_stage(embedding_stage)

        # Writer stage
        if self.output_filetype == "parquet":
            writer = ParquetWriter(
                path=self.embeddings_path,
                fields=[self.id_field, self.embedding_field, *self.metadata_fields],
                write_kwargs=self.write_kwargs,
            )
        else:
            msg = f"Output filetype {self.output_filetype} not supported yet"
            raise NotImplementedError(msg)

        pipeline.add_stage(writer)

        return pipeline.run(executor)

    def _run_semantic_deduplication(self, executor: BaseExecutor) -> dict[str, Any]:
        """Run semantic deduplication stage."""
        logger.info("Starting semantic deduplication stage...")

        workflow = SemanticDeduplicationWorkflow(
            input_path=self.embeddings_path,
            output_path=self.semantic_dedup_path,
            n_clusters=self.n_clusters,
            id_field=self.id_field,
            embedding_field=self.embedding_field,
            embedding_dim=self.embedding_dim,
            metadata_fields=self.metadata_fields,
            distance_metric=self.distance_metric,
            which_to_keep=self.which_to_keep,
            eps=self.eps,
            read_kwargs=self.read_kwargs,
            write_kwargs=self.write_kwargs,
            verbose=self.verbose,
        )

        return workflow.run(pairwise_executor=executor)

    def _run_duplicate_removal(self, executor: BaseExecutor) -> list[Any]:
        """Run duplicate removal stage."""
        if not self.perform_removal:
            logger.info("Skipping duplicate removal (perform_removal=False)")
            return []

        logger.info("Starting duplicate removal stage...")

        # Find the duplicates file from semantic deduplication
        duplicates_file = None
        duplicates_dir = os.path.join(self.semantic_dedup_path, "duplicates")
        if os.path.exists(duplicates_dir):
            for file in os.listdir(duplicates_dir):
                if file.endswith(".parquet"):
                    duplicates_file = os.path.join(duplicates_dir, file)
                    break

        if duplicates_file is None:
            logger.warning("No duplicates file found, skipping removal")
            return []

        workflow = TextDuplicatesRemovalWorkflow(
            # Use the original dataset as input so final outputs have original columns
            input_path=self.input_path,
            ids_to_remove_path=duplicates_file,
            output_path=self.final_output_path,
            id_generator_path=self.id_generator_state_file if self.use_id_generator else None,
            # Respect original input filetype; final outputs use configured output_filetype
            input_filetype=self.input_filetype,
            output_filetype=self.output_filetype,
            # Read and write all fields to preserve the original schema
            input_fields=None,
            output_fields=None,
            input_id_field=self.id_field,
            ids_to_remove_duplicate_id_field=self.id_field,
            input_kwargs=self.read_kwargs,
            output_kwargs=self.write_kwargs,
        )

        return workflow.run(executor=executor)

    def _log_configuration(self) -> None:
        """Log workflow configuration."""
        logger.info("=" * 80)
        logger.info("TEXT SEMANTIC DEDUPLICATION WORKFLOW CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Perform removal: {self.perform_removal}")

        logger.info("Embedding generation:")
        logger.info(f"  - Model: {self.model_identifier}")
        logger.info(f"  - Text field: {self.text_field}")
        logger.info(f"  - Embedding field: {self.embedding_field}")
        logger.info(f"  - Max sequence length: {self.max_seq_length}")
        logger.info(f"  - Batch size: {self.model_inference_batch_size}")
        logger.info(f"  - Executor: {type(self.embedding_executor).__name__}")

        logger.info("Semantic deduplication:")
        logger.info(f"  - Number of clusters: {self.n_clusters}")
        logger.info(f"  - ID field: {self.id_field}")
        logger.info(f"  - Distance metric: {self.distance_metric}")
        logger.info(f"  - Which to keep: {self.which_to_keep}")
        logger.info(f"  - Epsilon (similarity threshold): {self.eps}")
        logger.info(f"  - Pairwise executor: {type(self.pairwise_executor).__name__}")

        if self.perform_removal:
            logger.info("Duplicate removal:")
            logger.info(f"  - Removal executor: {type(self.removal_executor).__name__}")

        logger.info(f"Use ID generator: {self.use_id_generator}")
        if self.use_id_generator:
            logger.info(f"  - ID generator state file: {self.id_generator_state_file}")

        logger.info("=" * 80)

    def run(
        self, executor: BaseExecutor | tuple[BaseExecutor, BaseExecutor, BaseExecutor] | None = None
    ) -> dict[str, Any]:
        """
        Run the complete text semantic deduplication workflow.

        Returns:
            Dictionary with results and timing information from all stages
        """

        if isinstance(executor, tuple):
            if len(executor) != 3:
                msg = f"Expected 3 executors in tuple, got {len(executor)}"
                raise ValueError(msg)
            embedding_executor, pairwise_executor, removal_executor = executor
        else:
            # Use same executor for all stages
            if executor is None:
                from ray_curator.backends.xenna import XennaExecutor

                executor = XennaExecutor()
            embedding_executor = pairwise_executor = removal_executor = executor

        # Expose executors as attributes for logging and downstream access
        self.embedding_executor = embedding_executor
        self.pairwise_executor = pairwise_executor
        self.removal_executor = removal_executor

        total_start_time = time.time()

        try:
            # Setup
            self._setup_directories()
            self._log_configuration()

            # Setup ID generator if needed
            if self.use_id_generator:
                logger.info(f"Setting up ID generator, state will be saved to: {self.id_generator_state_file}")
                try:
                    create_id_generator_actor()
                except ValueError as e:
                    if "already taken" in str(e):
                        logger.info("ID generator actor already exists, using existing actor")
                    else:
                        raise

            # Stage 1: Embedding generation
            embedding_start_time = time.time()
            embedding_results = self._run_embedding_generation(embedding_executor)
            embedding_end_time = time.time()
            embedding_time = embedding_end_time - embedding_start_time

            logger.success(f"Embedding generation completed in {embedding_time:.2f} seconds")

            # Stage 2: Semantic deduplication
            semantic_start_time = time.time()
            semantic_results = self._run_semantic_deduplication(pairwise_executor)
            semantic_end_time = time.time()
            semantic_time = semantic_end_time - semantic_start_time

            logger.success(f"Semantic deduplication completed in {semantic_time:.2f} seconds")

            # Stage 3: Duplicate removal (optional)
            removal_results = []
            removal_time = 0.0
            if self.perform_removal:
                removal_start_time = time.time()
                removal_results = self._run_duplicate_removal(removal_executor)
                removal_end_time = time.time()
                removal_time = removal_end_time - removal_start_time

                logger.success(f"Duplicate removal completed in {removal_time:.2f} seconds")

            # Calculate total time
            total_end_time = time.time()
            total_time = total_end_time - total_start_time

            # Cleanup ID generator if it was created
            if self.use_id_generator:
                try:
                    logger.info(f"Saving ID generator state to: {self.id_generator_state_file}")
                    write_id_generator_to_disk(self.id_generator_state_file)
                    logger.info("Killing ID generator actor...")
                    kill_id_generator_actor()
                    logger.info("ID generator cleanup completed")
                except Exception as cleanup_error:
                    logger.warning(f"Error during ID generator cleanup: {cleanup_error}")

            # Log final summary
            logger.success("=" * 80)
            logger.success("TEXT SEMANTIC DEDUPLICATION WORKFLOW COMPLETED")
            logger.success("=" * 80)
            logger.success(f"Total execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
            logger.info(f"Embedding generation time: {embedding_time:.2f} seconds")
            logger.info(f"Semantic deduplication time: {semantic_time:.2f} seconds")
            if self.perform_removal:
                logger.info(f"Duplicate removal time: {removal_time:.2f} seconds")
            if semantic_results.get("total_duplicates_identified", 0) > 0:
                logger.success(
                    f"Total documents identified as duplicates: {semantic_results['total_duplicates_identified']}"
                )
            logger.success("=" * 80)

        except Exception as e:
            logger.error(f"Text semantic deduplication workflow failed: {e}")
            raise

        return {
            "total_execution_time": total_time,
            "embedding_execution_time": embedding_time,
            "semantic_execution_time": semantic_time,
            "removal_execution_time": removal_time,
            "embedding_results": embedding_results,
            "semantic_results": semantic_results,
            "removal_results": removal_results,
            "embeddings_path": self.embeddings_path,
            "semantic_dedup_path": self.semantic_dedup_path,
            "final_output_path": self.final_output_path if self.perform_removal else None,
        }
