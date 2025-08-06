from typing import TYPE_CHECKING, Any

import ray

from ray_curator.tasks import EmptyTask, Task

if TYPE_CHECKING:
    from ray_curator.stages.base import ProcessingStage


from loguru import logger

from ray_curator.backends.base import BaseExecutor
from ray_curator.backends.utils import register_loguru_serializer

from ..ray_data.utils import execute_setup_on_node
from .adapter import RayActorStageAdapter
from .object_coordinator import ObjectStoreCoordinator

_LARGE_INT = 2**31 - 1


class RayActorExecutor(BaseExecutor):
    """Ray-based executor with dynamic actor allocation per stage.

    This executor:
    1. Creates a central ObjectStoreCoordinator to manage object ownership
    2. Dynamically calculates optimal number of actors per stage
    3. Creates min(num_tasks, available_resources) actors for each stage
    4. Processes all tasks for a stage, then cleans up actors
    5. Moves to next stage and repeats the process
    6. Maximizes resource utilization at each stage
    7. Avoids serialization to driver by using coordinator for object ownership
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Configuration for resource management
        self.max_actors_per_stage = self.config.get("max_actors_per_stage", 128)  # Safety limit

    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> Any:  # noqa: ANN401
        """Execute the pipeline stages with dynamic actor allocation.

        Args:
            stages: List of processing stages to execute
            initial_tasks: Initial tasks to process (can be None for empty start)

        Returns:
            List of final processed tasks
        """
        if not stages:
            return []

        try:
            # Initialize Ray and register loguru serializer
            ray.init(ignore_reinit_error=True)
            register_loguru_serializer()

            # Create the object store coordinator
            self.object_coordinator = ObjectStoreCoordinator.options(name="object_coordinator").remote()
            self.object_coordinator_actor_id = ray.get(self.object_coordinator.get_actor_id.remote())

            logger.info(
                f"Created ObjectStoreCoordinator for managing object ownership with actor ID {self.object_coordinator_actor_id}"
            )

            # Execute setup on node for all stages BEFORE processing begins
            execute_setup_on_node(stages)
            logger.info(
                f"Setup on node complete for all stages. Starting Ray Actor pipeline with {len(stages)} stages"
            )

            # Initialize with initial tasks owned by coordinator to establish ownership
            if initial_tasks:
                current_task_refs = [ray.put(task, _owner=self.object_coordinator) for task in initial_tasks]
            else:
                current_task_refs = [ray.put(EmptyTask, _owner=self.object_coordinator)]
            # Process through each stage with dynamic actor allocation
            for i, stage in enumerate(stages):  # Process ALL stages, including the first one
                logger.info(f"\nProcessing stage {i + 1}/{len(stages)}: {stage}")
                logger.info(f"  Input tasks: {len(current_task_refs)}")

                if not current_task_refs:
                    logger.warning(f"{stage} - No tasks to process, skipping stage")
                    continue

                # Calculate optimal number of actors for this stage
                num_actors = self._calculate_optimal_actors(stage, len(current_task_refs))
                logger.info(
                    f" {stage} - Creating {num_actors} actors (CPUs: {stage.resources.cpus}, GPUs: {stage.resources.gpus})"
                )

                # Create actors for this stage with coordinator actor ID
                stage_actors = self._create_stage_actors(stage, num_actors)

                # Note: setup_on_node was already called for all stages before processing began

                # Process tasks through this stage
                current_task_refs = self._process_stage_batch(stage_actors, current_task_refs)

                # Clean up stage actors immediately to free resources for next stage
                self._cleanup_stage_actors(stage_actors, i + 1)

                logger.info(f"  Output tasks: {len(current_task_refs)}")

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise
        else:
            # Get final results - only materialize at the very end for return to user
            final_results = ray.get(current_task_refs) if current_task_refs else []
            logger.info(f"\nPipeline completed. Final results: {len(final_results)} tasks")

            return final_results
        finally:
            # Clean up all Ray resources including named actors
            logger.info("Shutting down Ray to clean up all resources...")
            ray.shutdown()

    def _calculate_optimal_actors(self, stage: "ProcessingStage", num_tasks: int) -> int:
        """Calculate optimal number of actors for a stage.

        Args:
            stage: Processing stage
            num_tasks: Number of tasks to process

        Returns:
            Optimal number of actors to create
        """
        # Get available resources (not total cluster resources)
        available_resources = ray.available_resources()
        available_cpus = available_resources.get("CPU", 0)
        available_gpus = available_resources.get("GPU", 0)

        # Reserve resources for ObjectStoreCoordinator, StageCallCounter, and system overhead
        reserved_cpus = 3.0  # 1 for ObjectStoreCoordinator + 1 for StageCallCounter + 1 for system
        available_cpus = max(0, available_cpus - reserved_cpus)

        # Calculate max actors based on CPU constraints
        max_actors_cpu = (
            int(available_cpus // stage.resources.cpus) if stage.resources.cpus > 0 else _LARGE_INT
        )  # Large number for unlimited CPU
        # Calculate max actors based on GPU constraints
        max_actors_gpu = (
            int(available_gpus // stage.resources.gpus) if stage.resources.gpus > 0 else _LARGE_INT
        )  # Large number for unlimited GPU

        # Take the minimum constraint
        max_actors_resources = min(max_actors_cpu, max_actors_gpu)

        # Ensure we don't create more actors than configured maximum
        max_actors_resources = min(max_actors_resources, self.max_actors_per_stage, stage.num_workers() or _LARGE_INT)
        # Don't create more actors than tasks
        optimal_actors = min(num_tasks, max_actors_resources)

        # Ensure at least 1 actor if we have tasks
        optimal_actors = max(1, optimal_actors) if num_tasks > 0 else 0

        logger.info(f"    Resource calculation: CPU limit={max_actors_cpu}, GPU limit={max_actors_gpu}")
        logger.info(f"    Available: {available_cpus} CPUs, {available_gpus} GPUs")
        logger.info(f"    Stage requirements: {stage.resources.cpus} CPUs, {stage.resources.gpus} GPUs")

        return optimal_actors

    def _create_stage_actors(self, stage: "ProcessingStage", num_actors: int) -> list:
        """Create actors for a specific stage.

        Args:
            stage: Processing stage
            num_actors: Number of actors to create

        Returns:
            List of actor references
        """
        actors = []
        for _ in range(num_actors):
            # Pass coordinator actor ID to each actor
            actor = RayActorStageAdapter.options(num_cpus=stage.resources.cpus, num_gpus=stage.resources.gpus).remote(
                stage, self.object_coordinator
            )
            actors.append(actor)

        return actors

    def _process_stage_batch(self, stage_actors: list, task_refs: list[ray.ObjectRef]) -> list[ray.ObjectRef]:
        """Process all tasks through the stage actors.

        Args:
            stage_actors: List of actor references for this stage
            task_refs: List of task object references to process

        Returns:
            List of processed task object references
        """
        if not task_refs or not stage_actors:
            return []

        # Get the batch size from the first actor's stage
        # All actors for a stage should have the same batch_size
        stage_batch_size: int = ray.get(stage_actors[0].get_batch_size.remote())

        # Create batches of tasks according to the stage's preferred batch size
        task_batches = []
        for i in range(0, len(task_refs), stage_batch_size):
            batch = task_refs[i : i + stage_batch_size]
            task_batches.append(batch)

        # Submit batches to actors in round robin fashion
        futures = []
        for i, batch in enumerate(task_batches):
            actor = stage_actors[i % len(stage_actors)]
            future = actor.process_tasks.remote(batch)
            futures.append(future)

        # Wait for all actors to complete and collect results
        all_results = []
        if futures:
            results_per_actor = ray.get(futures)
            for result_refs in results_per_actor:
                if result_refs:
                    all_results.extend(result_refs)

        return all_results

    def _cleanup_stage_actors(self, stage_actors: list, stage_num: int) -> None:
        """Clean up actors for a completed stage.

        Args:
            stage_actors: List of actor references to clean up
            stage_num: Stage number for logging
        """
        logger.info(f"    Cleaning up {len(stage_actors)} actors for stage {stage_num}...")

        for i, actor in enumerate(stage_actors):
            try:
                ray.get(actor.teardown.remote())
                ray.kill(actor)
            except Exception as e:  # noqa: BLE001, PERF203
                logger.warning(f"      Warning: Error cleaning up actor {i}: {e}")

        logger.info(f"    Stage {stage_num} actors cleaned up.")
