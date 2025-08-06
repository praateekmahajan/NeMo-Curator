from typing import TYPE_CHECKING, Any

import ray
from ray.util.actor_pool import ActorPool

from ray_curator.tasks import EmptyTask, Task

if TYPE_CHECKING:
    from ray_curator.stages.base import ProcessingStage

from loguru import logger

from ray_curator.backends.base import BaseExecutor
from ray_curator.backends.utils import register_loguru_serializer

from ..ray_actor.object_coordinator import ObjectStoreCoordinator
from ..ray_data.utils import execute_setup_on_node
from .adapter import RayActorPoolStageAdapter

_LARGE_INT = 2**31 - 1


class RayActorPoolExecutor(BaseExecutor):
    """Ray-based executor using ActorPool for better resource management.
    
    This executor:
    1. Creates a pool of actors per stage using Ray's ActorPool
    2. Uses map_unordered for better load balancing and fault tolerance
    3. Lets Ray handle object ownership and garbage collection automatically
    4. Provides better backpressure management through ActorPool
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Configuration for resource management
        self.max_actors_per_stage = self.config.get("max_actors_per_stage", 128)

    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> Any:
        """Execute the pipeline stages using ActorPool.
        
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
            self.object_coordinator = ObjectStoreCoordinator.options(name="object_coordinator_pool").remote()
            self.object_coordinator_actor_id = ray.get(self.object_coordinator.get_actor_id.remote())

            logger.info(
                f"Created ObjectStoreCoordinator for managing object ownership with actor ID {self.object_coordinator_actor_id}"
            )

            # Execute setup on node for all stages BEFORE processing begins
            execute_setup_on_node(stages)
            logger.info(f"Setup on node complete for all stages. Starting Ray Actor Pool pipeline with {len(stages)} stages")

            # Initialize with initial tasks owned by coordinator to establish ownership
            if initial_tasks:
                current_task_refs = [ray.put(task, _owner=self.object_coordinator) for task in initial_tasks]
            else:
                current_task_refs = [ray.put(EmptyTask, _owner=self.object_coordinator)]
            # Process through each stage with ActorPool
            for i, stage in enumerate(stages):
                logger.info(f"\nProcessing stage {i + 1}/{len(stages)}: {stage}")
                logger.info(f"  Input tasks: {len(current_task_refs)}")

                if not current_task_refs:
                    logger.warning(f"{stage} - No tasks to process, skipping stage")
                    continue

                # Create actor pool for this stage
                num_actors = self._calculate_optimal_actors(stage, len(current_task_refs))
                logger.info(
                    f" {stage} - Creating {num_actors} actors (CPUs: {stage.resources.cpus}, GPUs: {stage.resources.gpus})"
                )

                actor_pool = self._create_actor_pool(stage, num_actors)

                # Process tasks through this stage using ActorPool
                current_task_refs = self._process_stage_with_pool(actor_pool, stage, current_task_refs)

                # Clean up actor pool
                self._cleanup_actor_pool(actor_pool, i + 1)

                logger.info(f"  Output tasks: {len(current_task_refs)}")

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise
        else:
            # Get final results - only materialize at the very end
            final_results = ray.get(current_task_refs) if current_task_refs else []
            logger.info(f"\nPipeline completed. Final results: {len(final_results)} tasks")

            # Clean up coordinator
            logger.info("Cleaning up ObjectStoreCoordinator...")
            ray.kill(self.object_coordinator)

            return final_results
        finally:
            # Clean up all Ray resources including named actors
            logger.info("Shutting down Ray to clean up all resources...")
            ray.shutdown()

    def _calculate_optimal_actors(self, stage: "ProcessingStage", num_tasks: int) -> int:
        """Calculate optimal number of actors for a stage."""
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
        )

        # Calculate max actors based on GPU constraints
        max_actors_gpu = (
            int(available_gpus // stage.resources.gpus) if stage.resources.gpus > 0 else _LARGE_INT
        )

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

    def _create_actor_pool(self, stage: "ProcessingStage", num_actors: int) -> ActorPool:
        """Create an ActorPool for a specific stage."""
        actors = []
        for _ in range(num_actors):
            actor = RayActorPoolStageAdapter.options(
                num_cpus=stage.resources.cpus,
                num_gpus=stage.resources.gpus
            ).remote(stage, self.object_coordinator)
            actors.append(actor)

        # Setup actors on their respective nodes
        # Note: setup_on_node was already called for all stages before processing began

        return ActorPool(actors)

    def _process_stage_with_pool(self, actor_pool: ActorPool, _stage: "ProcessingStage", task_refs: list[ray.ObjectRef]) -> list[ray.ObjectRef]:
        """Process all tasks through the stage using ActorPool but keeping ObjectRefs."""
        if not task_refs:
            return []

        # Get batch size from one of the actors
        first_actor = next(iter(actor_pool._idle_actors))  # Access internal idle actors set
        stage_batch_size: int = ray.get(first_actor.get_batch_size.remote())

        # Create batches of tasks according to the stage's preferred batch size
        task_batches = []
        for i in range(0, len(task_refs), stage_batch_size):
            batch = task_refs[i : i + stage_batch_size]
            task_batches.append(batch)

        logger.info(f"    Processing {len(task_batches)} batches with ActorPool")

        # Submit all batches to the actor pool without using map_unordered
        # to avoid automatic ray.get() calls
        for batch in task_batches:
            actor_pool.submit(lambda actor, batch: actor.process_task_batch.remote(batch), batch)

        # Collect ObjectRefs directly without materializing them
        all_result_refs = []
        while actor_pool.has_next():
            # Get the ObjectRef result without calling ray.get()
            # This returns an ObjectRef[list[ray.ObjectRef]]
            batch_result_ref = self._get_next_objectref_unordered(actor_pool)
            all_result_refs.append(batch_result_ref)

        # Now we need to flatten the list of ObjectRef[list[ray.ObjectRef]] into list[ray.ObjectRef]
        # We'll need to materialize just the outer structure to get the inner ObjectRefs
        all_results = []
        if all_result_refs:
            # Get all batch results at once
            batch_results = ray.get(all_result_refs)
            for batch_result in batch_results:
                if batch_result:
                    all_results.extend(batch_result)

        return all_results

    def _get_next_objectref_unordered(self, actor_pool: ActorPool) -> list[ray.ObjectRef]:
        """Get the next result from ActorPool without materializing ObjectRefs.
        
        This bypasses ActorPool's built-in ray.get() call to keep results as ObjectRefs.
        """
        if not actor_pool.has_next():
            msg = "No more results to get"
            raise StopIteration(msg)
            
        # Wait for any task to complete (similar to get_next_unordered but without ray.get)
        res, _ = ray.wait(list(actor_pool._future_to_actor), num_returns=1, timeout=None)
        if res:
            [future] = res
        else:
            msg = "Failed to get result from actor pool"
            raise RuntimeError(msg)
            
        # Get the actor and clean up tracking
        i, actor = actor_pool._future_to_actor.pop(future)
        actor_pool._return_actor(actor)
        del actor_pool._index_to_future[i]
        actor_pool._next_return_index = max(actor_pool._next_return_index, i + 1)
        
        # Return the ObjectRef directly instead of calling ray.get(future)
        # The future is the result from process_task_batch.remote() which returns list[ray.ObjectRef]
        return future

    def _cleanup_actor_pool(self, actor_pool: ActorPool, stage_num: int) -> None:
        """Clean up actors in the pool."""
        logger.info(f"    Cleaning up ActorPool for stage {stage_num}...")

        # Get all actors from the pool
        all_actors = list(actor_pool._idle_actors) + [actor for actor, _ in actor_pool._future_to_actor.items()]

        for i, actor in enumerate(all_actors):
            try:
                ray.get(actor.teardown.remote())
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"      Warning: Error cleaning up actor {i}: {e}")

        logger.info(f"    Stage {stage_num} ActorPool cleaned up.")
