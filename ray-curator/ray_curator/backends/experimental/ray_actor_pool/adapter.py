from typing import TYPE_CHECKING

import ray

if TYPE_CHECKING:
    from ray.actor import ActorHandle

from loguru import logger

from ray_curator.backends.base import BaseStageAdapter
from ray_curator.backends.experimental.utils import get_worker_metadata_and_node_id
from ray_curator.stages.base import ProcessingStage


@ray.remote
class RayActorPoolStageAdapter(BaseStageAdapter):
    """Adapts ProcessingStage to Ray actors for use with ActorPool.

    This adapter is designed to work with Ray's ActorPool for better
    resource management and load balancing.
    """

    def __init__(self, stage: ProcessingStage, object_coordinator: "ActorHandle"):
        super().__init__(stage)

        # Get runtime context for worker metadata
        node_info, worker_metadata = get_worker_metadata_and_node_id()

        # Create WorkerMetadata with actor information
        self.worker_metadata = worker_metadata
        self.node_info = node_info

        # Setup the stage when the actor is created
        self.stage.setup(worker_metadata)

        self._batch_size = self.stage.batch_size
        if self._batch_size is None:
            logger.warning(f"batch size not set for stage {self.stage}. Setting it to 1.")
            self._batch_size = 1

        self.object_coordinator = object_coordinator

    def process_task_batch(self, task_refs: list[ray.ObjectRef]) -> list[ray.ObjectRef]:
        """Process a batch of tasks and return results as object references.

        This method is designed to be called by ActorPool.map_unordered().
        It keeps results in the object store to minimize memory overhead.

        Args:
            task_refs: List of Ray object references to tasks

        Returns:
            List of Ray object references to processed tasks
        """
        # Get actual task objects from object store
        tasks = ray.get(task_refs)

        # Process the batch of tasks using the base adapter method
        results = self.process_batch(tasks)

        # Use the object coordinator as the owner for the results
        # so that if this actor is killed, then the task reference lives
        return [ray.put(result, _owner=self.object_coordinator) for result in results]

    def get_batch_size(self) -> int:
        """Get the batch size for this stage."""
        return self._batch_size

    def setup_on_node(self) -> None:
        """Setup method for Ray actors.

        Note: This method is not used in the current implementation since we use
        the Ray Data pattern of calling setup_on_node before actor creation.
        """
        super().setup_on_node(self.node_info, self.worker_metadata)

    def get_node_id(self) -> str:
        """Get the node ID of the actor."""
        return ray.get_runtime_context().get_node_id()
