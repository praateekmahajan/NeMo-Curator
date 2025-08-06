from typing import TYPE_CHECKING

import ray

if TYPE_CHECKING:
    from ray.actor import ActorHandle
from loguru import logger

from ray_curator.backends.base import BaseStageAdapter
from ray_curator.backends.experimental.utils import get_worker_metadata_and_node_id
from ray_curator.stages.base import ProcessingStage


@ray.remote
class RayActorStageAdapter(BaseStageAdapter):
    """Adapts ProcessingStage to Ray actors.

    This adapter can be used as a Ray remote actor to execute processing stages
    in a distributed manner with proper resource allocation.
    """

    def __init__(self, stage: ProcessingStage, object_coordinator: "ActorHandle"):
        super().__init__(stage)

        # Get runtime context for worker metadata
        node_info, worker_metadata = get_worker_metadata_and_node_id()

        # Create WorkerMetadata with actor information
        self.worker_metadata = worker_metadata
        self.node_info = node_info

        # Setup the stage when the actor is created, passing worker metadata
        self.stage.setup(worker_metadata)

        self._batch_size = self.stage.batch_size
        if self._batch_size is None:
            logger.warning(f"using Ray Actors, batch size is not set for stage {self.stage}. Setting it to 1.")
            self._batch_size = 1

        self.object_coordinator = object_coordinator

    def get_batch_size(self) -> int:
        """Get the batch size for this stage."""
        return self._batch_size  # type:  ignore[reportReturnType]

    def process_tasks(self, task_refs: list[ray.ObjectRef]) -> list[ray.ObjectRef]:
        """Process tasks from Ray object references and return results as object references.

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

    def setup_on_node(self) -> None:
        """Setup method for Ray actors.

        Creates NodeInfo and WorkerMetadata objects with meaningful information
        from Ray's runtime context, similar to how Xenna does it.
        
        Note: This method is not used in the current implementation since we use
        the Ray Data pattern of calling setup_on_node before actor creation.
        """
        # Get node information from Ray's runtime context
        # Create WorkerMetadata with actor information
        # For Ray actors, we use the actor_id as worker_id and store basic allocation info

        # Call the base adapter's setup_on_node with meaningful metadata
        super().setup_on_node(self.node_info, self.worker_metadata)

    def get_node_id(self) -> str:
        """Get the node ID of the actor."""
        return ray.get_runtime_context().get_node_id()
