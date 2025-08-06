import ray


@ray.remote
class ObjectStoreCoordinator:
    """Central coordinator actor that owns object references to prevent garbage collection.

    This actor serves as a long-lived object store manager that maintains ownership
    of intermediate results between pipeline stages, avoiding serialization to the driver.
    """

    def get_actor_id(self) -> str:
        """Get the actor ID of this coordinator."""
        return ray.get_runtime_context().get_actor_id()
