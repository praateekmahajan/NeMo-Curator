from typing import TYPE_CHECKING

from loguru import logger

from ray_curator.backends.experimental.utils import get_available_cpu_gpu_resources

if TYPE_CHECKING:
    from ray_curator.stages.base import ProcessingStage

_LARGE_INT = 2**31 - 1


def calculate_optimal_actors_for_stage(
    stage: "ProcessingStage",
    num_tasks: int,
    reserved_cpus: float = 0.0,
    reserved_gpus: float = 0.0,
) -> int:
    """Calculate optimal number of actors for a stage."""
    # Get available resources (not total cluster resources)
    available_cpus, available_gpus = get_available_cpu_gpu_resources()
    # Reserve resources for system overhead
    available_cpus = max(0, available_cpus - reserved_cpus)
    available_gpus = max(0, available_gpus - reserved_gpus)

    # Calculate max actors based on CPU constraints
    max_actors_cpu = int(available_cpus // stage.resources.cpus) if stage.resources.cpus > 0 else _LARGE_INT

    # Calculate max actors based on GPU constraints
    max_actors_gpu = int(available_gpus // stage.resources.gpus) if stage.resources.gpus > 0 else _LARGE_INT

    # Take the minimum constraint
    max_actors_resources = min(max_actors_cpu, max_actors_gpu)

    # Ensure we don't create more actors than configured maximum
    max_actors_resources = min(max_actors_resources, stage.num_workers() or _LARGE_INT)

    # Don't create more actors than tasks
    optimal_actors = min(num_tasks, max_actors_resources)

    # Ensure at least 1 actor if we have tasks
    optimal_actors = max(1, optimal_actors) if num_tasks > 0 else 0

    logger.info(f"    Resource calculation: CPU limit={max_actors_cpu}, GPU limit={max_actors_gpu}")
    logger.info(f"    Available: {available_cpus} CPUs, {available_gpus} GPUs")
    logger.info(f"    Stage requirements: {stage.resources.cpus} CPUs, {stage.resources.gpus} GPUs")

    return optimal_actors
