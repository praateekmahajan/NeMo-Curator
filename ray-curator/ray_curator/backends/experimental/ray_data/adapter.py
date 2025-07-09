"""Ray Data adapter for processing stages."""

import inspect
from typing import Any

import ray
from loguru import logger
from ray.data import Dataset

from ray_curator.backends.base import BaseStageAdapter
from ray_curator.stages.base import ProcessingStage
from ray_curator.utils.performance_utils import StageTimer

from .setup_utils import setup_stage


class RayDataStageAdapter(BaseStageAdapter):
    """Adapts ProcessingStage to Ray Data operations.

    This adapter converts stages to work with Ray Data datasets by:
    1. Working directly with Task objects (no dictionary conversion)
    2. Using Ray Data's map_batches for parallel processing
    3. Handling single and batch processing modes
    4. Using stateful transforms (classes) for efficient setup handling
    """

    def __init__(self, stage: ProcessingStage):
        super().__init__(stage)

        self._batch_size = self.stage.batch_size
        if self._batch_size is None and self.stage.resources.gpus > 0:
            logger.warning(f"When using Ray Data, batch size is not set for GPU stage {self.stage}. Setting it to 1.")
            self._batch_size = 1

    @property
    def batch_size(self) -> int | None:
        """Get the batch size for this stage."""
        return self._batch_size

    def _calculate_concurrency(self) -> int | tuple[int, int]:
        """Calculate concurrency based on available resources and stage requirements.
        
        Returns:
            int or tuple[int, int]: Number of actors or (min_actors, max_actors) range
        """
        # If explicitly set, use the specified number of workers
        if self.stage.num_workers is not None:
            return max(1, self.stage.num_workers)
        
        # Get available resources from Ray
        try:
            available_resources = ray.available_resources()
        except Exception as e:
            logger.warning(f"Could not get Ray resources: {e}. Using default concurrency.")
            # Fallback to reasonable defaults
            if self.stage.resources.gpus > 0:
                return max(1, int(self.stage.resources.gpus))
            else:
                return (1, 4)  # min=1, max=4 actors
        
        # Calculate based on CPU and GPU requirements
        max_cpu_actors = float('inf')
        max_gpu_actors = float('inf')
        
        # CPU constraint
        if self.stage.resources.cpus > 0:
            available_cpus = available_resources.get('CPU', 0)
            max_cpu_actors = int(available_cpus / self.stage.resources.cpus)
        
        # GPU constraint
        if self.stage.resources.gpus > 0:
            available_gpus = available_resources.get('GPU', 0)
            max_gpu_actors = int(available_gpus / self.stage.resources.gpus)
        
        # Take the minimum of CPU and GPU constraints
        max_actors = min(max_cpu_actors, max_gpu_actors)
        
        # Ensure minimum is 1 and maximum is reasonable
        max_actors = max(1, int(max_actors))
        
        # For GPU stages, use exact number; for CPU stages, use range
        if self.stage.resources.gpus > 0:
            return max_actors
        else:
            return (1, max(max_actors, 4))  # Allow 1 to max_actors (minimum 4) for CPU stages

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a Ray Data dataset through this stage.

        Args:
            dataset (Dataset): Ray Data dataset containing Task objects

        Returns:
            Dataset: Processed Ray Data dataset
        """
        # TODO: Support nvdecs / nvencs
        if self.stage.resources.gpus <= 0 and (self.stage.resources.nvdecs > 0 or self.stage.resources.nvencs > 0):
            msg = "Ray Data does not support nvdecs / nvencs. Please use gpus instead."
            raise ValueError(msg)

        # Create the stateful stage processor class
        stage_processor_class = create_stateful_stage_processor(self.stage)

        # Calculate concurrency based on available resources
        concurrency = self._calculate_concurrency()

        processed_dataset = dataset.map_batches(
            stage_processor_class,  # type: ignore  # Ray Data accepts classes for stateful transforms
            concurrency=concurrency,  # Number of actors for stateful processing
            batch_size=self.batch_size,
            num_cpus=self.stage.resources.cpus,
            num_gpus=self.stage.resources.gpus,
        )

        if self.stage.ray_stage_spec.get("is_fanout_stage", False):
            processed_dataset = processed_dataset.repartition(target_num_rows_per_block=1)

        return processed_dataset


def create_stateful_stage_processor(stage: ProcessingStage):
    """Create a stateful stage processor class for Ray Data.
    
    This function creates a class that wraps the stage for stateful processing
    in Ray Data. The class will:
    1. Recreate the stage from its configuration in __init__
    2. Call setup methods once per actor
    3. Delegate batch processing to the stage
    
    Args:
        stage (ProcessingStage): The stage instance to wrap
        
    Returns:
        type: A class that can be used with Ray Data's stateful transforms
    """
    stage_class = type(stage)
    stage_name = stage.__class__.__name__
    
    # Extract constructor arguments for both dataclass and non-dataclass stages
    if hasattr(stage, '__dataclass_fields__'):
        # For dataclass stages, extract field values
        stage_init_args = {}
        for field_name in stage.__dataclass_fields__:  # type: ignore  # We checked hasattr above
            stage_init_args[field_name] = getattr(stage, field_name)
    else:
        # For non-dataclass stages, use extracted init args
        stage_init_args = _extract_stage_init_args(stage)

    # Create a serializable configuration
    stage_config = {
        'stage_class': stage_class,
        'stage_name': stage_name,
        'stage_init_args': stage_init_args,
    }
    
    # Return a class that can be instantiated with the configuration
    return SerializableStageProcessor.with_config(stage_config)


class SerializableStageProcessor:
    """Serializable stateful processor that wraps a ProcessingStage for Ray Data.
    
    This class avoids closure variables to ensure it can be properly serialized
    by Ray when used in stateful transforms.
    """
    
    def __init__(self, stage_config: dict[str, Any]):
        """Initialize the stage processor with configuration.
        
        Args:
            stage_config: Dictionary containing stage class, name, and init args
        """
        self.stage_config = stage_config
        self.stage = None
        self._timer = None
        
    @classmethod
    def with_config(cls, config: dict[str, Any]):
        """Create a processor class with embedded configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            A processor class that can be instantiated by Ray Data
        """
        class ConfiguredProcessor(cls):
            def __init__(self):
                super().__init__(config)
                
        # Set meaningful name for the dynamically created class
        ConfiguredProcessor.__name__ = f"{config['stage_name']}Processor"
        ConfiguredProcessor.__qualname__ = f"{config['stage_name']}Processor"
        
        return ConfiguredProcessor
    
    def _recreate_stage(self):
        """Recreate the stage instance from configuration."""
        stage_class = self.stage_config['stage_class']
        stage_init_args = self.stage_config['stage_init_args']
        
        if stage_init_args:
            self.stage = stage_class(**stage_init_args)
        else:
            # Fallback: try to create with no args
            try:
                self.stage = stage_class()
            except Exception as e:
                logger.warning(f"Could not recreate stage {stage_class.__name__}: {e}")
                raise
    
    def _setup_stage(self) -> None:
        """Setup the stage once per Ray Data actor."""
        if self.stage is None:
            self._recreate_stage()
            
        if self.stage is None:
            raise RuntimeError("Failed to recreate stage during setup")
            
        try:
            # Call setup_on_node and setup methods
            # These will be called once per actor, not per batch
            setup_stage(
                stage=self.stage,
                setup_fn=self.stage.setup,
                setup_on_node_fn=self.stage.setup_on_node
            )
            
            # Initialize performance timer after stage setup
            self._timer = StageTimer(self.stage)
            
            logger.debug(f"Setup completed for {self.stage}")
        except Exception as e:
            logger.error(f"Setup failed for {self.stage}: {e}")
            raise
    
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process a batch of Task objects.
        
        Args:
            batch: Dictionary with arrays/lists representing a batch of Task objects
            
        Returns:
            Dictionary with arrays/lists representing processed Task objects
        """
        # Ensure stage is set up (this is called once per actor)
        if self.stage is None:
            self._setup_stage()
            
        if self.stage is None:
            raise RuntimeError("Stage not properly initialized")
            
        if self._timer is None:
            raise RuntimeError("Timer not properly initialized")
            
        tasks = batch["item"]
        
        # Calculate input data size for timer
        input_size = sum(task.num_items for task in tasks)
        
        # Initialize performance timer for this batch
        self._timer.reinit(input_size)
        
        with self._timer.time_process(input_size):
            # Always use process_batch - it handles both batch and single processing
            results = self.stage.process_batch(tasks)
        
        # Log performance stats and add to result tasks
        _, stage_perf_stats = self._timer.log_stats()
        for task in results:
            task.add_stage_perf(stage_perf_stats)
        
        # Return the results as Ray Data expects them
        return {"item": results}


def _extract_stage_init_args(stage: ProcessingStage) -> dict[str, Any]:
    """Extract constructor arguments for non-dataclass stages.
    
    This function inspects the stage's __init__ method signature and
    extracts the current attribute values to use for reconstruction.
    
    Args:
        stage (ProcessingStage): The stage instance to analyze
        
    Returns:
        dict[str, Any]: Dictionary of argument names to values
    """
    try:
        # Get the __init__ method signature
        sig = inspect.signature(stage.__init__)
        init_args = {}
        
        # Skip 'self' parameter
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Try to get the current value of this attribute
            if hasattr(stage, param_name):
                init_args[param_name] = getattr(stage, param_name)
            elif hasattr(stage, f"_{param_name}"):
                # Check for private attributes (e.g., _batch_size)
                init_args[param_name] = getattr(stage, f"_{param_name}")
            elif param.default != inspect.Parameter.empty:
                # Use default value if available
                init_args[param_name] = param.default
            else:
                logger.warning(f"Could not find value for parameter '{param_name}' in stage {stage.__class__.__name__}")
        
        return init_args
        
    except Exception as e:
        logger.warning(f"Could not extract init args for stage {stage.__class__.__name__}: {e}")
        return {}


def create_named_ray_data_stage_adapter(stage: ProcessingStage) -> RayDataStageAdapter:
    """Create a named Ray Data stage adapter.

    This function is kept for backward compatibility but now delegates to
    the new stateful approach.

    Args:
        stage (ProcessingStage): Processing stage to adapt

    Returns:
        RayDataStageAdapter: Ray Data stage adapter with stateful processing
    """
    return RayDataStageAdapter(stage)
