"""Task data structures for the ray-curator pipeline framework."""

import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ray_curator.utils.performance_utils import StagePerfStats

if TYPE_CHECKING:
    import numpy as np

T = TypeVar("T")


@dataclass
class Task(ABC, Generic[T]):
    """Abstract base class for tasks in the pipeline.
    A task represents a batch of data to be processed. Different modalities
    (text, audio, video) can implement their own task types.
    Attributes:
        task_id: Unique identifier for this task
        dataset_name: Name of the dataset this task belongs to
        dataframe_attribute: Name of the attribute that contains the dataframe data. We use this for input/output validations.
        _stage_perf: List of stages perfs this task has passed through
    """

    task_id: str
    dataset_name: str
    data: T
    _stage_perf: list[StagePerfStats] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict)
    _uuid: str = field(init=False, default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        self.validate()

    @property
    @abstractmethod
    def num_items(self) -> int:
        """Get the number of items in this task."""

    def add_stage_perf(self, perf_stats: StagePerfStats) -> None:
        """Add performance stats for a stage."""
        self._stage_perf.append(perf_stats)

    def __repr__(self) -> str:
        subclass_name = self.__class__.__name__
        return f"{subclass_name}(task_id={self.task_id}, dataset_name={self.dataset_name})"

    @abstractmethod
    def validate(self) -> bool:
        """Validate the task data."""


@dataclass
class _EmptyTask(Task[None]):
    """Dummy task for testing."""

    @property
    def num_items(self) -> int:
        return 0

    def validate(self) -> bool:
        """Validate the task data."""
        return True


# Empty tasks are just used for `ls` stages
EmptyTask = _EmptyTask(task_id="empty", dataset_name="empty", data=None)


class TaskPerfUtils:
    """Utilities for aggregating stage performance metrics from tasks.

    Example output format:
    {
        "StageA": {"process_time": [...], "actor_idle_time": [...], "read_time_s": [...], ...},
        "StageB": {"process_time": [...], ...}
    }
    """

    @staticmethod
    def collect_stage_metrics(tasks: list[Task]) -> dict[str, dict[str, "np.ndarray[float]"]]:
        """Collect per-stage metric lists from a list of tasks.

        The returned mapping aggregates both built-in StagePerfStats metrics and any
        custom_stats recorded by stages.

        Args:
            tasks: Iterable of tasks, each having a `_stage_perf: list[StagePerfStats]` attribute.

        Returns:
            Dict mapping stage_name -> metric_name -> list of numeric values.
        """
        import numpy as np

        stage_to_metrics: dict[str, dict[str, list[float]]] = {}

        for task in tasks or []:
            perfs = task._stage_perf or []
            for perf in perfs:
                stage_name = perf.stage_name

                if stage_name not in stage_to_metrics:
                    stage_to_metrics[stage_name] = defaultdict(list)

                metrics_dict = stage_to_metrics[stage_name]

                # Built-in metrics
                for metric_name, metric_value in perf.items():
                    metrics_dict[metric_name].append(metric_value)

        return {k: np.ndarray(v) for k, v in stage_to_metrics.items()}
