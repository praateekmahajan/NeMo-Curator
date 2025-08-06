import os
from typing import Any, Literal

import cudf
import cupy as cp
import numpy as np
from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.deduplication.io_utils import DeduplicationIO
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask


def get_array_from_df(df: cudf.DataFrame, embedding_col: str) -> cp.ndarray:
    """
    Convert a column of lists to a 2D array.
    """
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)


class KMeansStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    """KMeans clustering stage that requires RAFT for distributed processing."""

    def __init__(
        self,
        id_col: str,
        embedding_col: str,
        output_path: str,
        n_clusters: int,
        # id_generator: IdGenerator,
        distance_metric_to_use: Literal["l2", "cosine"] | None = "cosine",
        verbose: bool = False,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 1,
        init: Literal["k-means||", "random"] | np.ndarray = "k-means||",
        n_init: int | Literal["auto"] = 1,
        oversampling_factor: float = 2.0,
        max_samples_per_batch: int = 1 << 15,
    ):
        """KMeans clustering stage that requires RAFT for distributed processing.

        Args:
            id_col (str): The column name of the id column.
            embedding_col (str): The column name of the embedding column.
            output_path (str): The path to the output directory.
            n_clusters (int): The number of clusters to create.
                # id_generator (IdGenerator): The id generator to use.
            which_to_keep (Literal["hard", "easy", "random"]): TODO: Add explanation
            distance_metric_to_use (Literal["l2", "cosine"] | None): TODO: Add explanation
            verbose (bool): Whether to print verbose output.
            max_iter (int): The maximum number of iterations to run.
            tol (float): Tolerance for stopping criteria of the kmeans algorithm.
            random_state (int): Seed for the random number generator. Unseeded by default. Does not currently fully guarantee the exact same results.
            init (Literal["k-means||", "random"] | np.ndarray): 'scalable-k-means++' or 'k-means||': Uses fast and stable scalable kmeans++ initialization. 'random': Choose 'n_cluster' observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
            n_init (int | Literal["auto"]): Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
            oversampling_factor (float): The amount of points to sample in scalable k-means++ initialization for potential centroids. Increasing this value can lead to better initial centroids at the cost of memory. The total number of centroids sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.
            max_samples_per_batch (int): The number of data samples to use for batches of the pairwise distance computation. This computation is done throughout both fit predict. The default should suit most cases. The total number of elements in the batched pairwise distance computation is max_samples_per_batch * n_clusters. It might become necessary to lower this number when n_clusters becomes prohibitively large.
        """
        self.id_col = id_col
        self.embedding_col = embedding_col
        self.output_path = output_path
        self.n_clusters = n_clusters
        self.distance_metric_to_use = distance_metric_to_use
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch

        self._name = "KMeansStage"
        self._resources = Resources(cpus=1.0, gpus=1.0)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        df = self.read_parquet(task.data, columns=[self.id_col, self.embedding_col])
        cupy_arr = get_array_from_df(df, self.embedding_col)
        centroids = self.kmeans.fit_predict(cupy_arr).astype(cp.int32)
        df["centroid"] = centroids
        df.to_parquet(
            self.output_path,
            partition_file_name=task.task_id,  # TODO: replace with `task._uuid`
            partition_cols=["centroid"],
            index=False,
        )

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[
                f"{os.path.join(self.output_path, f'centroid={i}', f'{task.task_id}.parquet')}"
                for i in range(self.n_clusters)
            ],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if not hasattr(self, "_raft_handle"):
            msg = "RAFT handle not found. Make sure the stage is initialized with RAFT"
            raise ValueError(msg)

        self.kmeans = cumlKMeans(
            handle=self._raft_handle,
            output_type="cupy",
            init=self.init,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            n_init=self.n_init,
            oversampling_factor=self.oversampling_factor,
            max_samples_per_batch=self.max_samples_per_batch,
            convert_dtype=False,
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        return {
            "is_raft_actor": True,
        }
