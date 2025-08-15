import argparse
import os
import time
from typing import Literal

import cudf
import cupy as cp
import ray
import torch
from loguru import logger

from ray_curator.utils.df_utils import get_array_from_df


def pairwise_cosine_similarity_batched(
    cluster_reps: torch.Tensor,
    device: Literal["cuda", "cpu"],
    batch_size: int = 1024,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Computes pairwise cosine similarity between cluster items,
    then replace to diagonal with zeros to ignore self similarity.
    This function is useful for large clusters where the pairwise similarity matrix
    does not fit into memory.
    We use a batched approach to compute the pairwise similarity matrix in batches.
    Memory requirements are O(N*B) where N is the number of items in the cluster and B is the batch size
    instead of O(N^2) for the full matrix.
    """
    cluster_reps = cluster_reps.to(device)
    max_similarity = torch.zeros(cluster_reps.shape[0], dtype=torch.float32, device=device)
    max_indices = torch.zeros(cluster_reps.shape[0], dtype=torch.int64, device=device)
    for start_idx in range(0, cluster_reps.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, cluster_reps.shape[0])
        batch = cluster_reps[start_idx:end_idx]
        pairwise_sim_matrix = torch.mm(cluster_reps, batch.T)
        triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal=1 - start_idx)
        del batch, pairwise_sim_matrix
        max_values_and_indices = torch.max(triu_sim_matrix, dim=0)
        max_similarity[start_idx:end_idx] = max_values_and_indices[0]
        max_indices[start_idx:end_idx] = max_values_and_indices[1]

    return cp.asarray(max_similarity), cp.asarray(max_indices)


@ray.remote(num_gpus=1)
class PairwiseCosineSimilarity:
    def __init__(
        self,
        id_col: str,
        embedding_col: str,
        cluster_base_path: str,
        output_path: str,
        verbose: bool = False,
    ):
        self.id_col = id_col
        self.embedding_col = embedding_col
        self.cluster_base_path = cluster_base_path
        self.output_path = output_path
        self.verbose = verbose
        # Use this to sort later
        # self.centroids = centroids # noqa: ERA001

    def similarity(self, cluster_id: int, storage_options: dict | None) -> None:
        cluster_dir = os.path.join(self.cluster_base_path, f"centroid={cluster_id}")
        output_file_path = os.path.join(self.output_path, f"cluster_{cluster_id}.parquet")
        t1 = time.perf_counter()
        # TODO : This could also run into the cudf 2bn row limit since we're reading embeddings as well
        cluster_df = cudf.read_parquet(cluster_dir, storage_options=storage_options)
        t2 = time.perf_counter()
        if self.verbose:
            logger.info(f"Read {cluster_id=}  in {(t2 - t1):.2f} seconds")

        if len(cluster_df) == 1:
            cluster_df["id"] = cluster_df[self.id_col]
            cluster_df["max_id"] = cluster_df[self.id_col]
            cluster_df["cosine_sim_score"] = cudf.Series([0], dtype="float32")
            cluster_df = cluster_df[["_curator_id", "id", "max_id", "cosine_sim_score"]]
            cluster_df.to_parquet(output_file_path, index=False)
            return

        cluster_embeddings = torch.as_tensor(get_array_from_df(cluster_df, self.embedding_col), device="cuda")
        ids = cluster_df[self.id_col]
        curator_ids = cluster_df["_curator_id"]
        max_similarity, max_indices = pairwise_cosine_similarity_batched(cluster_embeddings, "cuda", 1024)
        max_indices_id = ids.iloc[max_indices].reset_index(drop=True)
        points_to_remove_df = cudf.DataFrame(
            {
                "_curator_id": curator_ids,
                "id": ids,
                "max_id": max_indices_id,
                "cosine_sim_score": max_similarity,
            }
        )
        t3 = time.perf_counter()
        if self.verbose:
            logger.info(f"Pairwise for {cluster_id=}  done in {(t3 - t2):.2f} seconds")
        points_to_remove_df.to_parquet(output_file_path, index=False)
        t4 = time.perf_counter()
        if self.verbose:
            logger.info(f"Write for {cluster_id=}  done in {(t4 - t3):.2f} seconds")


def initiate_run_kill_pairwise_actors(args: argparse.Namespace, num_actors: int) -> None:
    from ray.util.actor_pool import ActorPool

    from ray_curator.utils.file_utils import remove_and_create_dir

    input_path = os.path.join(args.intermediate_path, "kmeans")
    if not os.path.exists(input_path):
        msg = f"Input path {input_path} does not exist. Please run kmeans first."
        raise ValueError(msg)

    output_path = os.path.join(args.intermediate_path, "pairwise")
    remove_and_create_dir(output_path)

    t0 = time.perf_counter()
    semantic_pool = ActorPool(
        [
            PairwiseCosineSimilarity.remote(
                id_col=args.id_col,
                embedding_col=args.embedding_col,
                cluster_base_path=input_path,
                output_path=output_path,
                verbose=args.verbose,
            )
            for i in range(num_actors)
        ]
    )

    results = semantic_pool.map_unordered(
        lambda actor, cluster_id: actor.similarity.remote(cluster_id, storage_options=None),
        range(args.n_clusters),
    )
    results = list(results)
    for actor in semantic_pool._idle_actors:
        ray.kill(actor)
    t1 = time.perf_counter()
    if args.verbose:
        logger.debug(f"Pairwise Cosine Similarity output written to {output_path} in {(t1 - t0):.2f} seconds.")


if __name__ == "__main__":
    import argparse

    from ray_curator.utils.ray_utils import get_client, get_num_gpus

    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate-path", type=str, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    # Extra args
    parser.add_argument("--id-col", type=str, required=False, default="id")
    parser.add_argument("--embedding-col", type=str, required=False, default="embeddings")
    parser.add_argument("--batch-size", type=int, required=False, default=1024)
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    args = parser.parse_args()

    # Create Ray client
    client = get_client(dashboard_host="0.0.0.0")  # noqa: S104
    args = parser.parse_args()
    num_gpus = get_num_gpus()
    if num_gpus is None:
        msg = "No GPUs available"
        raise ValueError(msg)
    logger.info(f"Running Ray Client {client.dashboard_url} on {num_gpus} GPUs.")

    # Check directories
    initiate_run_kill_pairwise_actors(args, num_actors=num_gpus)
