import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from datetime import datetime
from typing import Any, Literal

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import torch
from crossfit.backend.cudf.series import create_list_series_from_1d_or_2d_ar
from huggingface_hub import PyTorchModelHubMixin
from loguru import logger
from torch import nn
from torch.nn import functional as F  # noqa: N812
from transformers import AutoConfig, AutoModel

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch


def clip_tokens(
    token_o: dict, max_length: int | None, padding_side: Literal["left", "right"] = "right"
) -> dict[str, torch.Tensor]:
    clip_len = token_o["attention_mask"].sum(axis=1).max()
    clip_len = min(clip_len, max_length) if max_length is not None else clip_len
    if padding_side == "right":
        token_o["input_ids"] = token_o["input_ids"][:, :clip_len]
        token_o["attention_mask"] = token_o["attention_mask"][:, :clip_len]
    else:
        token_o["input_ids"] = token_o["input_ids"][:, -clip_len:]
        token_o["attention_mask"] = token_o["attention_mask"][:, -clip_len:]

    token_o.pop("metadata", None)

    return token_o


class BaseHFModel(ProcessingStage[DocumentBatch, DocumentBatch], ABC):
    """Base class for HuggingFace model stages that provides common functionality."""

    def __init__(  # noqa: PLR0913
        self,
        id_col: str,
        output_col: str,
        model_name: str,
        pooling: str = "mean_pooling",
        keep_cols: list[str] | None = None,
        has_seq_order: bool = True,
        micro_batch_size: int = 256,
        verbose: bool = True,
    ):
        self.id_col = id_col
        self.output_col = output_col
        self.model_name = model_name
        self.pooling = pooling
        self.keep_cols = keep_cols
        self.has_seq_order = has_seq_order

        self.micro_batch_size = micro_batch_size
        self.verbose = verbose

        self._resources = Resources(cpus=1, gpus=1)
        self._name = "model"

    def setup_on_node(self, _node_info: NodeInfo | None, _worker_metadata: WorkerMetadata = None) -> None:
        """Setup method called once per node to download model weights.
        This is more efficient than downloading in setup() which runs per worker.
        """
        # Download model weights once per node without loading into memory
        from huggingface_hub import snapshot_download

        try:
            snapshot_download(repo_id=self.model_name, local_files_only=False)
            if self.verbose:
                logger.debug(f"Downloaded model weights for {self.model_name} on node")
        except Exception as e:
            logger.warning(f"Failed to download {self.model_name}: {e}")
            # Fallback to the original approach if snapshot_download fails
            from transformers import AutoModel

            AutoModel.from_pretrained(self.model_name)
            if self.verbose:
                logger.debug(f"Downloaded model weights for {self.model_name} on node (fallback)")

    def setup(self, _: WorkerMetadata | None) -> None:
        # Model weights should already be cached from setup_on_node
        self.model = AutoModel.from_pretrained(self.model_name).cuda().eval().float()

        if self.verbose:
            logger.debug(f"Initialized model {self.model_name} on worker")

    def yield_next_batch(self, df: pd.DataFrame) -> Generator[dict[str, torch.Tensor]]:
        """Yields a generator of model inputs for the next batch.
        We only move the microbatch to the GPU to reduce the memory overhead.

        Args:
            df (pd.DataFrame): The dataframe to process.

        Yields:
            Generator[dict[str, torch.Tensor]]: A generator of model inputs for the next batch.
        """
        # TODO : Move to device after clipping
        for i in range(0, len(df), self.micro_batch_size):
            yield clip_tokens(
                {
                    "input_ids": torch.tensor(df["input_ids"][i : i + self.micro_batch_size].tolist()).to(
                        self.model.device
                    ),
                    "attention_mask": torch.tensor(df["attention_mask"][i : i + self.micro_batch_size].tolist()).to(
                        self.model.device
                    ),
                },
                max_length=None,
                padding_side="right",
            )

    @abstractmethod
    def process_model_output(self, outputs: Any, attention_mask: torch.Tensor) -> Any:
        """Process the model outputs. This method should be implemented by subclasses."""

    @abstractmethod
    def collect_outputs(self, processed_outputs: list[Any]) -> Any:
        """Collect and concatenate the processed outputs. This method should be implemented by subclasses."""

    @abstractmethod
    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: Any) -> pd.DataFrame:
        """Create the output dataframe with the processed results. This method should be implemented by subclasses."""

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if self.verbose:
            logger.debug(f"Calling ModelActor forward pass @ {datetime.now()}")  # noqa: DTZ005

        # Move tokens to GPU
        t0 = time.perf_counter()
        # Process in chunks and write incrementally
        model_time = 0
        processed_outputs = []
        model_time_start = time.perf_counter()
        df_cpu = task.to_pandas()

        if self.verbose:
            logger.debug(f"Starting forward pass for {len(task.data):,} rows with max length of TODO")

        for model_input_batch in self.yield_next_batch(df_cpu):
            # Forward pass
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                outputs = self.model(**model_input_batch)

            processed_output = self.process_model_output(outputs, model_input_batch["attention_mask"])
            processed_outputs.append(processed_output)

        model_time = time.perf_counter() - model_time_start
        model_throughput = len(task.data) / model_time
        if self.verbose:
            logger.debug(
                f"Model forward pass for {len(task.data):,} rows took {model_time:.2f}s @ {model_throughput:.2f}r/s @ {datetime.now()}"  # noqa: DTZ005
            )

        # Collect all outputs
        collected_output = self.collect_outputs(processed_outputs)

        # Drop all columns that are not id or keep_cols
        cols_to_keep = [self.id_col] + (self.keep_cols or []) + (["_curator_seq_order"] if self.has_seq_order else [])
        df_cpu = df_cpu.drop(columns=list(set(df_cpu.columns) - set(cols_to_keep)))

        # Create output dataframe
        df_cpu = self.create_output_dataframe(df_cpu, collected_output)

        # Sort by seq_order to preserve original order from
        if self.has_seq_order:
            df_cpu = df_cpu.sort_values(by="_curator_seq_order", ignore_index=True).drop(
                columns=["_curator_seq_order"]
            )

        t1 = time.perf_counter()
        if self.verbose:
            throughput = len(df_cpu) / (t1 - t0)
            logger.debug(
                f"Model Forward pass for {len(df_cpu):,} rows "
                f"took {(t1 - t0):.2f}s "
                f"({model_time:.2f}s model;"
                f"@ {throughput:.2f}r/s @ {datetime.now()}"  # noqa: DTZ005
            )
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df_cpu,
            _stage_perf=task._stage_perf,
            _metadata=task._metadata,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.id_col, *(self.keep_cols or []), "input_ids", "attention_mask"] + (
            ["_curator_seq_order"] if self.has_seq_order else []
        )

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.id_col, *(self.keep_cols or []), self.output_col]


class HFEmbeddingModel(BaseHFModel):
    """HuggingFace model stage that produces embeddings with pooling."""

    def process_model_output(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Process model outputs to create embeddings."""
        if self.pooling == "mean_pooling":
            return self._mean_pooling(outputs, attention_mask)
        else:
            return self._get_last_token(outputs, attention_mask)


    def collect_outputs(self, processed_outputs: list[torch.Tensor]) -> cp.ndarray:
        """Collect embeddings into a cupy array."""
        cupy_array_embeddings = [cp.asarray(emb_chunk) for emb_chunk in processed_outputs]
        return cp.concatenate(cupy_array_embeddings, axis=0)

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: cp.ndarray) -> pd.DataFrame:
        """Create output dataframe with embeddings."""
        # TODO: Consider if it even makes sense to goto cudf or just concat in numpy
        df_gpu = cudf.DataFrame(index=df_cpu.index)
        df_gpu[self.output_col] = create_list_series_from_1d_or_2d_ar(collected_output, index=df_gpu.index)
        # Add output_col back to cpu dataframe
        df_cpu[self.output_col] = df_gpu.to_pandas()[self.output_col]
        del df_gpu
        return df_cpu

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        # Mask out irrelevant tokens directly without expanding the mask
        masked_embeddings = token_embeddings.masked_fill(attention_mask.unsqueeze(-1) == 0, 0.0)
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, dim=1)

    def _get_last_token(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        # Get indices of last non-padded tokens for each sequence in batch
        last_token_indices = attention_mask.sum(dim=1) - 1  # -1 for 0-based indexing
        last_token_indices = last_token_indices.to(torch.long)  # Ensure indices are of type long
        batch_size = attention_mask.size(0)
        batch_indices = torch.arange(batch_size, device=attention_mask.device)
        # Get embeddings of last non-padded tokens
        last_token_embeddings = token_embeddings[batch_indices, last_token_indices]
        return F.normalize(last_token_embeddings, dim=1)


class HFModel(BaseHFModel):
    """HuggingFace model stage that produces raw model outputs."""

    def process_model_output(self, outputs: Any, attention_mask: torch.Tensor) -> np.ndarray:
        """Process model outputs to numpy arrays."""
        logger.debug(f"Processing model outputs: {outputs.shape}")
        return outputs.cpu().numpy()

    def collect_outputs(self, processed_outputs: list[np.ndarray]) -> np.ndarray:
        """Collect outputs into a numpy array."""
        return np.concatenate(processed_outputs, axis=0)

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: np.ndarray) -> pd.DataFrame:
        """Create output dataframe with raw model outputs."""
        df_cpu[self.output_col] = collected_output
        return df_cpu


class HFClassifierModel(BaseHFModel):
    """HuggingFace model stage that produces classification probabilities and predictions."""

    def __init__(  # noqa: PLR0913
        self,
        id_col: str,
        output_col: str,
        prob_col: str,
        model_identifier: str,
        keep_cols: list[str] | None = None,
        has_seq_order: bool = True,
        micro_batch_size: int = 256,
        verbose: bool = True,
        max_chars: int = 2000,
        autocast: bool = True,
    ):
        super().__init__(
            id_col=id_col,
            output_col=output_col,
            model_name=model_identifier,
            keep_cols=keep_cols,
            has_seq_order=has_seq_order,
            micro_batch_size=micro_batch_size,
            verbose=verbose,
        )
        self.prob_col = prob_col
        self.model_identifier = model_identifier
        self.max_chars = max_chars
        self.autocast = autocast
        self.labels = None

    def setup_on_node(self, _node_info: NodeInfo | None, _worker_metadata: WorkerMetadata = None) -> None:
        """Setup method called once per node to download model weights and config.
        This is more efficient than downloading in setup() which runs per worker.
        """
        # Download model weights and config once per node without loading into memory
        from huggingface_hub import snapshot_download

        try:
            snapshot_download(repo_id=self.model_identifier, local_files_only=False)
            if self.verbose:
                logger.debug(f"Downloaded model weights and config for {self.model_identifier} on node")
        except Exception as e:
            logger.warning(f"Failed to download {self.model_identifier}: {e}")
            # Fallback to the original approach if snapshot_download fails
            from transformers import AutoConfig

            HFDebertaClassifier.from_pretrained(self.model_identifier)
            AutoConfig.from_pretrained(self.model_identifier)
            if self.verbose:
                logger.debug(f"Downloaded model weights and config for {self.model_identifier} on node (fallback)")

    def setup(self, _: WorkerMetadata | None) -> None:
        """Load the pre-trained classifier model."""
        # Load the pre-trained model with classification head
        self.model = HFDebertaClassifier.from_pretrained(self.model_identifier).cuda().eval()
        self.model.set_autocast(self.autocast)

        # Load labels from config
        config = AutoConfig.from_pretrained(self.model_identifier)
        self.labels = list(config.label2id.keys())
        self.labels.sort(key=lambda x: config.label2id[x])

    def process_model_output(self, outputs: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, np.ndarray]:
        """Process model outputs to get probabilities and predictions."""
        probs = outputs.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        if self.labels is None:
            raise RuntimeError("Model labels not initialized. Make sure setup() has been called.")

        pred_labels = [self.labels[idx] for idx in preds]

        return {
            "probs": probs,
            "preds": np.array(pred_labels),
        }

    def collect_outputs(self, processed_outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Collect outputs into numpy arrays."""
        return {
            "probs": np.concatenate([out["probs"] for out in processed_outputs], axis=0),
            "preds": np.concatenate([out["preds"] for out in processed_outputs], axis=0),
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        """Create output dataframe with predictions and probabilities."""
        df_cpu[self.output_col] = collected_output["preds"]
        df_cpu[self.prob_col] = collected_output["probs"].tolist()

        return df_cpu

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.id_col, *(self.keep_cols or []), "input_ids", "attention_mask"] + (
            ["_curator_seq_order"] if self.has_seq_order else []
        )

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.id_col, *(self.keep_cols or []), self.output_col, self.prob_col]


class HFDebertaClassifier(nn.Module, PyTorchModelHubMixin):
    """DeBERTa-based classifier that matches the NVIDIA pre-trained models."""

    def __init__(self, config: dict):
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))
        self.autocast = False

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = self.model(input_ids, attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self._forward(input_ids, attention_mask)
        else:
            return self._forward(input_ids, attention_mask)

    def set_autocast(self, autocast: bool) -> None:
        self.autocast = autocast


class HFQualityClassifier(HFClassifierModel):
    """Quality classifier using nvidia/quality-classifier-deberta."""

    def __init__(  # noqa: PLR0913
        self,
        id_col: str,
        output_col: str = "quality_pred",
        prob_col: str = "quality_prob",
        keep_cols: list[str] | None = None,
        has_seq_order: bool = True,
        micro_batch_size: int = 256,
        verbose: bool = True,
        max_chars: int = 6000,
        autocast: bool = True,
    ):
        super().__init__(
            id_col=id_col,
            output_col=output_col,
            prob_col=prob_col,
            model_identifier="nvidia/quality-classifier-deberta",
            keep_cols=keep_cols,
            has_seq_order=has_seq_order,
            micro_batch_size=micro_batch_size,
            verbose=verbose,
            max_chars=max_chars,
            autocast=autocast,
        )


class HFDomainClassifier(HFClassifierModel):
    """Domain classifier using nvidia/domain-classifier."""

    def __init__(  # noqa: PLR0913
        self,
        id_col: str,
        output_col: str = "domain_pred",
        prob_col: str = "domain_prob",
        keep_cols: list[str] | None = None,
        has_seq_order: bool = True,
        micro_batch_size: int = 256,
        verbose: bool = True,
        max_chars: int = 2000,
        autocast: bool = True,
    ):
        super().__init__(
            id_col=id_col,
            output_col=output_col,
            prob_col=prob_col,
            model_identifier="nvidia/domain-classifier",
            keep_cols=keep_cols,
            has_seq_order=has_seq_order,
            micro_batch_size=micro_batch_size,
            verbose=verbose,
            max_chars=max_chars,
            autocast=autocast,
        )


class HFMultilingualDomainClassifier(HFClassifierModel):
    """Multilingual domain classifier using nvidia/multilingual-domain-classifier."""

    def __init__(  # noqa: PLR0913
        self,
        id_col: str,
        output_col: str = "domain_pred",
        prob_col: str = "domain_prob",
        keep_cols: list[str] | None = None,
        has_seq_order: bool = True,
        micro_batch_size: int = 256,
        verbose: bool = True,
        max_chars: int = 2000,
        autocast: bool = True,
    ):
        super().__init__(
            id_col=id_col,
            output_col=output_col,
            prob_col=prob_col,
            model_identifier="nvidia/multilingual-domain-classifier",
            keep_cols=keep_cols,
            has_seq_order=has_seq_order,
            micro_batch_size=micro_batch_size,
            verbose=verbose,
            max_chars=max_chars,
            autocast=autocast,
        )
