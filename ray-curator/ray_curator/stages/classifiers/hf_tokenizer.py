import time
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DocumentBatch


def set_torch_to_use_rmm() -> None:
    from rmm.allocators.torch import rmm_torch_allocator

    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)


class HFTokenizer(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(
        self,
        id_col: str,
        text_col: str,
        model_name: str,
        max_seq_length: int | None = None,
        padding_side: Literal["left", "right"] = "right",
        return_text: bool = True,
        sort_by_length: bool = True,
        verbose: bool = False,
    ):
        self._name = "hf_tokenizer"
        self.id_col = id_col
        self.text_col = text_col
        self.model_name = model_name
        self.max_seq_length = max_seq_length

        self.padding_side = padding_side
        self.return_text = return_text
        self.sort_by_length = sort_by_length
        self.verbose = verbose

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.max_seq_length is None:
            self.max_seq_length = self.tokenizer.model_max_length
            # Guard against the HF bug
            # which sets max_seq_length to max(int) for some models
            if self.max_seq_length > 1e5:  # noqa: PLR2004
                self.max_seq_length = AutoConfig.from_pretrained(self.model_name).max_position_embeddings
        if self.verbose:
            logger.debug(
                f"TokenizerActor ({self.model_name}) initialized with max_seq_length={self.max_seq_length} @ {datetime.now()}"  # noqa: DTZ005
            )

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if self.verbose:
            logger.debug(f"Calling TokenizerActor @ {datetime.now()}")  # noqa: DTZ005

        t0 = time.perf_counter()
        df = task.to_pandas()
        tokens = self.tokenizer(
            df[self.text_col].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )
        t2 = time.perf_counter()
        throughput = len(task.data) / (t2 - t0)
        if self.verbose:
            logger.debug(
                f"Tokenized {len(task.data):,} rows in {(t2 - t0):.2f}s @ {throughput:.2f}r/s @ {datetime.now()}"  # noqa: DTZ005
            )

        output = pd.DataFrame(
            {
                self.id_col: df[self.id_col],
                **({"text": df[self.text_col]} if self.return_text else {}),
                "input_ids": tokens.input_ids.tolist(),
                "attention_mask": tokens.attention_mask.tolist(),
            }
        )
        if self.sort_by_length:
            # add column to preserve original order
            output["_curator_seq_order"] = np.arange(len(df))
            output["_curator_token_length"] = tokens.attention_mask.sum(axis=1)
            output = output.sort_values(by="_curator_token_length", kind="stable", ignore_index=True).drop(
                columns=["_curator_token_length"]
            )

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=output,
            _stage_perf=task._stage_perf,
            _metadata=task._metadata,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.id_col, self.text_col]

    def outputs(self) -> tuple[list[str], list[str]]:
        output_cols = [self.id_col, "input_ids", "attention_mask"]
        if self.sort_by_length:
            output_cols.append("_curator_seq_order")
        if self.return_text:
            output_cols.insert(1, self.text_col)
        return ["data"], output_cols

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": True}
