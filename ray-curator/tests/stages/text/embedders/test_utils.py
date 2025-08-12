# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf
import cupy as cp

from ray_curator.stages.text.embedders.utils import create_list_series_from_1d_or_2d_ar


class TestCreateListSeriesFrom1dOr2dAr:
    """Test create_list_series_from_1d_or_2d_ar function."""

    def test_embedding_creation_like_base_usage(self):
        """Test the function as used in EmbeddingModelStage.create_output_dataframe."""
        # Simulate embeddings from a model (2D array: batch_size x embedding_dim)
        collected_output = cp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

        # Create a GPU dataframe with index like in the actual usage
        df_gpu = cudf.DataFrame(index=cudf.RangeIndex(3))

        # Test the function call as it appears in base.py
        embedding_series = create_list_series_from_1d_or_2d_ar(collected_output, index=df_gpu.index)

        # Verify the result
        assert isinstance(embedding_series, cudf.Series)
        assert len(embedding_series) == 3
        assert embedding_series.index.equals(df_gpu.index)

        # Convert to pandas to check the actual embedding values
        embedding_series_cpu = embedding_series.to_pandas()
        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        for i, expected in enumerate(expected_embeddings):
            assert embedding_series_cpu.iloc[i] == expected
