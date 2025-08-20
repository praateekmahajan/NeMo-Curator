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

import os
import tempfile

import pandas as pd

from ray_curator.stages.deduplication.semantic.identify_duplicates import IdentifyDuplicatesStage
from ray_curator.tasks import FileGroupTask


class TestIdentifyDuplicatesStage:
    """Test cases for IdentifyDuplicatesStage."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "identify_duplicates_output")
        os.makedirs(self.output_dir, exist_ok=True)

    def teardown_method(self) -> None:
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_test_similarity_file(self, file_path: str, data: dict) -> None:
        """Create a test similarity file with the given data."""
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)

    def test_identify_duplicates_stage_basic(self) -> None:
        """Test basic functionality of IdentifyDuplicatesStage."""
        # Create test data with varying similarity scores
        input_file = os.path.join(self.temp_dir, "cluster_0.parquet")
        test_data = {
            "id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "max_id": ["doc2", "doc1", "doc4", "doc3", "doc1"],
            "cosine_sim_score": [0.95, 0.95, 0.85, 0.85, 0.75],  # Some above/below threshold
        }
        self.create_test_similarity_file(input_file, test_data)

        # Create stage with eps=0.1 (threshold = 0.9)
        stage = IdentifyDuplicatesStage(
            output_path=self.output_dir,
            eps=0.1,  # threshold will be 1.0 - 0.1 = 0.9
            id_field="id",
            verbose=True,
        )

        # Create task
        task = FileGroupTask(
            task_id="test_identify_duplicates",
            dataset_name="test",
            data=[input_file],
            _metadata={"centroid_id": 0, "filetype": "parquet"},
        )

        # Process task
        result_task = stage.process_batch([task])

        # Verify output
        assert len(result_task) == 1
        assert len(result_task[0].data) == 1
        output_file = result_task[0].data[0]
        assert os.path.exists(output_file)

        # Read and verify results
        result_df = pd.read_parquet(output_file)
        # Should only contain docs with similarity >= 0.9 (doc1 and doc2)
        assert len(result_df) == 2
        assert set(result_df["id"].tolist()) == {"doc1", "doc2"}

    def test_identify_duplicates_stage_empty_input(self) -> None:
        """Test handling of empty input."""
        stage = IdentifyDuplicatesStage(output_path=self.output_dir, eps=0.1, id_field="id", verbose=True)

        result_tasks = stage.process_batch([])
        assert len(result_tasks) == 0

    def test_identify_duplicates_stage_single_item(self) -> None:
        """Test handling of single item clusters."""
        input_file = os.path.join(self.temp_dir, "cluster_single.parquet")
        test_data = {
            "id": ["doc1"],
            "max_id": ["doc1"],
            "cosine_sim_score": [0.0],  # Self-similarity is 0 for single items
        }
        self.create_test_similarity_file(input_file, test_data)

        stage = IdentifyDuplicatesStage(output_path=self.output_dir, eps=0.1, id_field="id", verbose=True)

        task = FileGroupTask(
            task_id="test_single",
            dataset_name="test",
            data=[input_file],
            _metadata={"centroid_id": 1, "filetype": "parquet"},
        )

        result_tasks = stage.process_batch([task])

        # Should create empty result for single items
        assert len(result_tasks) == 1
        result_df = pd.read_parquet(result_tasks[0].data[0])
        assert len(result_df) == 0

    def test_identify_duplicates_stage_no_similar_items(self) -> None:
        """Test case where no items meet the similarity threshold."""
        input_file = os.path.join(self.temp_dir, "cluster_no_similar.parquet")
        test_data = {
            "id": ["doc1", "doc2", "doc3"],
            "max_id": ["doc2", "doc3", "doc1"],
            "cosine_sim_score": [0.5, 0.6, 0.7],  # All below threshold
        }
        self.create_test_similarity_file(input_file, test_data)

        stage = IdentifyDuplicatesStage(
            output_path=self.output_dir,
            eps=0.1,  # threshold = 0.9
            id_field="id",
            verbose=True,
        )

        task = FileGroupTask(
            task_id="test_no_similar",
            dataset_name="test",
            data=[input_file],
            _metadata={"centroid_id": 2, "filetype": "parquet"},
        )

        result_tasks = stage.process_batch([task])

        # Should create empty result
        assert len(result_tasks) == 1
        result_df = pd.read_parquet(result_tasks[0].data[0])
        assert len(result_df) == 0

    def test_identify_duplicates_stage_different_eps(self) -> None:
        """Test different epsilon values."""
        input_file = os.path.join(self.temp_dir, "cluster_eps_test.parquet")
        test_data = {
            "id": ["doc1", "doc2", "doc3", "doc4"],
            "max_id": ["doc2", "doc1", "doc4", "doc3"],
            "cosine_sim_score": [0.98, 0.98, 0.85, 0.85],
        }
        self.create_test_similarity_file(input_file, test_data)

        # Test with eps=0.01 (threshold = 0.99) - only very similar items
        stage_strict = IdentifyDuplicatesStage(output_path=self.output_dir, eps=0.01, id_field="id", verbose=True)

        task = FileGroupTask(
            task_id="test_eps_strict",
            dataset_name="test",
            data=[input_file],
            _metadata={"centroid_id": 3, "filetype": "parquet"},
        )

        result_tasks = stage_strict.process_batch([task])
        result_df = pd.read_parquet(result_tasks[0].data[0])
        assert len(result_df) == 0  # No items meet 0.99 threshold

        # Test with eps=0.2 (threshold = 0.8) - more permissive
        stage_permissive = IdentifyDuplicatesStage(output_path=self.output_dir, eps=0.2, id_field="id", verbose=True)

        result_tasks = stage_permissive.process_batch([task])
        result_df = pd.read_parquet(result_tasks[0].data[0])
        assert len(result_df) == 4  # All items meet 0.8 threshold

    def test_identify_duplicates_stage_single_task(self) -> None:
        """Test the IdentifyDuplicatesStage with single task processing."""
        input_file = os.path.join(self.temp_dir, "cluster_single.parquet")
        test_data = {"id": ["doc1", "doc2"], "max_id": ["doc2", "doc1"], "cosine_sim_score": [0.95, 0.95]}
        self.create_test_similarity_file(input_file, test_data)

        stage = IdentifyDuplicatesStage(output_path=self.output_dir, eps=0.1, id_field="id")

        task = FileGroupTask(
            task_id="test_single",
            dataset_name="test",
            data=[input_file],
            _metadata={"centroid_id": 4, "filetype": "parquet"},
        )

        result_tasks = stage.process_batch([task])
        result_df = pd.read_parquet(result_tasks[0].data[0])
        assert len(result_df) == 2  # Both items should be identified as duplicates

    def test_identify_duplicates_stage_batch_processing(self) -> None:
        """Test batch processing of multiple clusters."""
        # Create test data for multiple clusters
        cluster_files = []
        for cluster_id in range(3):
            input_file = os.path.join(self.temp_dir, f"cluster_{cluster_id}.parquet")
            test_data = {
                "id": [f"doc{cluster_id}_1", f"doc{cluster_id}_2", f"doc{cluster_id}_3"],
                "max_id": [f"doc{cluster_id}_2", f"doc{cluster_id}_1", f"doc{cluster_id}_1"],
                "cosine_sim_score": [0.95, 0.95, 0.85],  # First two above threshold
            }
            self.create_test_similarity_file(input_file, test_data)
            cluster_files.append(input_file)

        stage = IdentifyDuplicatesStage(
            output_path=self.output_dir,
            eps=0.1,  # threshold = 0.9
            id_field="id",
            verbose=True,
        )

        # Create tasks for each cluster
        tasks = []
        for i, file_path in enumerate(cluster_files):
            task = FileGroupTask(
                task_id=f"test_batch_{i}",
                dataset_name="test",
                data=[file_path],
                _metadata={"centroid_id": i, "filetype": "parquet"},
            )
            tasks.append(task)

        # Process batch
        result_tasks = stage.process_batch(tasks)
        assert len(result_tasks) == 1  # Should combine into one output task

        # Check combined results
        result_df = pd.read_parquet(result_tasks[0].data[0])
        # Should have 6 items total (2 per cluster with similarity >= 0.9)
        assert len(result_df) == 6

        # Check metadata
        assert "num_removed" in result_tasks[0]._metadata
        assert result_tasks[0]._metadata["num_removed"] == 6

    def test_identify_duplicates_stage_id_sorting(self) -> None:
        """Test ID-based sorting with numeric IDs."""
        input_file = os.path.join(self.temp_dir, "cluster_sorting.parquet")
        test_data = {
            "id": [300, 50, 800, 150, 1200, 10],
            "max_id": [50, 300, 1200, 800, 150, 10],
            "cosine_sim_score": [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],  # All above threshold
        }
        self.create_test_similarity_file(input_file, test_data)

        stage = IdentifyDuplicatesStage(output_path=self.output_dir, eps=0.1, id_field="id", verbose=True)

        task = FileGroupTask(
            task_id="test_sorting",
            dataset_name="test",
            data=[input_file],
            _metadata={"centroid_id": 5, "filetype": "parquet"},
        )

        result_tasks = stage.process_batch([task])
        result_df = pd.read_parquet(result_tasks[0].data[0])

        # Should have all 6 items, sorted by ID
        assert len(result_df) == 6
        # Check that data is sorted
        assert result_df["id"].tolist() == sorted(result_df["id"].tolist())

    def test_identify_duplicates_stage_string_ids_no_sorting(self) -> None:
        """Test that string IDs get sorted properly."""
        input_file = os.path.join(self.temp_dir, "cluster_string_ids.parquet")
        test_data = {
            "id": ["doc_c", "doc_a", "doc_b"],
            "max_id": ["doc_a", "doc_c", "doc_b"],
            "cosine_sim_score": [0.95, 0.95, 0.85],
        }
        self.create_test_similarity_file(input_file, test_data)

        stage = IdentifyDuplicatesStage(output_path=self.output_dir, eps=0.1, id_field="id", verbose=True)

        task = FileGroupTask(
            task_id="test_string_ids",
            dataset_name="test",
            data=[input_file],
            _metadata={"centroid_id": 6, "filetype": "parquet"},
        )

        result_tasks = stage.process_batch([task])
        result_df = pd.read_parquet(result_tasks[0].data[0])

        # Should have 2 items (above threshold), sorted by ID
        assert len(result_df) == 2
        assert result_df["id"].tolist() == sorted(result_df["id"].tolist())

    def test_identify_duplicates_stage_empty_batch(self) -> None:
        """Test handling of empty batch."""
        stage = IdentifyDuplicatesStage(output_path=self.output_dir, eps=0.1, id_field="id")

        result_tasks = stage.process_batch([])
        assert len(result_tasks) == 0

    def test_identify_duplicates_stage_no_id_field(self) -> None:
        """Test functionality without id_field specified."""
        input_file = os.path.join(self.temp_dir, "cluster_no_id_field.parquet")
        test_data = {
            "doc_id": ["doc1", "doc2", "doc3"],
            "max_id": ["doc2", "doc1", "doc1"],
            "cosine_sim_score": [0.95, 0.95, 0.85],
        }
        self.create_test_similarity_file(input_file, test_data)

        stage = IdentifyDuplicatesStage(
            output_path=self.output_dir,
            eps=0.1,
            id_field=None,  # No sorting will be performed
            verbose=True,
        )

        task = FileGroupTask(
            task_id="test_no_id_field",
            dataset_name="test",
            data=[input_file],
            _metadata={"centroid_id": 7, "filetype": "parquet"},
        )

        result_tasks = stage.process_batch([task])
        result_df = pd.read_parquet(result_tasks[0].data[0])

        # Should have 2 items (above threshold), no guaranteed sorting
        assert len(result_df) == 2
