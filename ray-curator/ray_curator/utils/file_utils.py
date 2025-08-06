import os
import warnings

import fsspec
import pyarrow.parquet as pq
from fsspec.core import get_filesystem_class, split_protocol
from fsspec.parquet import open_parquet_file
from loguru import logger


def get_fs(path: str, storage_options: dict[str, str] | None = None) -> fsspec.AbstractFileSystem:
    if not storage_options:
        storage_options = {}
    protocol, path = split_protocol(path)
    return get_filesystem_class(protocol)(**storage_options)


def filter_files_by_extension(
    files_list: list[str],
    keep_extensions: str | list[str],
) -> list[str]:
    filtered_files = []
    if isinstance(keep_extensions, str):
        keep_extensions = [keep_extensions]

    # Ensure that the extensions are prefixed with a dot
    file_extensions = tuple([s if s.startswith(".") else f".{s}" for s in keep_extensions])

    for file in files_list:
        if file.endswith(file_extensions):
            filtered_files.append(file)

    if len(files_list) != len(filtered_files):
        warnings.warn("Skipped at least one file due to unmatched file extension(s).", stacklevel=2)

    return filtered_files


def get_all_files_paths_under(
    input_dir: str,
    recurse_subdirectories: bool = False,
    keep_extensions: str | list[str] | None = None,
    storage_options: dict | None = None,
    fs: fsspec.AbstractFileSystem | None = None,
) -> list[str]:
    # TODO: update with a more robust fsspec method
    if fs is None:
        fs = get_fs(input_dir, storage_options)

    file_ls = fs.find(input_dir, maxdepth=None if recurse_subdirectories else 1)
    if "://" in input_dir:
        protocol = input_dir.split("://")[0]
        file_ls = [f"{protocol}://{f}" for f in file_ls]

    file_ls.sort()
    if keep_extensions is not None:
        file_ls = filter_files_by_extension(file_ls, keep_extensions)
    return file_ls


def remove_and_create_dir(
    dir_path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict | None = None
) -> None:
    if fs is None:
        fs = get_fs(dir_path, storage_options)

    if fs.exists(dir_path):
        logger.warning(f"Removing and recreating directory {dir_path}")
        fs.rm(dir_path, recursive=True)
    # if AWS then since mkdir doesn't exist we need to touch
    if fs.protocol == "s3":
        fs.touch(os.path.join(dir_path, ".empty"))
    else:
        fs.mkdir(dir_path)


def get_embedding_dim_from_pq_file(
    file: str,
    storage_options: dict | None = None,
) -> int:
    """
    Get the total embedding dimension by summing lengths of all list[X] columns.
    Assumes all rows in each list column have the same length.
    """
    with fsspec.open(file, "rb", **(storage_options or {})) as f:
        parquet_file = pq.ParquetFile(f)
        schema = parquet_file.schema_arrow

        total_dim = 0
        # Read first row group to get actual lengths
        table = parquet_file.read_row_groups([0], use_threads=False)

        for field in schema:
            # Check if field is a list type
            if hasattr(field.type, "value_type"):
                # Get length from first row
                arr = table[field.name][0]
                if hasattr(arr, "__len__"):
                    total_dim += len(arr)

        if total_dim == 0:
            # If no list columns found, or all list columns are empty, return 0
            return 0

        return total_dim


def get_parquet_num_rows(
    file_path: str,
    storage_options: dict | None = None,
) -> int:
    """Get number of rows for local/cloud Parquet files efficiently by only reading the footer"""
    with open_parquet_file(file_path, storage_options=storage_options) as f:
        return pq.read_metadata(f).num_rows


def split_pq_files_by_max_elements(
    files: list[str],
    embedding_dim: int | None = None,
    max_total_elements: int = 2_000_000_000,
    storage_options: dict | None = None,
) -> list[list[str]]:
    """
    Split files into microbatches such that the total number of elements in the embedding column
    (sum_rows * embedding_dim) in each microbatch does not exceed max_total_elements.
    Uses a simple greedy bin-packing based on cumulative row count.
    """
    # Get num_rows for each file up front
    num_rows_list = [get_parquet_num_rows(file, storage_options) for file in files]
    if embedding_dim is None:
        embedding_dim = get_embedding_dim_from_pq_file(files[0], storage_options)

    microbatches = []
    current_batch = []
    current_rows = 0
    for file, num_rows in zip(files, num_rows_list, strict=False):
        if (current_rows + num_rows) * embedding_dim > max_total_elements and current_batch:
            microbatches.append(current_batch)
            current_batch = []
            current_rows = 0
        current_batch.append(file)
        current_rows += num_rows
    if current_batch:
        microbatches.append(current_batch)
    return microbatches
