import logging
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset_builder, SplitDict, SplitInfo
from dotenv import load_dotenv,find_dotenv
import os
import random
import duckdb
from typing import Optional, Dict, Any, Generator,Union
from huggingface_hub import HfApi


load_dotenv(find_dotenv())  # Load environment variables from .env file if present
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Monkey patch IterableDataset to support a user-provided length
def _iterable_dataset_len(self):
    if hasattr(self, "_known_length"):
        return self._known_length
    raise TypeError("object of type 'IterableDataset' has no len()")

# Only patch once
if not hasattr(IterableDataset, "__len__"):
    IterableDataset.__len__ = _iterable_dataset_len

def row_filter_logic(example, filter_dict: Dict[str, Any]) -> bool:
    for col, target_val in filter_dict.items():
        if col not in example:
            logger.warning(f"Column '{col}' not found in dataset example. Skipping this filter.")
            return False
        
        # Smart matching: case-insensitive substring check if both are strings
        if isinstance(target_val, str) and isinstance(example[col], str):
            if target_val.lower() not in example[col].lower():
                return False
        else:
            # Fallback to strict match for non-strings (ints, bools, etc.)
            if example[col] != target_val:
                return False
    return True

def load_hf_dataset(
    path: str,
    split: Optional[str] = None,
    streaming: bool = False,
    token: Optional[str] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """
    Loads a dataset from the Hugging Face Hub with support for streaming and standard mode,
    and dynamically filters rows based on column key-value pairs.
    
    Args:
        path (str): Path or name of the dataset on HF Hub (e.g., 'm-a-p/CodeFeedback-Filtered-Instruction').
        split (str, optional): Which split to load (e.g., 'train', 'test').
        streaming (bool): If True, streams data iteratively instead of downloading completely.
        token (str, optional): HF token for private or gated repositories.
        filter_dict (dict, optional): Dict of {'column_name': 'target_value'} to filter rows.
                                      Performs a case-insensitive substring match for strings.
        **kwargs: Additional parameters passed directly to load_dataset.
        
    Returns:
        Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]: The loaded and filtered dataset.
    """
    mode_str = "STREAMING (Lazy Loading)" if streaming else "STANDARD (Full Download)"
    logger.info(f"Initiating dataset retrieval for: '{path}' via {mode_str} mode.")
    
    try:
        # Load flat data structure
        dataset = load_dataset(
            path=path,
            split=split,
            streaming=streaming,
            token=token,
            **kwargs
        )
        
        # Apply programmatic row filtering if requested
        if filter_dict:
            logger.info(f"Applying in-memory column filters: {filter_dict}")

            # Check if dataset is a mapping dictionary of splits or a single standalone dataset split
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                # Regenerate the dict layout applying filter lazily/eagerly across splits
                dataset = type(dataset)({
                    split_key: split_obj.filter(lambda example: row_filter_logic(example, filter_dict))
                    for split_key, split_obj in dataset.items()
                })
            else:
                dataset = dataset.filter(lambda example: row_filter_logic(example, filter_dict))
        
        # Telemetry logging
        if streaming:
            logger.info(f"Successfully connected to stream interface: {type(dataset)}")
        else:
            logger.info(f"Retrieval complete. Final structure layout:\n{dataset}")
            
        return dataset

    except Exception as e:
        logger.error(f"Failed to process dataset pipeline for '{path}'. Details: {e}")
        raise e


def _hf_stream_generator(
    path: str, 
    split_name: str, 
    filter_dict: Optional[Dict[str, Any]], 
    target_set: set, 
    token: str
) -> Generator[Dict[str, Any], None, None]:
    """
    Leverages the custom load_hf_dataset utility to stream pre-filtered rows,
    yielding only the indices allocated to this specific split partition.
    """
    # Reuse your custom utility function directly!
    filtered_stream = load_hf_dataset(
        path=path, split=split_name, streaming=True, token=token, filter_dict=filter_dict
    )
    
    # Track the matching rows as they stream through
    for idx, row in enumerate(filtered_stream):
        if idx in target_set:
            yield row


def stream_filtered_splits_to_hub(
    path: str,
    target_repo_id: str,
    split_name: str = "train",
    filter_dict: Optional[Dict[str, Any]] = None,
    sample_percentage: float = 0.20,
    train_ratio: float = 0.80,
    seed: int = 42,
    private: bool = True
) -> None:
    """
    Uses DuckDB solely for a lightweight remote COUNT(*) query, then pipes 
    unmaterialized data splits to the Hub using your custom load_hf_dataset stream.
    """
    random.seed(seed)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("A valid Hugging Face Write Token must be present to push datasets to the Hub.")

    HfApi().create_repo(repo_id=target_repo_id, token=hf_token, repo_type="dataset", private=private, exist_ok=True)

    # 1. Quick DuckDB execution strictly to grab the filtered total length
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"CREATE OR REPLACE SECRET hf_secret (TYPE huggingface, TOKEN '{hf_token}');")
    con.execute("SET threads=2; SET http_retries=10;")
    
    hf_parquet_url = f"hf://datasets/{path}@~parquet/default/{split_name}/*.parquet"
    
    where_clauses = []
    if filter_dict:
        for col, val in filter_dict.items():
            where_clauses.append(f"LOWER({col}) LIKE '%{val.lower()}%'" if isinstance(val, str) else f"{col} = {val}")
    where_stmt = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    
    logger.info("Querying remote dataset length via DuckDB...")
    try:
        total_len = con.execute(f"SELECT COUNT(*) FROM '{hf_parquet_url}' {where_stmt};").fetchone()[0]
    finally:
        con.close() # Instantly close connection

    if total_len == 0:
        raise ValueError(f"No rows matched filter criteria {filter_dict} in dataset {path}.")

    # 2. Derive randomized index splits
    subset_size = int(total_len * sample_percentage)
    train_size = int(subset_size * train_ratio)
    val_size = (subset_size - train_size) // 2
    
    chosen_indices = random.sample(range(total_len), subset_size)
    splits = {
        "train": set(chosen_indices[:train_size]),
        "validation": set(chosen_indices[train_size:train_size + val_size]),
        "test": set(chosen_indices[train_size + val_size:])
    }

    # Extract schema features to preserve target formats
    try:
        repo_features = load_dataset_builder(path, token=hf_token).info.features
    except Exception:
        repo_features = None

    # 3. Stream each split channel straight to the Hub
    for split_label, index_target in splits.items():
        if len(index_target) == 0:
            continue
            
        logger.info(f"Streaming data channel directly to target repository split: '{split_label}'...")
        
        lazy_dataset = IterableDataset.from_generator(
            _hf_stream_generator, 
            gen_kwargs={
                "path": path,
                "split_name": split_name,
                "filter_dict": filter_dict,
                "target_set": index_target,
                "token": hf_token
            },
            features=repo_features
        )
        lazy_dataset._known_length = len(index_target)
        lazy_dataset.push_to_hub(repo_id=target_repo_id, split=split_label, token=hf_token, private=private)
        
    logger.info(f"Pipeline complete! Splits successfully streamed to https://huggingface.co/datasets/{target_repo_id}")