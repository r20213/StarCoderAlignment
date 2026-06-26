import logging
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset_builder
from dotenv import load_dotenv,find_dotenv
import os
import random
import duckdb
from typing import Optional, Dict, Any, Generator,Union
from huggingface_hub import HfApi


load_dotenv(find_dotenv())  # Load environment variables from .env file if present
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    Queries Hugging Face dataset via local DuckDB, calculates exact split partitions, 
    and pipes data directly to the Hub via lazy stream generators without local materialization.
    """
    random.seed(seed)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("A valid Hugging Face Write Token must be present to push datasets to the Hub.")

    HfApi().create_repo(repo_id=target_repo_id, token=hf_token, repo_type="dataset", private=private, exist_ok=True)

    # 1. Open a local DuckDB session and throttle network aggressiveness to prevent 429s
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute(f"CREATE OR REPLACE SECRET hf_secret (TYPE huggingface, TOKEN '{hf_token}');")
    
    # --- THE ANTI-429 FIX FOR DUCKDB ---
    # Throttle DuckDB so it doesn't slam Hugging Face with parallel requests
    con.execute("SET threads=2;")       # Reduce from default (substantially lower concurrency)
    con.execute("SET http_retries=10;")                # Force automatic exponential backoff on 429/503 errors
    con.execute("SET http_retry_backoff=2.0;")        # Wait longer between retries
    
    # Target the precise default parquet directory instead of scanning everything via global wildcards
    hf_parquet_url = f"hf://datasets/{path}@~parquet/default/{split_name}/*.parquet"
    
    # Compile constraints matching filter_dict
    where_clauses = []
    if filter_dict:
        for col, val in filter_dict.items():
            if isinstance(val, str):
                where_clauses.append(f"LOWER({col}) LIKE '%{val.lower()}%'")
            else:
                where_clauses.append(f"{col} = {val}")
    where_stmt = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    
    query = f"""
        SELECT *, ROW_NUMBER() OVER() - 1 as __row_index 
        FROM '{hf_parquet_url}' 
        {where_stmt};
    """
    
    logger.info(f"Analyzing and isolating row schemas remotely via DuckDB (throttled mode)...")
    try:
        con.execute(f"CREATE OR REPLACE VIEW filtered_source AS {query}")
        total_len = con.execute("SELECT COUNT(*) FROM filtered_source;").fetchone()[0]
    except Exception as e:
        logger.error(f"DuckDB remote view creation failed: {e}")
        raise e

    logger.info(f"Identified {total_len} matching records. Computing partition allocations...")

    if total_len == 0:
        raise ValueError(f"No rows matched filter criteria {filter_dict} in dataset {path}.")

    # 2. Derive randomized index splits
    subset_size = int(total_len * sample_percentage)
    train_size = int(subset_size * train_ratio)
    remaining_size = subset_size - train_size
    val_size = remaining_size // 2
    
    all_indices = list(range(total_len))
    chosen_indices = random.sample(all_indices, subset_size)
    
    train_set = set(chosen_indices[:train_size])
    val_set = set(chosen_indices[train_size:train_size + val_size])
    test_set = set(chosen_indices[train_size + val_size:])

    # 3. Define the unmaterialized generator factory
    def make_split_generator(target_set: set) -> Generator[Dict[str, Any], None, None]:
        """
        Inner generator function that reads rows individually from the DuckDB view
        and yields them instantly to the streaming uploader process.
        """
        # Open separate connection thread context for generator isolation
        g_con = duckdb.connect()
        g_con.execute("INSTALL httpfs; LOAD httpfs;")
        g_con.execute(f"CREATE OR REPLACE SECRET hf_secret (TYPE huggingface, TOKEN '{hf_token}');")
        
        # Pull rows sequentially via stream cursor
        result_cursor = g_con.execute("SELECT * FROM filtered_source;")
        column_names = [desc[0] for desc in result_cursor.description]
        
        while True:
            row = result_cursor.fetchone()
            if row is None:
                break
            
            row_dict = dict(zip(column_names, row))
            current_idx = row_dict.pop("__row_index") # Strip our synthetic row key
            
            if current_idx in target_set:
                yield row_dict
                
        g_con.close()
    try:
        ds_builder = load_dataset_builder(path, token=hf_token)
        repo_features = ds_builder.info.features
        logger.info(f"Successfully captured dataset schema features.")
    except Exception as e:
        logger.warning(f"Could not automatically resolve remote features schema: {e}. Defaulting to None.")
        repo_features = None
        # 4. Construct lazy Iterable Datasets and stream directly to the Hub
        splits = {"train": train_set, "validation": val_set, "test": test_set}
    
    for split_label, index_target in splits.items():
        logger.info(f"Streaming data channel directly to target repository split: '{split_label}'...")
        
        lazy_iterable = IterableDataset.from_generator(
            make_split_generator, 
            gen_kwargs={"target_set": index_target},
            features=repo_features # 2. Pass the schema here!
        )
        
        lazy_iterable.push_to_hub(
            repo_id=target_repo_id,
            split=split_label,
            token=hf_token,
            private=private
        )
        
    logger.info(f"Pipeline complete! Splits successfully streamed to https://huggingface.co/datasets/{target_repo_id}")
    con.close()