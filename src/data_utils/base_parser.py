import logging
from typing import Optional, Union, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from dotenv import load_dotenv,find_dotenv

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