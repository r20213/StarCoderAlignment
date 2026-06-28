import os
import torch
from dataclasses import dataclass
from typing import Optional, List
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from datasets import load_dataset, Dataset, DatasetDict

load_dotenv(find_dotenv())

# =========================================================
# CONFIG
# =========================================================
@dataclass
class DPODataGenConfig:
    model_path: str = "./muon_sft_model"  # Path to your trained SFT model
    source_dataset: str = "LastTransformer/m-a-p-CodeFeedback-Filtered-Instruction-Splits"
    
    output_dir: str = "./dpo_generated_dataset"
    hub_repo_id: Optional[str] = "LastTransformer/tinystarcoder-dpo-dataset"
    hub_token: Optional[str] = os.getenv("HF_TOKEN")
    
    prompt_field: str = "query"
    response_field: str = "answer"
    
    # Sampling Config
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.92
    repetition_penalty: float = 1.2  # Dynamic repetition penalty factor (> 1.0)
    
    # Subsample bounds for building preference data (None for full dataset)
    train_count: Optional[int] = 5000
    val_count: Optional[int] = 500


# =========================================================
# REPETITION PENALTY LOGITS PROCESSOR
# =========================================================
class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    Enforces a repetition penalty directly onto the sampling logit stream.
    For values > 1.0, tokens already generated will have their logit scores divided (if positive)
    or multiplied (if negative) to decrease their likelihood of being selected again.
    """
    def __init__(self, penalty: float):
        if not (penalty > 0):
            raise ValueError(f"Penalty must be strictly positive, got {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(input_ids.shape[0]):
            for previous_id in set(input_ids[i].tolist()):
                logit = scores[i, previous_id]
                if logit < 0:
                    scores[i, previous_id] = logit * self.penalty
                else:
                    scores[i, previous_id] = logit / self.penalty
        return scores


# =========================================================
# GENERATION ENGINE
# =========================================================
def generate_rejected_responses(model, tokenizer, dataset, cfg: DPODataGenConfig, max_count: Optional[int] = None) -> Dataset:
    generated_records = []
    total_samples = max_count if max_count is not None else len(dataset)
    
    # Register custom logits processors for generation
    logits_processor = LogitsProcessorList()
    if cfg.repetition_penalty != 1.0:
        logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=cfg.repetition_penalty))

    device = next(model.parameters()).device

    for i in tqdm(range(total_samples), desc="Generating rejected responses"):
        query = dataset[i][cfg.prompt_field]
        chosen = dataset[i][cfg.response_field]
        
        # Format matching SFT layout
        inputs = tokenizer(query, return_tensors="pt", add_special_tokens=False).to(device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Isolate only the newly generated text tokens
        generated_tokens = outputs[0][input_length:]
        rejected = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        generated_records.append({
            "query": query,
            "chosen": chosen,
            "rejected": rejected
        })
        
    return Dataset.from_list(generated_records)


# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    cfg = DPODataGenConfig()
    
    print(f"Loading tokenizer & SFT model from {cfg.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float32,  # Match your native FP32 setup
        device_map="auto"
    )
    model.eval()

    print(f"Loading source dataset splits from {cfg.source_dataset}...")
    raw_train = load_dataset(cfg.source_dataset, split="train")
    raw_val = load_dataset(cfg.source_dataset, split="validation")

    # Construct the DPO columns via generation
    print("\n--- Processing Training Split ---")
    dpo_train = generate_rejected_responses(model, tokenizer, raw_train, cfg, max_count=cfg.train_count)
    
    print("\n--- Processing Validation Split ---")
    dpo_val = generate_rejected_responses(model, tokenizer, raw_val, cfg, max_count=cfg.val_count)

    dpo_dataset_dict = DatasetDict({
        "train": dpo_train,
        "validation": dpo_val
    })

    # Save artifact locally
    print(f"\nSaving generated dataset locally to: {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    dpo_dataset_dict.save_to_disk(cfg.output_dir)

    # Push structured data to the Hub
    if cfg.hub_repo_id:
        print(f"Uploading DPO dataset to the HF Hub: {cfg.hub_repo_id}")
        dpo_dataset_dict.push_to_hub(
            repo_id=cfg.hub_repo_id,
            token=cfg.hub_token,
            private=False
        )
        print("Upload complete!")

if __name__ == "__main__":
    main()