import os
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import find_dotenv, load_dotenv
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from muon import MuonWithAuxAdam
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
# =========================================================
# CONFIG
# =========================================================
@dataclass
class TrainConfig:
    model_id: str = "bigcode/tiny_starcoder_py"
    output_dir: str = "./muon_sft_model"

    max_seq_len: int = 4096
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 3

    adamw_base_lr: float = 2e-5
    muon_base_lr: float = 0.02 # Use the industry standard lr for Muon.
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0

    warmup_steps: int = 200
    min_lr_ratio: float = 0.1

    num_workers: int = 4
    seed: int = 42

    wandb_project: str = "tinystarcoder-muon-sft-ddp"

    hub_repo_id: Optional[str] = "LastTransformer/tinystarcoder-muon-sft-ddp"
    hub_private_repo: bool = False
    hub_token: Optional[str] = os.getenv("HF_TOKEN")

    eval_at_epoch_end: bool = True
    log_every_optimizer_step: int = 1

    bf16: bool = False  # set True if your GPUs support bf16
    fp16: bool = True   # default AMP mode
    train_dataset_hub_id: Optional[str] = "LastTransformer/m-a-p-CodeFeedback-Filtered-Instruction-Splits"  # if you have a dataset on the hub, specify it here
    train_dataset_split: str = "train"
    val_dataset_hub_id: Optional[str] = "LastTransformer/m-a-p-CodeFeedback-Filtered-Instruction-Splits"  # if you have a dataset on the hub, specify it here
    val_dataset_split: str = "validation"
    train_dataset_prompt_field: str = "query"
    train_dataset_response_field: str = "answer"
    val_dataset_prompt_field: str = "query"
    val_dataset_response_field: str = "answer"



# =========================================================
# REPRO
# =========================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 1. PACKED DATASET
# =========================================================
class PackedSFTDataset(Dataset):
    """
    Expects already-tokenized examples:
    [
        {"input_ids": [...], "labels": [...]},
        ...
    ]

    labels should already contain -100 where prompt tokens must be masked.
    This class packs multiple examples into constant-length sequences without
    allowing loss to spill across padding/gaps because gaps are filled with
    pad_token_id in input_ids and -100 in labels.
    """
    def __init__(
        self,
        tokenized_examples: List[Dict[str, List[int]]],
        max_seq_len: int,
        pad_token_id: int,
    ):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.samples: List[Dict[str, torch.Tensor]] = []

        cur_input_ids: List[int] = []
        cur_labels: List[int] = []

        for ex in tokenized_examples:
            ex_input_ids = ex["input_ids"]
            ex_labels = ex["labels"]

            if len(ex_input_ids) != len(ex_labels):
                raise ValueError("input_ids and labels must have same length")

            start = 0
            while start < len(ex_input_ids):
                remaining = self.max_seq_len - len(cur_input_ids)
                take = min(remaining, len(ex_input_ids) - start)

                cur_input_ids.extend(ex_input_ids[start:start + take])
                cur_labels.extend(ex_labels[start:start + take])
                start += take

                if len(cur_input_ids) == self.max_seq_len:
                    self.samples.append({
                        "input_ids": torch.tensor(cur_input_ids, dtype=torch.long),
                        "labels": torch.tensor(cur_labels, dtype=torch.long),
                        "attention_mask": torch.ones(self.max_seq_len, dtype=torch.long),
                    })
                    cur_input_ids, cur_labels = [], []

        if len(cur_input_ids) > 0:
            pad_len = self.max_seq_len - len(cur_input_ids)
            self.samples.append({
                "input_ids": torch.tensor(
                    cur_input_ids + [self.pad_token_id] * pad_len,
                    dtype=torch.long
                ),
                "labels": torch.tensor(
                    cur_labels + [-100] * pad_len,
                    dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    [1] * len(cur_input_ids) + [0] * pad_len,
                    dtype=torch.long
                ),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =========================================================
# 2. DDP SETUP
# =========================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# =========================================================
# 3. SFT LOSS
# =========================================================
def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Standard next-token causal LM loss with masking.
    labels should contain -100 for tokens excluded from loss.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )


# =========================================================
# 4. PARAM GROUPING FOR MUON
# =========================================================

def build_muon_param_groups(model: torch.nn.Module):
    """
    Groups parameters for Muon and an auxiliary optimizer (like AdamW).
    
    Targeted Muon parameters must be exactly 2D and exclude structural 
    embeddings (transformer.wte.weight, transformer.wpe.weight). 
    """
    muon_params = []
    aux_adam_params = []

    # Exact names of 2D parameters to exclude from Muon optimization
    excluded_names = {
        "transformer.wte.weight",
        "transformer.wpe.weight"
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Check if the parameter satisfies the 2D requirement and isn't an embedding
        if p.ndim == 2 and name not in excluded_names:
            muon_params.append(p)
        else:
            aux_adam_params.append(p)

    return muon_params, aux_adam_params


# =========================================================
# 5. LR SCHEDULE
# =========================================================
def get_cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr_ratio: float):
    if total_steps <= 0:
        return base_lr

    min_lr = base_lr * min_lr_ratio

    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lrs(optimizer, lr: float, cfg : TrainConfig):
    """
    Safely adjusts the learning rates over time.
    'lr' coming from the scheduler acts as Muon's active learning rate.
    """
    for group in optimizer.param_groups:
        if group.get("use_muon", False):
            # Muon parameters are directly scaled by the scheduler's 'lr'
            group["lr"] = lr
            
            # Dynamically scale the nested/auxiliary AdamW parameters
            if "adamw_lr" in group and "adamw_lr_ratio" in group:
                group["adamw_lr"] = lr * group["adamw_lr_ratio"]
        else:
            # Standalone AdamW parameters (biases, layernorms)
            # Since the incoming 'lr' is scaled for Muon (~0.02), we need to step 
            # down this group to AdamW levels using the approximate ratio (~1/1000x)
            # If your custom optimizer saves 'adamw_lr_ratio' globally, use that here too!
            muon_to_adamw_ratio = cfg.adamw_base_lr / cfg.muon_base_lr
            group["lr"] = lr * muon_to_adamw_ratio


# =========================================================
# 6. METRIC HELPERS
# =========================================================
@torch.no_grad()
def reduce_mean(value: torch.Tensor, world_size: int) -> float:
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value = value / world_size
    return value.item()


@torch.no_grad()
def evaluate(model, loader, local_rank, world_size, use_amp: bool, amp_dtype: torch.dtype):
    model.eval()
    loss_sum = 0.0
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
        labels = batch["labels"].to(local_rank, non_blocking=True)
        attention_mask = batch["attention_mask"].to(local_rank, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            loss = sft_loss(outputs.logits, labels)

        loss_sum += reduce_mean(loss, world_size)
        num_batches += 1

    return loss_sum / max(1, num_batches)


# =========================================================
# 7. EXAMPLE DATA LOADING
# =========================================================
def build_sft_examples(tokenizer, dataset, cfg, count: int | None = None) -> List[Dict[str, Any]]:
    """
    Replace this with your real preprocessing.
    Each example returns input_ids and labels where prompt tokens are masked with -100.
    """
    # TODO : Replace this with your real preprocessing logic to load and tokenize your SFT dataset.
    examples = []
    total_count = count if count is not None else len(dataset)
    eos = tokenizer.eos_token_id
    for i in range(total_count):
        prompt = dataset[i][cfg.dataset_prompt_field]
        response = dataset[i][cfg.dataset_response_field]
        if len(response_ids) == 0 or response_ids[-1] != eos:
            response_ids = response_ids + [eos]

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids
        labels = ([-100] * len(prompt_ids)) + response_ids

        examples.append({"input_ids": input_ids, "labels": labels})
    return examples


# =========================================================
# 8. MAIN
# =========================================================
def main():
    cfg = TrainConfig()

    local_rank, global_rank, world_size = setup_ddp()
    is_main = global_rank == 0

    set_seed(cfg.seed + global_rank)

    if cfg.bf16 and cfg.fp16:
        raise ValueError("Choose only one of bf16 or fp16")

    use_amp = cfg.bf16 or cfg.fp16
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    if is_main:
        wandb.init(
            project=cfg.wandb_project,
            config=vars(cfg)
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=amp_dtype if use_amp else torch.float32,
        attn_implementation="sdpa"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model = model.to(local_rank)

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )

    # -------------------------------------------------
    # Load train/val once at startup
    # Replace these two calls with your real tokenized splits
    # -------------------------------------------------
    train_dataset = load_dataset(cfg.train_dataset_hub_id, split=cfg.train_dataset_split) if cfg.train_dataset_hub_id else None
    val_dataset = load_dataset(cfg.val_dataset_hub_id, split=cfg.val_dataset_split) if cfg.val_dataset_hub_id else None
    train_examples = build_sft_examples(tokenizer, train_dataset, cfg, count=None)
    val_examples = build_sft_examples(tokenizer, val_dataset, cfg, count=None)

    train_dataset = PackedSFTDataset(
        tokenized_examples=train_examples,
        max_seq_len=cfg.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    val_dataset = PackedSFTDataset(
        tokenized_examples=val_examples,
        max_seq_len=cfg.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )

    if is_main:
        logger.info(f"packed train samples: {len(train_dataset)}")
        logger.info(f"packed val samples:   {len(val_dataset)}")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=cfg.seed,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.per_device_eval_batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    muon_params, aux_adam_params = build_muon_param_groups(ddp_model.module)

    optimizer = MuonWithAuxAdam(
    [
        {
            "params": muon_params, 
            "use_muon": True, 
            "adamw_lr_ratio": cfg.adamw_base_lr / cfg.muon_base_lr
        },
        {
            "params": aux_adam_params, 
            "use_muon": False
        },
    ],
    lr=cfg.muon_base_lr,
    adamw_lr=cfg.adamw_base_lr,
    weight_decay=cfg.weight_decay,
    betas=cfg.betas,
)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_optim_steps = steps_per_epoch * cfg.epochs
    optim_step = 0

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()

        running_loss = 0.0
        running_microbatches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
            labels = batch["labels"].to(local_rank, non_blocking=True)
            attention_mask = batch["attention_mask"].to(local_rank, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = ddp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                loss = sft_loss(outputs.logits, labels)
                raw_loss = loss.detach()
                loss = loss / cfg.grad_accum_steps

            if cfg.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += reduce_mean(raw_loss, world_size)
            running_microbatches += 1

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                lr = get_cosine_warmup_lr(
                    step=optim_step,
                    total_steps=total_optim_steps,
                    warmup_steps=cfg.warmup_steps,
                    base_lr=cfg.muon_base_lr,
                    min_lr_ratio=cfg.min_lr_ratio,
                )
                set_optimizer_lrs(optimizer, lr, cfg)

                if cfg.fp16:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.max_grad_norm)

                if cfg.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                optim_step += 1

                train_loss = running_loss / max(1, running_microbatches)
                running_loss = 0.0
                running_microbatches = 0

                if is_main and optim_step % cfg.log_every_optimizer_step == 0:
                    adamw_current_lr = lr * (cfg.adamw_base_lr / cfg.muon_base_lr)
                    logger.info(
                        f"epoch={epoch} step={optim_step}/{total_optim_steps} "
                        f"muon_lr={lr:.6f} adamw_lr={adamw_current_lr:.8f} train_loss={train_loss:.4f}"
                    )
                    wandb.log({
                        "train/loss": train_loss,
                        "train/muon_lr": lr,
                        "train/adamw_lr": adamw_current_lr,
                        "epoch": epoch,
                        "step": optim_step,
                    })

        if cfg.eval_at_epoch_end:
            val_loss = evaluate(
                model=ddp_model,
                loader=val_loader,
                local_rank=local_rank,
                world_size=world_size,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            if is_main:
                # Calculate the exact current state of AdamW learning rate for the eval step log
                adamw_current_lr = lr * (cfg.adamw_base_lr / cfg.muon_base_lr)
                
                logger.info(
                    f"epoch={epoch} step={optim_step}/{total_optim_steps} val_loss={val_loss:.4f}"
                )
                wandb.log({
                    "val/loss": val_loss,
                    "val/muon_lr": lr,           # Added for clean tracking tracking
                    "val/adamw_lr": adamw_current_lr, # Added for clean tracking tracking
                    "epoch": epoch,
                    "step": optim_step,
                })

    if is_main:
        logger.info("saving model...")
        os.makedirs(cfg.output_dir, exist_ok=True)
        ddp_model.module.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        if cfg.hub_repo_id:
            logger.info(f"pushing model to hub: {cfg.hub_repo_id}")
            ddp_model.module.push_to_hub(
                cfg.hub_repo_id,
                private=cfg.hub_private_repo,
                token=cfg.hub_token,
            )
            tokenizer.push_to_hub(
                cfg.hub_repo_id,
                private=cfg.hub_private_repo,
                token=cfg.hub_token,
            )

        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()