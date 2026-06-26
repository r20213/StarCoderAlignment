tiny-starcoder-alignment/
├── config/
│   ├── sft_config.yaml
│   ├── dpo_config.yaml
│   ├── grpo_config.yaml
│   └── p_gspo_config.yaml
│
├── data/
│   ├── raw/                           # Raw downloaded datasets (gitignored)
│   └── processed/                     # Standardized JSONL outputs ready for tokenization
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_utils/                    # DATASET PROCESSING
│   │   ├── __init__.py
│   │   ├── base_parser.py             # Shared abstract parser/regex utilities
│   │   ├── process_jondurbin.py       # Conversational -> Code-only pair extractor
│   │   ├── process_codefeedback.py   # Extraction of clean verification targets
│   │   └── process_openr1.py          # Stripping reasoning steps to match base model
│   │
│   ├── models/                        # ARCHITECTURE & HYBRID LAYERS
│   │   ├── __init__.py
│   │   └── dora.py                    # Custom From-Scratch DoRA Linear Wrapper
│   │
│   ├── engine/                        # CUSTOM OPTIMIZATION LOOPS (FROM SCRATCH)
│   │   ├── __init__.py
│   │   ├── sft_trainer.py             # Standard cross-entropy loop
│   │   ├── dpo_trainer.py             # Contrastive implicit reward loss loop
│   │   ├── grpo_trainer.py            # Token-level group advantage loop
│   │   └── p_gspo_trainer.py          # Sequence-level unified importance ratio loop
│   │
│   └── evaluation/                    # EVALUATION RUNNERS
│       ├── __init__.py
│       ├── sandbox.py                 # Isolated execution environment (Subprocess/Docker)
│       └── humaneval_runner.py        # Evaluates pass@1 on openai_humaneval
│
├── scripts/                           # EXECUTION ENTRY POINTS
│   ├── run_preprocessing.py
│   ├── run_sft.py
│   ├── run_dpo.py
│   ├── run_grpo.py
│   └── run_pgspo.py
│
├── requirements.txt
└── README.md