import torch
from transformers import AutoModelForCausalLM
import argparse

def count_model_params(checkpoint):
  print(f"Loading {checkpoint}...")
  model = AutoModelForCausalLM.from_pretrained(checkpoint)

  # Fetch all named parameters
  named_params = [(name, p.ndim) for name, p in model.named_parameters()]
  all_params_count = len(list(model.parameters()))

  # Verification check
  print("---")
  print(f"Total parameters found via .parameters(): {all_params_count}")
  print(f"Total named parameters found via .named_parameters(): {len(named_params)}")

  if all_params_count == len(named_params):
      print("✅ Success: All model parameters are named. No tensors were left out.")
  else:
      print("❌ Warning: Mismatch detected between total parameters and named parameters!")
  print("---")

  # Print the parameter names and their dimensions for your review
  print("List of Named Parameters:")
  for name, ndim in named_params:
      print(f"  - {name} (ndim: {ndim})")
  return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect model parameters.")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint.")
    args = parser.parse_args()
    count_model_params(args.checkpoint)