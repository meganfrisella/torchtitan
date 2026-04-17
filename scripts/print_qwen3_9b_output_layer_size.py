#!/usr/bin/env python3
import torch

# From torchtitan/models/qwen3/__init__.py flavor "9B-A3B"
VOCAB_SIZE = 151_936
HIDDEN_DIM = 2_048
DTYPE = torch.bfloat16


def main() -> None:
    output_weight = torch.empty((VOCAB_SIZE, HIDDEN_DIM), dtype=DTYPE)
    bytes_per_elem = torch.tensor([], dtype=DTYPE).element_size()
    total_bytes = output_weight.numel() * bytes_per_elem

    print("model=qwen3_9b")
    print(f"dtype={DTYPE}")
    print(f"output_weight_shape={tuple(output_weight.shape)}")
    print(f"output_features={output_weight.shape[0]}")
    print(f"input_features={output_weight.shape[1]}")
    print(f"output_weight_numel={output_weight.numel()}")
    print(f"output_weight_bytes={total_bytes}")
    print(f"output_weight_size_bytes={total_bytes}")
    print(f"output_weight_mib={total_bytes / 1024**2:.2f}")


if __name__ == "__main__":
    main()
