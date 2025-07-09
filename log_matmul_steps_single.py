import torch
import os
from datetime import datetime

LOG_FILE = "matmul_step_log.txt"


def log_matmul_steps(seed=42, size=1024, dtype=torch.float16, log_file=LOG_FILE):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    torch.manual_seed(seed)
    matrix_a = torch.randn(size, size, dtype=dtype)
    matrix_b = torch.randn(size, size, dtype=dtype)

    row_idx, col_idx = 0, 0
    row_a = matrix_a[row_idx, :]
    col_b = matrix_b[:, col_idx]

    accumulator = 0.0
    with open(log_file, "w") as f:
        f.write(f"# Matmul step log\n")
        f.write(f"# Date: {datetime.now().isoformat()}\n")
        f.write(f"# Seed: {seed}, Size: {size}, Dtype: {dtype}\n")
        f.write(f"# Calculating C[{row_idx}, {col_idx}] step by step\n")
        f.write(f"# step, a, b, product, accumulator\n")
        for i in range(size):
            a = float(row_a[i].item())
            b = float(col_b[i].item())
            product = a * b
            accumulator += product
            f.write(f"{i+1},{a},{b},{product},{accumulator}\n")
    print(f"Matmul steps logged to {log_file}")

if __name__ == "__main__":
    log_matmul_steps()
