import torch
import numpy as np
import random
import os
from datetime import datetime

# MUST set environment variables before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['PYTHONHASHSEED'] = '0'

LOG_FILE = "matmul_step_log.txt"


def log_matmul_steps(seed=42, size=1024, dtype=torch.float16, log_file=LOG_FILE):
    """
    Logs the step-by-step calculation of a single element in a matrix multiplication
    on the GPU to observe error accumulation.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = 'cuda'
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Set seed for reproducibility on the same GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Create matrices directly on the GPU
    matrix_a = torch.randn(size, size, device=device, dtype=dtype)
    matrix_b = torch.randn(size, size, device=device, dtype=dtype)

    # We will calculate the element at [0, 0] of the result matrix
    row_idx, col_idx = 0, 0
    row_a = matrix_a[row_idx, :]
    col_b = matrix_b[:, col_idx]

    # Use a tensor on the GPU for accumulation to ensure GPU arithmetic
    accumulator = torch.tensor(0.0, device=device, dtype=dtype)

    print(f"Logging matmul steps to {log_file} using device '{device}' and dtype '{dtype}'")

    with open(log_file, "w") as f:
        f.write(f"# Matmul step log\n")
        f.write(f"# Date: {datetime.now().isoformat()}\n")
        f.write(f"# GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"# Seed: {seed}, Size: {size}, Dtype: {dtype}\n")
        f.write(f"# Calculating C[{row_idx}, {col_idx}] step by step on the GPU\n")
        f.write(f"# step,a,b,product,accumulator\n")

        # Perform the dot product step-by-step on the GPU
        for i in range(size):
            # All operations are now on GPU tensors
            product = row_a[i] * col_b[i]
            accumulator += product

            # Fetch values from GPU to CPU only for logging
            a_val = row_a[i].item()
            b_val = col_b[i].item()
            prod_val = product.item()
            acc_val = accumulator.item()
            f.write(f"{i+1},{a_val},{b_val},{prod_val},{acc_val}\n")

    print(f"Finished logging.")
    final_val = accumulator.item()
    print(f"Final accumulated value for C[{row_idx}, {col_idx}]: {final_val}")

if __name__ == "__main__":
    # Allow specifying the log file from command line for convenience
    import sys
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]
        print(f"Using log file specified from command line: {log_file_path}")
        log_matmul_steps(log_file=log_file_path)
    else:
        log_matmul_steps()
