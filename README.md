docker build -t pytorch-determism .
mkdir results
sudo docker run --rm --gpus all -v $(pwd):/workspace -w /workspace pytorch-determism python3 main.py
sudo docker run --rm --gpus all -v $(pwd):/workspace -w /workspace pytorch-determism python3 compare_results.py results/results_NVIDIA_GeForce_RTX_4090_20250708_124452 results/results_NVIDIA_H100_PCIe_20250708_122219