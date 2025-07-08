docker build -t pytorch-determism .
mkdir results
sudo docker run --rm --gpus all -v $(pwd):/workspace -w /workspace pytorch-determism python3 main.py
