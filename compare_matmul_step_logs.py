import sys

def compare_matmul_step_logs(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = [line for line in f1 if not line.startswith('#')]
        lines2 = [line for line in f2 if not line.startswith('#')]

    if len(lines1) != len(lines2):
        print(f"Files have different number of steps: {len(lines1)} vs {len(lines2)}")
        return

    print(f"Comparing {file1} and {file2} step by step:")
    print(f"{'Step':>6} | {'Acc1':>15} | {'Acc2':>15} | {'Diff':>15}")
    print('-'*60)
    for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        parts1 = l1.strip().split(',')
        parts2 = l2.strip().split(',')
        step = int(parts1[0])
        acc1 = float(parts1[-1])
        acc2 = float(parts2[-1])
        diff = acc1 - acc2
        print(f"{step:6d} | {acc1:15.8f} | {acc2:15.8f} | {diff:15.8f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_matmul_step_logs.py <log1> <log2>")
    else:
        compare_matmul_step_logs(sys.argv[1], sys.argv[2])
