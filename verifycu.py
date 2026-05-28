import numpy as np
import time
import csv
import subprocess
import matplotlib.pyplot as plt
import os
import collections

def save_matrix(filename, matrix):
    n = matrix.shape[0]
    with open(filename, "w") as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

def load_matrix(filename):
    with open(filename) as f:
        n = int(f.readline())
        data = [list(map(float, f.readline().split())) for _ in range(n)]
    return np.array(data)

def generate_matrix(n):
    return np.random.randint(-500, 500, (n, n)).astype(np.float32)

def compile_cuda():
    cmd = ["nvcc", "-O3", "main_cuda.cu", "-o", "cuda_mm.exe" if os.name == "nt" else "cuda_mm"]
    subprocess.run(cmd, check=True, capture_output=False)

def run_cuda(block_x, block_y):
    exe = "cuda_mm.exe" if os.name == "nt" else "./cuda_mm"
    start = time.time()
    subprocess.run([exe, str(block_x), str(block_y)], check=True, capture_output=False)
    return time.time() - start

def verify():
    A = load_matrix("matrix_a.txt")
    B = load_matrix("matrix_b.txt")
    C_cuda = load_matrix("result.txt")
    C_py = A @ B
    return np.allclose(C_cuda, C_py, atol=1e-3)

def run_experiments(sizes, block_configs):
    results = []
    compile_cuda()

    for n in sizes:
        print(f"\nГенерация матриц размера {n}...")
        A = generate_matrix(n)
        B = generate_matrix(n)
        save_matrix("matrix_a.txt", A)
        save_matrix("matrix_b.txt", B)

        for bx, by in block_configs:
            print(f"Запуск с блоком {bx}x{by}...")
            t = run_cuda(bx, by)
            ok = verify()
            results.append((n, bx, by, t, n**3))
            print(f"Size: {n}, Block: {bx}x{by}, Time: {t:.4f} sec | {'OK' if ok else 'FAIL'}")
    return results

def save_csv(results, filename="results_cuda.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "block_x", "block_y", "time_sec", "operations"])
        writer.writerows(results)

def plot_results(results):
    config_data = collections.defaultdict(lambda: {"sizes": [], "times": []})
    for n, bx, by, t, _ in results:
        key = f"{bx}x{by}"
        config_data[key]["sizes"].append(n)
        config_data[key]["times"].append(t)

    plt.figure(figsize=(10, 6))
    for cfg in sorted(config_data.keys()):
        plt.plot(config_data[cfg]["sizes"], config_data[cfg]["times"], marker='o', label=f'Block {cfg}')
    plt.xlabel("Matrix size"); plt.ylabel("Time (sec)")
    plt.title("CUDA Matrix Multiplication Performance")
    plt.legend(); plt.grid()
    plt.savefig("plot_cuda.png"); plt.show()

if __name__ == "__main__":
    sizes = [200, 400, 800, 1200, 1600, 2000]
    block_configs = [(8, 8), (16, 16), (32, 32)]
    results = run_experiments(sizes, block_configs)
    save_csv(results)
    plot_results()
    print("\nAll done")