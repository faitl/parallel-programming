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
        data = []
        for _ in range(n):
            data.append(list(map(float, f.readline().split())))
    return np.array(data)

def generate_matrix(n):
    return np.random.randint(-500, 500, (n, n))


def run_cpp(num_threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads) # [TASK2] Управление числом потоков OpenMP

    start = time.time()
    subprocess.run(["main.exe"], env=env, check=True, capture_output=False)
    end = time.time()
    return end - start

def verify():
    A = load_matrix("matrix_a.txt")
    B = load_matrix("matrix_b.txt")
    C_cpp = load_matrix("result.txt")
    C_py = A @ B
    return np.array_equal(C_cpp.astype(np.int64), C_py.astype(np.int64))

def run_experiments(sizes, threads_list):
    results = []

    for n in sizes:
        print(f"\nГенерация матриц размера {n}...")
        A = generate_matrix(n)
        B = generate_matrix(n)
        save_matrix("matrix_a.txt", A)
        save_matrix("matrix_b.txt", B)

        for t in threads_list:
            print(f"Запуск с {t} поток(ами)...")
            cpp_time = run_cpp(t)

            ok = verify()
            if not ok:
                print("Верификация не пройдена!")
                break

            operations = n ** 3
            results.append((n, t, cpp_time, operations))
            print(f"Size: {n}, Threads: {t}, Time: {cpp_time:.4f} sec | OK")

    return results

def save_csv(results, filename="results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "threads", "time_sec", "operations"])
        writer.writerows(results)

def plot_results(results):
    thread_data = collections.defaultdict(lambda: {"sizes": [], "times": []})

    for n, t, time_sec, ops in results:
        thread_data[t]["sizes"].append(n)
        thread_data[t]["times"].append(time_sec)

    plt.figure(figsize=(10, 6))
    for t in sorted(thread_data.keys()):
        plt.plot(thread_data[t]["sizes"], thread_data[t]["times"],
                 marker='o', label=f'{t} thread(s)')
    plt.xlabel("Matrix size")
    plt.ylabel("Time (sec)")
    plt.title("Matrix multiplication performance (OpenMP)")
    plt.legend()
    plt.grid()
    plt.savefig("plot.png")
    plt.show()

def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]
    threads = [1, 2, 4, 8]

    results = run_experiments(sizes, threads)
    save_csv(results)
    plot_results(results)
    print("\nAll done")

if __name__ == "__main__":
    main()