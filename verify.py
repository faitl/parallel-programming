import numpy as np
import time
import csv
import subprocess
import matplotlib.pyplot as plt
import os
import collections
import platform


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



def run_mpi(num_procs):
    start = time.time()


    if platform.system() == "Windows":
        mpi_cmd = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
        exe_path = os.path.abspath("main.exe")
    else:
        mpi_cmd = "mpirun"
        exe_path = os.path.abspath("./a.out")


    cmd = [mpi_cmd, "-n", str(num_procs), exe_path]
    subprocess.run(cmd, check=True, capture_output=False)

    end = time.time()
    return end - start


def verify():
    A = load_matrix("matrix_a.txt")
    B = load_matrix("matrix_b.txt")
    C_cpp = load_matrix("result.txt")
    C_py = A @ B
    return np.array_equal(C_cpp.astype(np.int64), C_py.astype(np.int64))



def run_experiments(sizes, procs_list):
    results = []
    for n in sizes:
        print(f"\nГенерация матриц размера {n}...")
        A = generate_matrix(n)
        B = generate_matrix(n)
        save_matrix("matrix_a.txt", A)
        save_matrix("matrix_b.txt", B)

        for p in procs_list:
            print(f"Запуск с {p} процессом(ами)...")
            cpp_time = run_mpi(p)

            if not verify():
                print("Верификация не пройдена!");
                break

            results.append((n, p, cpp_time, n ** 3))
            print(f"Size: {n}, Procs: {p}, Time: {cpp_time:.4f} sec | OK")
    return results


def save_csv(results, filename="results_mpi.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "processes", "time_sec", "operations"])  # [TASK3]
        writer.writerows(results)


def plot_results(results):
    proc_data = collections.defaultdict(lambda: {"sizes": [], "times": []})
    for n, p, t, _ in results:
        proc_data[p]["sizes"].append(n)
        proc_data[p]["times"].append(t)

    plt.figure(figsize=(10, 6))
    for p in sorted(proc_data.keys()):
        plt.plot(proc_data[p]["sizes"], proc_data[p]["times"], marker='o', label=f'{p} process(es)')
    plt.xlabel("Matrix size");
    plt.ylabel("Time (sec)")
    plt.title("Matrix multiplication performance (MPI)")
    plt.legend();
    plt.grid()
    plt.savefig("plot_mpi.png");
    plt.show()


def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]
    procs = [1, 2, 4, 8]
    results = run_experiments(sizes, procs)
    save_csv(results);
    plot_results(results)
    print("\nAll done")


if __name__ == "__main__":
    main()