import numpy as np
import time
import csv
import subprocess
import matplotlib.pyplot as plt



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


def run_cpp():
    start = time.time()

    subprocess.run(["./a.out"], check=True)

    end = time.time()
    return end - start


def verify():
    A = load_matrix("matrix_a.txt")
    B = load_matrix("matrix_b.txt")
    C_cpp = load_matrix("result.txt")

    C_py = A @ B

    return np.array_equal(C_cpp, C_py)


def run_experiments(sizes):
    results = []

    for n in sizes:
        print(f"\nRunning size {n}...")

        A = generate_matrix(n)
        B = generate_matrix(n)

        save_matrix("matrix_a.txt", A)
        save_matrix("matrix_b.txt", B)

        cpp_time = run_cpp()

        ok = verify()

        if not ok:
            print("Verification failed!")
            break

        operations = n ** 3

        results.append((n, cpp_time, operations))

        print(f"Time: {cpp_time:.4f} sec | OK")

    return results


def save_csv(results, filename="results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "time_sec", "operations"])
        writer.writerows(results)


def plot_results(results):
    sizes = [r[0] for r in results]
    times = [r[1] for r in results]

    plt.figure()
    plt.plot(sizes, times, marker='o')
    plt.xlabel("Matrix size")
    plt.ylabel("Time (sec)")
    plt.title("Matrix multiplication performance")
    plt.grid()

    plt.savefig("plot.png")
    plt.show()


def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]

    results = run_experiments(sizes)

    save_csv(results)
    plot_results(results)

    print("\nAll done")


if __name__ == "__main__":
    main()