import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def load_csv(filename):
    data = []
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'size': int(row['size']),
                'processes': int(row['processes']),
                'time': float(row['time_sec']),
                'ops': int(row['operations'])
            })
    return data

def verify_data(data):

    for row in data:
        if row['time'] <= 0:
            return False, f"Отрицательное или нулевое время для n={row['size']}"
        if row['ops'] != row['size']**3:
            return False, f"Несоответствие операций для n={row['size']}"
    return True, "Данные корректны"

def plot_results(data, output="plot_mpi_hpc.png"):

    proc_data = {}
    for row in data:
        p = row['processes']
        if p not in proc_data:
            proc_data[p] = {'sizes': [], 'times': []}
        proc_data[p]['sizes'].append(row['size'])
        proc_data[p]['times'].append(row['time'])

    plt.figure(figsize=(9, 5))
    for p in sorted(proc_data.keys()):
        plt.plot(proc_data[p]['sizes'], proc_data[p]['times'], 
                 marker='o', label=f'{p} proc')
    plt.xlabel('Matrix size'); plt.ylabel('Time (sec)')
    plt.title('MPI Matrix Multipulation on Supercomputer')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"График сохранён: {output}")

def main():
    csv_file = 'results_mpi_sc.csv'
    try:
        data = load_csv(csv_file)
    except FileNotFoundError:
        print(f"Файл {csv_file} не найден. Запустите сначала MPI-задачи.")
        sys.exit(1)

    ok, msg = verify_data(data)
    print(f"Проверка данных: {msg}")
    if not ok:
        sys.exit(1)

    plot_results(data)
    

    print("\nУскорение (S_p = T_1 / T_p):")
    base_times = {row['size']: row['time'] for row in data if row['processes'] == 1}
    for row in data:
        if row['processes'] > 1:
            speedup = base_times.get(row['size'], 0) / row['time']
            print(f"  n={row['size']}, p={row['processes']}: {speedup:.2f}×")

if __name__ == '__main__':
    main()