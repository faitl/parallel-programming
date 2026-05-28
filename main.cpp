#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>
#include <chrono>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
  
    if (rank == 0) {
        ifstream file("matrix_a.txt");
        file >> n;
        if (n % size != 0) {
            cerr << "[WARNING] n is not divisible by the number of processes. Only the first " 
                 << (n / size) * size << " rows will be processed. " << endl;
        }
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_rows = n / size;       
    int chunk_size = chunk_rows * n;   


    vector<long long> A_local(chunk_size);
    vector<long long> B(n * n);
    vector<long long> C_local(chunk_size, 0);
    vector<long long> C_full(n * n); 


    if (rank == 0) {
        vector<long long> A_flat(n * n);
        ifstream fa("matrix_a.txt"); fa >> n;
        for (int i = 0; i < n * n; i++) fa >> A_flat[i];

        ifstream fb("matrix_b.txt"); fb >> n;
        for (int i = 0; i < n * n; i++) fb >> B[i];

        MPI_Scatter(A_flat.data(), chunk_size, MPI_LONG_LONG,
                    A_local.data(), chunk_size, MPI_LONG_LONG,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(nullptr, chunk_size, MPI_LONG_LONG,
                    A_local.data(), chunk_size, MPI_LONG_LONG,
                    0, MPI_COMM_WORLD);
    }


    MPI_Bcast(B.data(), n * n, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();


    for (int i = 0; i < chunk_rows; i++) {
        for (int k = 0; k < n; k++) {
            long long a_val = A_local[i * n + k];
            for (int j = 0; j < n; j++) {
                C_local[i * n + j] += a_val * B[k * n + j];
            }
        }
    }

    double end_time = MPI_Wtime();

 
    MPI_Gather(C_local.data(), chunk_size, MPI_LONG_LONG,
               C_full.data(), chunk_size, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        ofstream fout("result.txt");
        fout << n << "\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) fout << C_full[i * n + j] << " ";
            fout << "\n";
        }

        cout << "Processes: " << size << endl;
        cout << "Time: " << (end_time - start_time) << " sec" << endl;
        cout << "Operations: " << (long long)n * n * n << endl;
    }

    MPI_Finalize();
    return 0;
}