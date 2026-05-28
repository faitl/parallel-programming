#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <blockX> <blockY>" << endl;
        return 1;
    }

    int blockX = atoi(argv[1]), blockY = atoi(argv[2]);
    ifstream f("matrix_a.txt"); int n; f >> n; f.close();
    int size = n * n;
    
    vector<float> h_A(size), h_B(size), h_C(size);
    ifstream a("matrix_a.txt"); a >> n; for(int i=0;i<size;i++) a >> h_A[i];
    ifstream b("matrix_b.txt"); b >> n; for(int i=0;i<size;i++) b >> h_B[i];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size*sizeof(float));
    cudaMalloc(&d_B, size*sizeof(float));
    cudaMalloc(&d_C, size*sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 bs(blockX, blockY), gs((n+blockX-1)/blockX, (n+blockY-1)/blockY);
    
    auto t0 = chrono::high_resolution_clock::now();
    matrixMulKernel<<<gs, bs>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();

    cudaMemcpy(h_C.data(), d_C, size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    ofstream out("result.txt"); out << n << "\n";
    for(int i=0;i<n;i++) { 
        for(int j=0;j<n;j++) out << h_C[i*n+j] << " "; 
        out << "\n"; 
    }

    cout << "Block: " << blockX << "x" << blockY << endl;
    cout << "Time: " << chrono::duration<double>(t1-t0).count() << " sec" << endl;
    cout << "Operations: " << (long long)n * n * n << endl;
    return 0;
}