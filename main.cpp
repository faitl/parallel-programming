#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

vector<vector<long long>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
    file >> n;
    vector<vector<long long>> matrix(n, vector<long long>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> matrix[i][j];

    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<long long>>& matrix) {
    ofstream file(filename);
    int n = matrix.size();
    file << n << endl;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << matrix[i][j] << " ";
        file << endl;
    }
}

vector<vector<long long>> multiply(
    const vector<vector<long long>>& A,
    const vector<vector<long long>>& B,
    int n
) {
    vector<vector<long long>> C(n, vector<long long>(n, 0));

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i][j] += (long long)A[i][k] * B[k][j];

    return C;
}

int main() {
    int n1, n2;

    auto A = readMatrix("matrix_a.txt", n1);
    auto B = readMatrix("matrix_b.txt", n2);

    if (n1 != n2) {
        cout << "Matrix sizes mismatch!" << endl;
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();

    auto C = multiply(A, B, n1);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    writeMatrix("result.txt", C);

    cout << "Time: " << duration.count() << " sec" << endl;
    cout << "Operations: " << n1 * n1 * n1 << endl;

    return 0;
}