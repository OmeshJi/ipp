#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main() {
    const int N = 3, M = 3, P = 3;

    vector<vector<int>> A(N, vector<int>(M, 1));
    vector<vector<int>> B(M, vector<int>(P, 2));
    vector<vector<int>> C(N, vector<int>(P, 0));

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            for (int k = 0; k < M; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}

