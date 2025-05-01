#include <omp.h>
#include <algorithm>
#include <iostream>

void parallel_histogram_sort(int *A, int *B, int n) {
    int T = omp_get_max_threads();

    int *local_mins = new int[T];
    int *local_maxs = new int[T];
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        local_mins[tid] = A[0];
        local_maxs[tid] = A[0];
        #pragma omp for
        for (int i = 0; i < n; i++) {
            if (A[i] < local_mins[tid]) local_mins[tid] = A[i];
            if (A[i] > local_maxs[tid]) local_maxs[tid] = A[i];
        }
    }
    int global_min = local_mins[0];
    int global_max = local_maxs[0];
    for (int t = 1; t < T; t++) {
        global_min = std::min(global_min, local_mins[t]);
        global_max = std::max(global_max, local_maxs[t]);
    }
    delete[] local_mins;
    delete[] local_maxs;

    int num_bins = global_max - global_min + 1;

    int *local_hist = new int[T * num_bins];
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int k = 0; k < num_bins; k++) {
            local_hist[tid * num_bins + k] = 0;
        }
        int start = (n / T) * tid;
        int end = (tid == T - 1) ? n : start + (n / T);
        for (int i = start; i < end; i++) {
            int k = A[i] - global_min;
            local_hist[tid * num_bins + k]++;
        }
    }

    int *global_hist = new int[num_bins];
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int bins_per_thread = (num_bins + T - 1) / T;
        int start_k = tid * bins_per_thread;
        int end_k = std::min(start_k + bins_per_thread, num_bins);
        for (int k = start_k; k < end_k; k++) {
            global_hist[k] = 0;
            for (int t = 0; t < T; t++) {
                global_hist[k] += local_hist[t * num_bins + k];
            }
        }
    }

    int *cumsum = new int[num_bins + 1];
    cumsum[0] = 0;
    for (int k = 0; k < num_bins; k++) {
        cumsum[k + 1] = cumsum[k] + global_hist[k];
    }

    int *prefix = new int[T * num_bins];
    #pragma omp parallel for
    for (int k = 0; k < num_bins; k++) {
        prefix[0 * num_bins + k] = 0;
        for (int t = 1; t < T; t++) {
            prefix[t * num_bins + k] = prefix[(t - 1) * num_bins + k] + local_hist[(t - 1) * num_bins + k];
        }
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = (n / T) * tid;
        int end = (tid == T - 1) ? n : start + (n / T);
        int *counter = new int[num_bins]();
        for (int i = start; i < end; i++) {
            int a = A[i];
            int k = a - global_min;
            int pos = cumsum[k] + prefix[tid * num_bins + k] + counter[k];
            B[pos] = a;
            counter[k]++;
        }
        delete[] counter;
    }

    delete[] local_hist;
    delete[] global_hist;
    delete[] cumsum;
    delete[] prefix;
}

int main() {
    int n = 16;
    int *A = new int[n] {5, 2, 9, 1, 5, 6, 3, 8, 2, 7, 4, 9, 1, 6, 3, 8};
    int *B = new int[n];

    parallel_histogram_sort(A, B, n);

    std::cout << "Sorted array: ";
    for (int i = 0; i < n; i++) {
        std::cout << B[i] << " ";
    }
    std::cout << std::endl;

    delete[] A;
    delete[] B;
    return 0;
}

