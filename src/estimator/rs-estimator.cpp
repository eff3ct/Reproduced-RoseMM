/**
 * RS-estimator
 * It computes estimated row-wise sparsity vector of O=AxB
 */

#include <thread>
#include "rs-estimator.hpp"
#include "common/matrix.hpp"

Vector<double> compute_row_wise_sparsity(Matrix<double>& target) {
    int m = target.get_row_size();
    int n = target.get_col_size();

    // Create "m" amount space (where the target is a m by n matrix)
    Vector<double> result_sparsity(m);

    for (int i = 0; i < m; ++i) {
        double sparsity = target[i].size() / (double)n;
        result_sparsity[i] = sparsity;
    }
    
    return result_sparsity;
}

void calc_estimated_rs(int start, 
                       int end, 
                       Matrix<double>& A, 
                       Vector<double>& rs_B, 
                       Vector<double>& estimated) {
    for (int i = start; i < end; ++i) {
        double p = 1;
        for (auto nz: A[i]) {
            p *= (1 - rs_B[nz.col]);
        }
        estimated[i] = 1 - p;
    }
}

Vector<double> RS_estimator(Matrix<double>& A, Matrix<double>& B, uint num_thread) {
    int row_size = A.get_row_size();
    
    Vector<double> rs_B = compute_row_wise_sparsity(B);
    Vector<double> estimated_sparsity(row_size, 0);

    std::vector<std::thread> threads;

    int start = 0, end = 0;
    int delta = row_size / num_thread;
    int remainder = row_size % num_thread;
    for (int i = 0; i < num_thread; ++i) {
        end = start + delta;
        
        if (i < remainder) 
            end++;

        threads.emplace_back(
            calc_estimated_rs,
            start,
            end,
            std::ref(A),
            std::ref(rs_B),
            std::ref(estimated_sparsity)
        );

        start = end;
    }

    for (int i = 0; i < num_thread; ++i)
        threads[i].join();

    return estimated_sparsity;
}

Vector<double> RS_estimator(Matrix<double>& A, Vector<double>& rs_B, uint num_thread) {
    int row_size = A.get_row_size();
    
    Vector<double> estimated_sparsity(row_size, 0);

    std::vector<std::thread> threads;

    int start = 0, end = 0;
    int delta = row_size / num_thread;
    int remainder = row_size % num_thread;
    for (int i = 0; i < num_thread; ++i) {
        end = start + delta;
        
        if (i < remainder) 
            end++;

        threads.emplace_back(
            calc_estimated_rs,
            start,
            end,
            std::ref(A),
            std::ref(rs_B),
            std::ref(estimated_sparsity)
        );

        start = end;
    }

    for (int i = 0; i < num_thread; ++i)
        threads[i].join();

    return estimated_sparsity;
}