#include <gtest/gtest.h>
#include "common/matrix.hpp"
#include "parallel-matmul.hpp"

#include <ctime>
#include <random>

Vector<Vector<double>> naive_matrix_mult(Vector<Vector<double>>& A, Vector<Vector<double>>& B) {
    int m = A.size();
    int n = A[0].size();
    int k = B[0].size();

    Vector<Vector<double>> O(m, Vector<double>(k));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int t = 0; t < n; ++t) {
                O[i][j] += A[i][t] * B[t][j];
            }
        }
    }

    return O;
}

Vector<Vector<double>> convert_to_dense_form(Matrix<double>& A) {
    int m = A.get_row_size();
    int n = A.get_col_size();

    Vector<Vector<double>> O(m, Vector<double>(n));
    for (int i = 0; i < m; ++i) {
        for (auto a: A[i]) {
            O[i][a.col] = a.val;
        }
    }

    return O;
}

TEST(ParallelMatrixMultiplicationTest, SimpleMatrixSquare) {
    Vector<Vector<double>> _A1 = {
        {1, 0, 1, 0, 1, 0, 1, 0},
        {0, 1, 1, 0, 1, 0, 0, 0},
        {1, 0, 1, 1, 1, 0, 1, 1},
        {0, 1, 1, 0, 1, 1, 1, 0},
        {1, 0, 1, 0, 1, 0, 1, 1},
        {1, 0, 1, 0, 0, 0, 1, 0},
        {0, 1, 1, 1, 1, 1, 0, 1},
        {1, 0, 1, 0, 1, 0, 1, 0},
    };
    Matrix<double> A(8, 8, _A1);

    Matrix<double> AA = parallel_matrix_mult(A, A);
    Vector<Vector<double>> _AA = convert_to_dense_form(AA);
    Vector<Vector<double>> expected = naive_matrix_mult(_A1, _A1);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            EXPECT_EQ(_AA[i][j], expected[i][j])
                << "[i, j]: [" << i << ", " << j << "]\n";
        }
    }
}

Vector<Vector<double>> gen_random_sparse_matrix(int m, int n) {
    int non_zero_count = (m+n) / 2;
    Vector<Vector<double>> gen_mat(m, Vector<double>(n, 0.0));

    std::mt19937 mt(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> row_dist(0, m-1);
    std::uniform_int_distribution<int> col_dist(0, n-1);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0); 

    for (int i = 0; i < non_zero_count; ++i) {
        int row = row_dist(mt);
        int col = col_dist(mt);

        while (gen_mat[row][col] != 0.0) {
            row = row_dist(mt);
            col = col_dist(mt);
        }

        gen_mat[row][col] = val_dist(mt);
    }

    return gen_mat;
}

TEST(ParallelMatrixMultiplicationTest, LargeSparseMatrixMultiplication) {
    const int m = 1000;
    const int n = 1000;

    Vector<Vector<double>> _A = gen_random_sparse_matrix(m, n);
    Vector<Vector<double>> _B = gen_random_sparse_matrix(m, n);
    Matrix<double> A(m, n, _A);
    Matrix<double> B(m, n, _B);

    clock_t st, ed;
    st = clock();
    Matrix<double> AB = parallel_matrix_mult(A, B);
    ed = clock();
    double p_matmul_time = (double)(ed - st);
    
    st = clock();
    Vector<Vector<double>> expected = naive_matrix_mult(_A, _B);
    ed = clock();
    double n_matmul_time = (double)(ed - st);

    // Threaded version should be faster.
    EXPECT_LT(p_matmul_time, n_matmul_time);

    std::cout.precision(7);
    std::cout << std::fixed;

    std::cout << "[INFO] p_matmul_time: " << p_matmul_time / CLOCKS_PER_SEC << " (s)\n";
    std::cout << "[INFO] n_matmul_time: " << n_matmul_time / CLOCKS_PER_SEC << " (s)\n";

    Vector<Vector<double>> _AB = convert_to_dense_form(AB);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_EQ(_AB[i][j], expected[i][j])
                << "[i, j]: [" << i << ", " << j << "]\n";
        }
    }
}