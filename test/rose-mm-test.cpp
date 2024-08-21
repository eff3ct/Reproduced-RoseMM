#include <gtest/gtest.h>
#include <random>
#include "common/matrix.hpp"
#include "rose-mm.hpp"

TEST(RoseMmTest, SimpleMatrixChainMultiplication) {
    Vector<Vector<double>> _A1 = {
        {1, 0},
    };

    Vector<Vector<double>> _A2 = {
        {1, 0, 1},
        {0, 0, 1},
    };

    Vector<Vector<double>> _A3 = {
        {0},
        {0},
        {1},
    }; 

    Matrix<double> A1(1, 2, _A1);
    Matrix<double> A2(2, 3, _A2);
    Matrix<double> A3(3, 1, _A3);
    MatrixChain<double> A = { A1, A2, A3 };
    
    Matrix<double> O = RoseMM(A);
    EXPECT_EQ(O.get_col_size(), 1);
    EXPECT_EQ(O.get_row_size(), 1);
    EXPECT_EQ(O[0][0].val, 1.0);
}

Matrix<double> _gen_random_sparse_matrix(int m, int n) {
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

    return Matrix(m, n, gen_mat);
}

MatrixChain<double> gen_random_matrix_chain(int k, int l, int r) {
    std::mt19937 mt(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> size_dist(l, r);

    MatrixChain<double> matrix_chain;

    int m = size_dist(mt);
    int n = size_dist(mt);
    for (int i = 0; i < k; ++i) {
        Matrix<double> mat = _gen_random_sparse_matrix(m, n);
        m = n;
        n = size_dist(mt);
        matrix_chain.push_back(mat);
    }

    return matrix_chain;
}

TEST(RoseMmTest, ComplexMatrixChainMultiplication) {
    const int chain_length = 100;
    const int min_m = 10;
    const int max_m = 500;

    MatrixChain<double> matrix_chain = gen_random_matrix_chain(100, 10, 500);
    Matrix<double> result = RoseMM(matrix_chain);
}

TEST(RoseMmTest, SmcmComparedWithAnotherMethods) {
    // TODO: Add some other methods.
}