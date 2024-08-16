#include <gtest/gtest.h>
#include "common/matrix.hpp"
#include "ordering.hpp"
#include "sub-chain-sparsity-estimator.hpp"

TEST(SmcmOrderingTest, DenseMatrixOrderingTest) {
    Vector<Vector<double>> _A1 = {
        {1, 1},
    };

    Vector<Vector<double>> _A2 = {
        {1, 1, 1},
        {1, 1, 1},
    };

    Vector<Vector<double>> _A3 = {
        {1},
        {1},
        {1},
    }; 

    Matrix<double> A1(1, 2, _A1);
    Matrix<double> A2(2, 3, _A2);
    Matrix<double> A3(3, 1, _A3);
    MatrixChain<double> A = { A1, A2, A3 };

    // A1(A2A3) is the best case.
    // T[0][2] equals to 0.
    Vector<Vector<double>> S = estimate_sub_chain_sparsity(A);
    Vector<Vector<int>> T = SMCM_ordering(A, S);

    EXPECT_EQ(T[0][2], 0);
}

TEST(SmcmOrderingTest, SparseMatrixOrderingTest) {
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
    
    Vector<Vector<double>> S = estimate_sub_chain_sparsity(A);
    Vector<Vector<int>> T = SMCM_ordering(A, S);

    EXPECT_EQ(T[0][2], 1);
}