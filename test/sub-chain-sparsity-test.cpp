#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "common/matrix.hpp"
#include "sub-chain-sparsity-estimator.hpp"

constexpr double EPS = 1e-6;

TEST(SubChainSparsityTest, OnlyTwoMatricesTest) {
    Vector<Vector<double>> _A1 = {
        {1, 0},
        {0, 1},
    };

    Vector<Vector<double>> _A2 = {
        {1, 1},
        {0, 1},
    };

    Matrix<double> A1(2, 2, _A1);
    Matrix<double> A2(2, 2, _A2);
    MatrixChain<double> A = { A1, A2 };

    Vector<Vector<double>> S = estimate_sub_chain_sparsity(A);
    
    // Single range [i, i] test.
    EXPECT_LE(abs(S[0][0] - 0.5), EPS);
    EXPECT_LE(abs(S[1][1] - 0.75), EPS);

    // Interval [0, 1] test
    EXPECT_LE(abs(S[0][1] - 0.75), EPS);
}

TEST(SubChainSparsityTest, TrippleMatricesTest) {
    Vector<Vector<double>> _A1 = {
        {1, 0, 1},
        {0, 1, 0},
        {0, 0, 0},
    };

    Vector<Vector<double>> _A2 = {
        {1, 0, 0},
        {1, 1, 0},
        {1, 0, 1},
    };

    Vector<Vector<double>> _A3 = {
        {0, 0, 1},
        {1, 1, 1},
        {0, 0, 1},
    }; 

    Matrix<double> A1(3, 3, _A1);
    Matrix<double> A2(3, 3, _A2);
    Matrix<double> A3(3, 3, _A3);
    MatrixChain<double> A = { A1, A2, A3 };

    Vector<Vector<double>> S = estimate_sub_chain_sparsity(A);

    // Interval [0, 1], [1, 2] test.
    EXPECT_LE(abs(S[0][1] - (13.0/27.0)), EPS);
    EXPECT_LE(abs(S[1][2] - (17.0/27.0)), EPS);

    // Interval [0, 2] test.
    EXPECT_LE(abs(S[0][2] - (46.0/81.0)), EPS);
}