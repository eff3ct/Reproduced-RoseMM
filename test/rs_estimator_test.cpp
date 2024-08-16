#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "common/matrix.hpp"
#include "rs-estimator.hpp"

constexpr double EPS = 1e-6;

TEST(RsEstimatorTest, ComputeRowWiseSparsityTest_Square) {
    Vector<Vector<double>> _mat = {
        {1, 2, 3},
        {4, 0, 0},
        {5, 0, 6},
    };
    Matrix<double> mat(3, 3, _mat);

    Vector<double> expected = { 1.0, 1.0/3.0, 2.0/3.0 };
    Vector<double> computed = compute_row_wise_sparsity(mat);

    for (int i = 0; i < 3 ; ++i) {
        EXPECT_LE(abs(expected[i] - computed[i]), EPS);
    }
}

TEST(RsEstimatorTest, ComputeRowWiseSparsityTest_NonSquare) {
    Vector<Vector<double>> _mat = {
        {1, 2},
        {4, 0},
        {0, 5},
        {0, 0},
        {9, 2},
    };
    Matrix<double> mat(5, 2, _mat);

    Vector<double> expected = { 1.0, 1.0/2.0, 1.0/2.0, 0.0, 1.0 };
    Vector<double> computed = compute_row_wise_sparsity(mat);

    for (int i = 0; i < 5 ; ++i) {
        EXPECT_LE(abs(expected[i] - computed[i]), EPS);
    }
}

TEST(RsEstimatorTest, ComputeRowWiseSparsityTest_SingleRow) {
    Vector<Vector<double>> _mat = {
        {1, 0, 3, 0, 4, 2},
    };
    Matrix<double> mat(1, 6, _mat);

    Vector<double> expected = { 4.0/6.0 };
    Vector<double> computed = compute_row_wise_sparsity(mat);

    EXPECT_LE(abs(expected[0] - computed[0]), EPS);
}

TEST(RsEstimatorTest, RsEstimatorTest_Simple) {
    Vector<Vector<double>> _A = {
        {1, 0},
        {0, 1},
    };

    Vector<Vector<double>> _B = {
        {1, 1},
        {0, 1},
    };

    Matrix<double> A(2, 2, _A);
    Matrix<double> B(2, 2, _B);

    Vector<double> expected = { 1.0, 0.5 };
    Vector<double> rs_O = RS_estimator(A, B);

    for (int i = 0; i < 2; ++i)
        EXPECT_LE(abs(expected[0] - rs_O[0]), EPS);
}

TEST(RsEstimatorTest, RsEstimatorTest_NonTrivial) {
    Vector<Vector<double>> _A = {
        {1, 0, 1},
        {1, 1, 0},
    };

    Vector<Vector<double>> _B = {
        {1, 1, 0},
        {0, 1, 0},
        {0, 0, 0},
    };

    Matrix<double> A(2, 3, _A);
    Matrix<double> B(3, 3, _B);

    /**
     * r(O, 1) = 1 - (1 - 2/3)*(1 - 0)
     * r(O, 2) = 1 - (1 - 2/3)*(1 - 1/3)
     * * Estimated row-wise sparsity is a little bit different from the real value of it.
     * * + Actual value is { 2/3, 2/3 } (\approx { 2/3, 7/9 })
     */
    Vector<double> expected = { 2.0/3.0, 7.0/9.0 };
    Vector<double> rs_O = RS_estimator(A, B);

    for (int i = 0; i < 2; ++i)
        EXPECT_LE(abs(expected[0] - rs_O[0]), EPS);
}