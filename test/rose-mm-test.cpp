#include <gtest/gtest.h>
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