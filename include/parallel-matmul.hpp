#include "common/matrix.hpp"

#ifndef __PARALLEL_MATMUL__
#define __PARALLEL_MATMUL__

/**
 * @brief Parallel matrix multiplication. (Row-wise based)
 * 
 * @param A Matrix
 * @param B Matrix
 * @return Matrix<double> O=AxB
 */
Matrix<double> parallel_matrix_mult(Matrix<double>& A, Matrix<double>& B);

#endif