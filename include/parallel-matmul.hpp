#include "common/matrix.hpp"

#ifndef __PARALLEL_MATMUL__
#define __PARALLEL_MATMUL__

/**
 * @brief Parallel matrix multiplication. (Row-wise based)
 * 
 * @param A Matrix
 * @param B Matrix
 * @param num_thread number of the thread to calculate (default=16)
 * @return Matrix<double> O=AxB
 */
Matrix<double> parallel_matrix_mult(Matrix<double>& A, Matrix<double>& B, uint num_thread=16);

#endif