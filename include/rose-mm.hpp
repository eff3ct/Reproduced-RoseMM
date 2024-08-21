#include "common/matrix.hpp"

/**
 * @brief Compute sparse matrix chain multiplication
 * 
 * @param A Matrix chain
 * @param num_thread number of the thread to calculate (default=16)
 * @return Matrix<double> Result of SMCM 
 */
Matrix<double> RoseMM(MatrixChain<double>& A, uint num_thread=16);