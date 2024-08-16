#include "common/matrix.hpp"

/**
 * @brief Compute sparse matrix chain multiplication
 * 
 * @param A Matrix chain
 * @return Matrix<double> Result of SMCM 
 */
Matrix<double> RoseMM(MatrixChain<double>& A);