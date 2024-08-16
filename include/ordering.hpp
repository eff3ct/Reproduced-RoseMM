#include "common/matrix.hpp"

#ifndef __ORDERING_HPP__
#define __ORDERING_HPP__

#define VAL_ALPHA -130
#define VAL_BETA -16
#define VAL_GAMMA 83


/**
 * @brief Compute the almost? optimal order of SMCM
 * 
 * @param A Matrix Chain
 * @param S Pre-computed sub-chain sparsity
 * @return Vector<Vector<int>> order
 */
Vector<Vector<int>> SMCM_ordering(MatrixChain<double>& A, Vector<Vector<double>>& S);

#endif