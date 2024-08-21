#include "common/matrix.hpp"

#ifndef __SUB_CHAIN_SPARSITY_ESTIMATOR_HPP__
#define __SUB_CHAIN_SPARSITY_ESTIMATOR_HPP__

/**
 * @brief Estimate the sparsity of all sub-chains in matrix chain A
 * 
 * @param A Matrix Chain
 * @param num_thread number of the thread to calculate (default=16)
 * @return Vector<Vector<double>> Sub-chain sparsity values in full matrix format
 */
Vector<Vector<double>> estimate_sub_chain_sparsity(MatrixChain<double>& A, uint num_thread=16);

#endif