#include "common/matrix.hpp"

#ifndef __RS_ESTIMATOR_HPP__
#define __RS_ESTIMATOR_HPP__

/**
 * @brief Compute the row-wise sparsity of "target" matrix.
 * 
 * @param target Matrix.
 * @return Vector<double> Row-wise sparsity values.
 */
Vector<double> compute_row_wise_sparsity(Matrix<double>& target);

/**
 * @brief Compute the row-wise "estimated" sparsity of O=AxB
 * 
 * @param A input matrix A
 * @param B input matrix B
 * @param num_thread number of the thread to calculate (default=16)
 * @return Vector<double> Estimated row-wise sparsity vector of O=AxB
 */
Vector<double> RS_estimator(Matrix<double>& A, Matrix<double>& B, uint num_thread=16);

/**
 * @brief Compute the row-wise "estimated" sparsity, where rs_B is a sparsity vector
 * 
 * @param A input matrix A
 * @param rs_B input sparsity vector rs_B
 * @param num_thread number of the thread to calculate (default=16)
 * @return Vector<double> Estimated row-wise sparsity vector
 */
Vector<double> RS_estimator(Matrix<double>& A, Vector<double>& rs_B, uint num_thread=16);

#endif