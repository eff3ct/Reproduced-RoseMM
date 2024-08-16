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
 * @return Vector<double> Estimated row-wise sparsity vector of O=AxB
 */
Vector<double> RS_estimator(Matrix<double>& A, Matrix<double>& B);

/**
 * @brief Compute the row-wise "estimated" sparsity, where rs_B is a sparsity vector
 * 
 * @param A input matrix A
 * @param rs_B input sparsity vector rs_B
 * @return Vector<double> Estimated row-wise sparsity vector
 */
Vector<double> RS_estimator(Matrix<double>& A, Vector<double>& rs_B);

#endif