#include "common/matrix.hpp"
#include "rose-mm.hpp"
#include "sub-chain-sparsity-estimator.hpp"
#include "parallel-matmul.hpp"
#include "ordering.hpp"

Matrix<double> _mat_mul(int left, 
                        int right, 
                        MatrixChain<double>& A, 
                        Vector<Vector<int>>& order,
                        uint num_thread) {
    if (left == right) return A[left];

    int mid = order[left][right];
    Matrix<double> left_matrix = _mat_mul(left, mid, A, order, num_thread);
    Matrix<double> right_matrix = _mat_mul(mid+1, right, A, order, num_thread);

    return parallel_matrix_mult(left_matrix, right_matrix, num_thread);
}

Matrix<double> compute_SMCM(MatrixChain<double>& A, Vector<Vector<int>>& order, uint num_thread) {
    return _mat_mul(0, A.size()-1, A, order, num_thread);
}

Matrix<double> RoseMM(MatrixChain<double>& A, uint num_thread) {
    auto S = estimate_sub_chain_sparsity(A, num_thread);
    auto T = SMCM_ordering(A, S);
    auto O = compute_SMCM(A, T, num_thread);
    return O;
}