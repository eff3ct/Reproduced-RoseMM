#include "common/matrix.hpp"
#include "rose-mm.hpp"
#include "sub-chain-sparsity-estimator.hpp"
#include "parallel-matmul.hpp"
#include "ordering.hpp"

Matrix<double> _mat_mul(int left, 
                        int right, 
                        MatrixChain<double>& A, 
                        Vector<Vector<int>>& order) {
    if (left == right) return A[left];

    int mid = order[left][right];
    Matrix<double> left_matrix = _mat_mul(left, mid, A, order);
    Matrix<double> right_matrix = _mat_mul(mid+1, right, A, order);

    return parallel_matrix_mult(left_matrix, right_matrix);
}

Matrix<double> compute_SMCM(MatrixChain<double>& A, Vector<Vector<int>>& order) {
    return _mat_mul(0, A.size()-1, A, order);
}

Matrix<double> RoseMM(MatrixChain<double>& A) {
    auto S = estimate_sub_chain_sparsity(A);
    auto T = SMCM_ordering(A, S);
    auto O = compute_SMCM(A, T);
    return O;
}