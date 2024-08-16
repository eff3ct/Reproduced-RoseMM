/**
 * Sub Chain Sparsity Estimator
 * Compute all of the sub-chain sparsity of input matrix-chain.
 * Estimated values are used for SMCM ordering. 
 */

#include "common/matrix.hpp"
#include "rs-estimator.hpp"
#include "sub-chain-sparsity-estimator.hpp"

Vector<Vector<double>> estimate_sub_chain_sparsity(MatrixChain<double>& A) {
    // length of the matrix chain A
    int p = A.size();

    // p by p matrix which stores vector
    Vector<Vector<Vector<double>>> R(p, Vector<Vector<double>>(p));

    // p by p matrix that stores sub-chain sparsity
    Vector<Vector<double>> S(p, Vector<double>(p));

    // get mean value of vector (sum(v_i) / #row)
    auto get_mean = [](Vector<double>& v) {
        double sum = 0;
        for (double v_i: v)
            sum += v_i;
        return sum / v.size();
    };

    for (int i = p-1; i >= 0; --i) {
        R[i][i] = compute_row_wise_sparsity(A[i]);
        S[i][i] = get_mean(R[i][i]);

        for (int j = p-1; j >= i+1; --j) {
            R[i][j] = RS_estimator(A[i], R[i+1][j]);
            S[i][j] = get_mean(R[i][j]);
        }
    }

    return S;
}
