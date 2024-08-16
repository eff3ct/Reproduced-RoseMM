#include <functional>
#include "common/matrix.hpp"
#include "ordering.hpp"

Vector<Vector<int>> SMCM_ordering(MatrixChain<double>& A, Vector<Vector<double>>& S) {
    /**
     * This function computes order of the matrix chain,
     * based on a classic interval dynamic programming (O(n^3)).
     * A cost is the different point, compared to the classic one.
     * The Cost is computed by following formula.
     * cost(A x B) = alpha * (m * n * p(A)) 
     *              + beta * (m * n * p(A) * l * p(B))
     *              + gamma * (m * l * p_hat(O))
     * This provides matrix sparsity informations into the DP computation.
     * These co-efficients can be obtained by multi-linear regression.
     * In this code, I'll use the suggested ones on the paper. (alpha = -130, beta = -16, gamma = 83)
     */
    constexpr int64_t INF = 1e18;

    // A_i,k x A_k+1,j
    auto get_cost = [&](int i, int j, int k) {
        int m = A[i].get_row_size();
        int n = A[i].get_col_size();
        int l = A[j].get_row_size();
        return VAL_ALPHA * (m * n * S[i][k])
               + VAL_BETA * (m * n * S[i][k] * l * S[k+1][j])
               + VAL_GAMMA * (m * l * S[i][j]);
    };

    int p = S.size();

    // cost dp 2-dim vector
    Vector<Vector<int64_t>> C(p, Vector<int64_t>(p));

    // order 2-dim vector
    Vector<Vector<int>> T(p, Vector<int>(p));

    for (int l = 1; l <= p-1; ++l) {
        for (int i = 0; i < p-l; ++i) {
            int j = i + l;
            C[i][j] = INF;
            for (int k = i; k < j; ++k) {
                int64_t cost = get_cost(i, j, k);
                int64_t curr_cost = C[i][k] + C[k+1][j] + cost;
                if (curr_cost < C[i][j]) {
                    C[i][j] = curr_cost;
                    T[i][j] = k;
                }
            }
        }
    }

    return T;
}