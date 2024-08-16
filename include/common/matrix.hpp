#include <vector>
#include <iostream>
#include <stdexcept>

#ifndef __COMMON_MATRIX_HPP__
#define __COMMON_MATRIX_HPP__

#define NUM_THREAD 16

template<class T>
using Vector = std::vector<T>;

template<class T>
class Matrix {
    private:
        struct Entry {
            int col;
            T val;
        };

        // m = #row, n = #col
        int m, n;

        // Sparse adjacent list for matrix
        Vector<Vector<Entry>> sp_adj_matrix;

    public:
        Matrix(int m, int n): m(m), n(n) { sp_adj_matrix.resize(m); }
        Matrix(int m, int n, Vector<Vector<T>> mat): m(m), n(n) {
            sp_adj_matrix.resize(m);
            // Make ordinary matrix representation into sparse adjacent list.
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (mat[i][j] != 0) {
                        // {col, val}
                        Entry new_entry = {j, mat[i][j]};
                        sp_adj_matrix[i].push_back(new_entry);
                    }
                }
            }
        }
        int get_row_size(void) { return m; }
        int get_col_size(void) { return n; }
        Vector<Entry>& operator[] (int idx) {
            // TODO: exception handling
            return sp_adj_matrix[idx];
        }
};

template <class T>
using MatrixChain = Vector<Matrix<T>>;

#endif