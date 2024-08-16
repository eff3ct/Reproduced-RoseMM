#include <thread>
#include <parallel-matmul.hpp>

void calc_row(int start, 
              int end, 
              Matrix<double>& A,
              Matrix<double>& B,
              Matrix<double>& O) {
    int l = B.get_col_size();
    for (int i = start; i < end; ++i) {
        Vector<double> V(l);
        Vector<int> C(l);
        Vector<bool> F(l);
        int count = 0;

        for (auto a: A[i]) {
            for (auto b: B[a.col]) {
                if (!F[b.col]) {
                    F[b.col] = true;
                    C[count++] = b.col;
                    V[b.col] = a.val * b.val;
                } 
                else 
                    V[b.col] += a.val * b.val;
            }
        }

        for (int j = 0; j < count-1; ++j) {
            if (V[C[j]] != 0) 
                O[i].push_back({ C[j], V[C[j]] });
        }
    }
}

Matrix<double> parallel_matrix_mult(Matrix<double>& A, Matrix<double>& B) {
    int m = A.get_row_size();
    int n = A.get_col_size();
    int l = B.get_col_size();

    Matrix<double> O(m, l);

    Vector<std::thread> threads;

    int start = 0, end = 0;
    int delta = m / NUM_THREAD;
    int remainder = m % NUM_THREAD;
    for (int i = 0; i < NUM_THREAD; ++i) {
        end = start + delta;
        
        if (i < remainder) 
            end++;

        threads.emplace_back(
            calc_row,
            start,
            end,
            std::ref(A),
            std::ref(B),
            std::ref(O)
        );

        start = end;
    }

    for (int i = 0; i < NUM_THREAD; ++i)
        threads[i].join();

    return O;
}