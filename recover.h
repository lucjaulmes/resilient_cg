#ifndef RECOVER_H_INCLUDED
#define RECOVER_H_INCLUDED

#include "matrix.h" // define (Dense|Sparse|)Matrix

// before calling those, make sure that all the lost elements of x have been replaced
// either by their initial guess (uncorrelated)
// or by 0 (decorellated)
// (only for multiple faults with local strategies)
void get_rhs_dense(const int n, const int *rows, const int m, const int *except_cols, const int bs, const DenseMatrix *A, const double *b, const double *x, double *rhs);
void get_rhs_sparse(const int n, const int *rows, const int m, const int *except_cols, const int bs, const SparseMatrix *A, const double *b, const double *x, double *rhs);
void get_rhs_sparse_with_grad(const int n, const int *rows, const int m, const int *except_cols, const int bs, const SparseMatrix *A, const double *b, const double *g, const double *x, double *rhs);

#ifdef MATRIX_DENSE // using dense matrices
	#define get_rhs(n, rows, m, cols, bs, A, b, x, rhs) get_rhs_dense(n, rows, m, cols, bs, (DenseMatrix*)A, b, x, rhs)
#else
	#define get_rhs(n, rows, m, cols, bs, A, b, x, rhs) get_rhs_sparse(n, rows, m, cols, bs, (SparseMatrix*)A, b, x, rhs)
#endif

void do_interpolation_with_grad( const Matrix *A, const double *b, const double *g, double *x, const int nb_lost, const int *lost_blocks );
#define do_interpolation(A, b, x, nb_lost, lost_blocks) do_interpolation_with_grad(A, b, NULL, x, nb_lost, lost_blocks)
void do_leastsquares( const Matrix *A, const double *b, double *x, const int nb_lost, const int *lost_blocks );

void recover( const Matrix *A, const double *b, double *x, const char A_full_rank, const int strategy );

// aliases of recover, setting the A_full_rank parameter
static inline void recover_interpolation(const Matrix *A, const double *b, double *x, const int strategy )
{
	recover( A, b, x, 1, strategy );
}

static inline void recover_leastsquares( const Matrix *A, const double *b, double *x, const int strategy )
{
	recover( A, b, x, 0, strategy );
}

#endif // RECOVER_H_INCLUDED

