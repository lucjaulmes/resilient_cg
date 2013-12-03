#ifndef RECOVER_H_INCLUDED
#define RECOVER_H_INCLUDED

#include "solvers.h" // to define the SolveFunction s
#include "matrix.h" // define (Dense|Sparse|)Matrix

// before calling those, make sure that all the lost elements of x have been replaced
// either by their initial guess (uncorrelated)
// or by 0 (decorellated)
// (only for multiple faults with local strategies)
void get_rhs_dense(const int n, const int *rows, const int m, const int *except_cols, const int bs, const DenseMatrix *A, const double *b, const double *x, double *rhs);
void get_rhs_sparse(const int n, const int *rows, const int m, const int *except_cols, const int bs, const SparseMatrix *A, const double *b, const double *x, double *rhs);

#ifdef MATRIX_DENSE // using dense matrices
	#define get_rhs(n, rows, m, cols, bs, A, b, x, rhs) get_rhs_dense(n, rows, m, cols, bs, (DenseMatrix*)A, b, x, rhs)
#else
	#define get_rhs(n, rows, m, cols, bs, A, b, x, rhs) get_rhs_sparse(n, rows, m, cols, bs, (SparseMatrix*)A, b, x, rhs)
#endif

void do_interpolation( const Matrix *A, const double *b, double *x, const int nb_lost, const int *lost_blocks, SolveFunction solver);
void do_leastsquares( const Matrix *A, const double *b, double *x, const int nb_lost, const int *lost_blocks );

void recover( const Matrix *A, const double *b, double *x, SolveFunction solver, const char A_full_rank, const int strategy );

// aliases setting A_full_rank
void recover_interpolation(const Matrix *A, const double *b, double *x, SolveFunction solver, const int strategy );
void recover_leastsquares( const Matrix *A, const double *b, double *x, const int strategy );

#endif // RECOVER_H_INCLUDED

