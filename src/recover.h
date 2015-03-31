#ifndef RECOVER_H_INCLUDED
#define RECOVER_H_INCLUDED

#include "matrix.h" // define (Dense|Sparse|)Matrix
#include "csparse.h" // define css/csn
#include "pcg.h" // define Precond

// before calling those, make sure that all the lost elements of x have been replaced
// either by their initial guess (uncorrelated)
// or by 0 (decorellated)
// (only for multiple faults with local strategies)
void get_rhs(const int n, const int *rows, const int m, const int *except_cols, const int bs, const Matrix *A, const double *b, const double *x, double *rhs);
void get_rhs_with_grad(const int n, const int *rows, const int m, const int *except_cols, const int bs, const Matrix *A, const double *b, const double *g, const double *x, double *rhs);

void do_interpolation( const double *rhs, double *x, const int total_lost, css *S, csn *N );
void do_single_interpolation( const Matrix *A, const double *b, const double *g, double *x, const int lost_block, css *S, csn *N );
void do_multiple_interpolation( const Matrix *A, const double *b, const double *g, double *x, const int nb_lost, const int *lost_blocks, css **S, csn **N );

void recover_xk( const Matrix *A, const double *b, const double *g, double *x, const Precond *M, const int strategy, int *lost_blocks, const int nb_lost );

static inline void recover( const Matrix *A, const double *b, double *x, const Precond *M, const int strategy, int *lost_blocks, const int nb_lost )
{
	recover_xk(A, b, NULL, x, M, strategy, lost_blocks, nb_lost);
}

// aliases of recover, setting the A_full_rank parameter
#endif // RECOVER_H_INCLUDED

