#ifndef RECOVER_H_INCLUDED
#define RECOVER_H_INCLUDED

#include "solvers.h" // to define the SolveFunction s

// before calling those, make sure that all the lost elements of x have been replaced
// either by their initial guess (uncorrelated)
// or by 0 (decorellated)
// (only for multiple faults with local strategies)
void rhs_dense(const int n, const int *rows, const int m, const int *except_cols, const void *mat, const double *b, const double *x, double *rhs);
void rhs_sparse(const int n, const int *rows, const int m, const int *except_cols, const void *mat, const double *b, const double *x, double *rhs);

typedef void (*RhsFunction)(const int, const int*, const int, const int*, const void*, const double*, const double*, double*);

void do_interpolation( const void *A, const double *b, double *x, const int nb_lost, const int *lost_blocks, SolveFunction solver);
void do_leastsquares( const void *A, const double *b, double *x, const int nb_lost, const int *lost_blocks );

void recover( const void *A, const double *b, double *x, SolveFunction solver, const char A_full_rank, const int strategy );

// aliases setting A_full_rank
void recover_interpolation(const void *A, const double *b, double *x, SolveFunction solver, const int strategy );
void recover_leastsquares( const void *A, const double *b, double *x, const int strategy );

#endif // RECOVER_H_INCLUDED

