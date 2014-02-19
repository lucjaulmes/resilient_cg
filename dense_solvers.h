#ifndef SOLVERS_H_INCLUDED
#define SOLVERS_H_INCLUDED

#include "matrix.h"

// use for any matrices when doing least square interpolation (A is not square, but full column rank. A->n > A->m)
void solve_qr_house(const DenseMatrix *A, const double* rhs, double *x);

// use for non-spd matrices when doing linear interpolation (A is full rank)
void solve_lu(const DenseMatrix *A, const double* rhs, double *x);

// use for spd matrices when doing linear interpolation (A is full rank)
void solve_cholesky(const DenseMatrix *A, const double* rhs, double *x);


#endif // SOLVERS_H_INCLUDED

