#ifndef GMRES_H_INCLUDED
#define GMRES_H_INCLUDED

#include "global.h"

void solve_gmres( const int n, const void *A, const double *b, double *x, double thres, const int restart );
void restart_gmres( const int n, const void *A, const double *b, double *x, double thres, const int max_steps, double *error, int *it );

void givens_rotate( const int n, double *r1, double *r2, const double cos, const double sin );

// choice of the gram schmidt method
void mgs(const int n, const int r, double *q_r, double *h, const double **q);

#endif // GMRES_H_INCLUDED
