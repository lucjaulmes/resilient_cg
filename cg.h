#ifndef CG_H_INCLUDED
#define CG_H_INCLUDED

void solve_cg( const int n, const void *A, const double *b, double *x, double thres );
void restart_cg( const int n, const void *A, const double *b, double *x, double thres_sq, double *error, int *it );

#endif // CG_H_INCLUDED
