#ifndef CG_H_INCLUDED
#define CG_H_INCLUDED

#include "matrix.h"
#include "csparse.h"

typedef struct Precond 
{
	css **S;
	csn **N;
} Precond;


void make_blockedjacobi_preconditioner( Precond *M, const Matrix *A );
void allocate_preconditioner(Precond *M, const int maxblocks);
void deallocate_preconditioner(Precond *M);

void factorize_jacobiblock( const int n, const int block, const Matrix *A, css **S, csn **N );
void apply_preconditioner(const int n, const double *g, double *z, Precond *M);

void solve_pcg( const int n, const Matrix *A, const double *b, double *x, double thres );

#endif // CG_H_INCLUDED
