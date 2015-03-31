#ifndef CG_H_INCLUDED
#define CG_H_INCLUDED

#include "global.h"
#include "matrix.h"

#include "csparse.h"

typedef struct Precond 
{
	css **S;
	csn **N;
} Precond;


void allocate_preconditioner(Precond **M, const int maxblocks);
void make_blockedjacobi_preconditioner(Precond *M, const Matrix *A, char **wait_for_precond);
void deallocate_preconditioner(Precond *M);

void factorize_jacobiblock(const int n, const int block, const Matrix *A, css **S, csn **N);
void apply_preconditioner(const int n, const double *g, double *z, Precond *M, char **wait_for_precond);

void solve_pcg( const int n, const Matrix *A, Precond *M, const double *b, double *iterate, double thres );

// all the algorithmical steps of CG that will be subdivided into tasks : 
void update_gradient(const int n, double *gradient, double *Ap, double *alpha);
void recompute_gradient(const int n, double *gradient, const Matrix *A, double *iterate, char *wait_for_iterate, double *Aiterate, const double *b);
void update_p(const int n, double *p, double *old_p, char *wait_for_p, double *gradient, double *beta);
void update_iterate(const int n, double *iterate, char *wait_for_iterate, double *p, double *alpha);
void compute_Ap(const int n, const Matrix *A, double *p, char *wait_for_p, double *Ap);

void scalar_product_task( const int n, const double *v, const double *w, double* r );
void norm_task( const int n, const double *v, double* r );

void compute_beta(const double *rho, const double *old_rho, double *beta);
void compute_alpha(double *rho, double *normA_p_sq, double *old_rho, double *err_sq, double *old_err_sq, double *alpha);

#endif // CG_H_INCLUDED
