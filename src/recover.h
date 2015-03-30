#ifndef RECOVER_H_INCLUDED
#define RECOVER_H_INCLUDED

#include "matrix.h"   // (Dense|Sparse)Matrix
#include "cg.h"       // magic_pointers

// before calling those, make sure that all the lost elements of x have been replaced
// either by their initial guess (uncorrelated) or by 0 (decorellated)
// (only for multiple faults with local strategies)
void get_rhs(const int n, const int *rows, const int m, const int *except_cols, const int bs, const Matrix *A, const double *b, const double *g, const double *x, double *rhs);

// actual work for recover_inverse done here, separated for all sets of disjoint neighbouring errors
void cluster_neighbour_failures(const Matrix *A, const double *b, double *x, int *lost_blocks, const int nb_lost, int *recovery_sizes);
void do_interpolation(const Matrix *A, const double *b, const double *g, double *x, const int *lost_blocks, const int nb_lost);

// recovering x using b - g = A * x (g may be NULL then b = A * x, e.g. use for Ap = A * p )
void recover_inverse(const Matrix *A, const double *b, const double *g, double *x, int *lost_blocks, const int nb_lost);

// w = v + sgn * ( A u ) 
void recover_direct(const Matrix *A, const int sgn, const double *u, const double *v, double *w, int lost_block);


// aliases of interpolations for situations, first come for single block errors ; different kinds :
// - partial matrix-vector multiplications
int recover_g_recompute(magic_pointers *mp, double *g, int block);
int recover_Ap(magic_pointers *mp, double *Ap, const double *p, int block);
int recover_Ax(magic_pointers *mp, double *Ax, int block);
// - daxpy's
int recover_g_update(magic_pointers *mp, double *g, int block);
int recover_p_repeat(magic_pointers *mp, double *p, const double *old_p, int block);

// for recovery/rectification tasks, when repairing vectors fully is needed
int recover_x_lossy          (magic_pointers *mp, double *x);
int recover_full_xk          (magic_pointers *mp, double *x,                      const int mark_clean);
int recover_full_p_repeat    (magic_pointers *mp, double *p, const double *old_p, const int mark_clean);
int recover_full_p_invert    (magic_pointers *mp, double *p,                      const int mark_clean);
int recover_full_g_recompute (magic_pointers *mp, double *g,                      const int mark_clean);
int recover_full_g_update    (magic_pointers *mp, double *g,                      const int mark_clean);
int recover_mvm_skips_g      (magic_pointers *mp, double *g,                      const int mark_clean);
int recover_full_Ap          (magic_pointers *mp, double *Ap, const double *p,    const int mark_clean);
int recover_full_old_p_invert(magic_pointers *mp, double *old_p,                  const int mark_clean);

void save_oldAp_for_old_p_recovery(magic_pointers *mp, double *old_p, const int s, const int e);

#endif // RECOVER_H_INCLUDED

