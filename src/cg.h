#ifndef CG_H_INCLUDED
#define CG_H_INCLUDED

#include "global.h"
#include "matrix.h"

typedef struct checkpoint_data
{
	int instructions;
	#if CKPT == CKPT_IN_MEMORY
	double *save_x, *save_p, *save_err_sq, *save_alpha;
	#elif CKPT == CKPT_TO_DISK
	double *save_err_sq, *save_alpha;
	const char *checkpoint_path;
	#endif
} checkpoint_data;

// structure to hold the pointers and get them when needed
typedef struct magic_pointers
{
	const Matrix *A;
	const double *b;
	double *x, *p, *old_p, *g, *Ap, *Ax;
	double *alpha, *beta, *err_sq, *old_err_sq, *old_err_sq2, *normA_p_sq;
	#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
	double *shared_page_reductions;
	#endif
	#if CKPT
	checkpoint_data *ckpt_data;
	#endif
	int n;
} magic_pointers;

// define a X-macro to associate the name of a variable in magic_pointers to its constant
#define ASSOC_CONST_MP \
	X(VECT_ITERATE,    x) \
	X(VECT_A_ITERATE,  Ax) \
	X(VECT_GRADIENT,   g) \
	X(VECT_P,          p) \
	X(VECT_OLD_P,      old_p) \
	X(VECT_A_P,        Ap)

// vects 1-8
#define VECT_ITERATE   1
#define VECT_A_ITERATE 2
#define VECT_GRADIENT  3
#define VECT_P         4
#define VECT_OLD_P     5
#define VECT_A_P       6

// vect copies for checkpoints 9-16
#define SAVE_ITERATE   VECT_ITERATE   +8
#define SAVE_GRADIENT  VECT_GRADIENT  +8
#define SAVE_P         VECT_P         +8

// tasks needed: both reductions, recovery, etc. 17-24
#define NORM_GRADIENT  17
#define NORM_A_P       18
#define RECOVERY       20
#define CHECKPOINT     21

void solve_cg(const Matrix *A, const double *b, double *iterate, double convergence_thres);

// all the algorithmical steps of CG that will be subdivided into tasks :
void update_gradient(double *gradient, double *Ap, double *alpha);
void recompute_gradient_mvm(const Matrix *A, double *iterate, double *Aiterate);
void recompute_gradient_update(double *gradient, double *Aiterate, const double *b);
void update_p(double *p, double *old_p, double *gradient, double *beta);
void update_iterate(double *iterate, double *p, double *alpha);
void compute_Ap(const Matrix *A, double *p, double *Ap);

void scalar_product_task(const double *p, const double *Ap, double* r);
void norm_task(const double *v, double* r);

void compute_beta(double *err_sq, const double *old_err_sq, double *beta, double *old_p, int recomputed);
void compute_alpha(double *err_sq, double *normA_p_sq, double *old_err_sq, double *old_err_sq2, double *alpha, double *p, double *old_p);

void force_rollback(checkpoint_data *ckpt_data, double *iterate, double *p);
void force_checkpoint(checkpoint_data *ckpt_data, double *iterate, double *p);
void due_checkpoint(checkpoint_data *ckpt_data, double *iterate, double *p);
void checkpoint_vectors(checkpoint_data *ckpt_data, int *behaviour, double *iterate, double *p);

void recover_rectify_xk(const int n, magic_pointers *mp, double *x);
void recover_rectify_g(const int n, magic_pointers *mp, const double *p, double *Ap, double *gradient, double *err_sq);
void recover_rectify_x_g(const int n, magic_pointers *mp, double *x, double *gradient, double *err_sq);
void recover_rectify_p_Ap(const int n, magic_pointers *mp, double *p, double *old_p, double *Ap, double *normA_p_sq);

#endif // CG_H_INCLUDED
