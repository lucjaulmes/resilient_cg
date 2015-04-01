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

// structure used to compactly convey the data for abft
typedef struct checkpoint_data 
{
	int instructions;
	#if CKPT == CKPT_IN_MEMORY
	double *save_x, *save_g, *save_p, *save_rho, *save_alpha;
	#elif CKPT == CKPT_TO_DISK
	double *save_rho, *save_alpha;
	const char *checkpoint_path;
	#endif
} checkpoint_data;

// structure to hold the pointers and get them when needed
typedef struct magic_pointers
{
	const Matrix *A;
	Precond *M;
	const double *b;
	double *x, *p, *old_p, *g, *Ap, *Ax, *z;
	double *alpha, *beta, *err_sq, *rho, *old_rho, *normA_p_sq;
	#if CKPT
	checkpoint_data *ckpt_data;
	#endif
} magic_pointers;

// define a X-macro to associate the name of a variable in magic_pointers to its constant
#define ASSOC_CONST_MP \
	X(VECT_ITERATE,    x) \
	X(VECT_A_ITERATE,  Ax) \
	X(VECT_GRADIENT,   g) \
	X(VECT_P,          p) \
	X(VECT_OLD_P,      old_p) \
	X(VECT_A_P,        Ap) \
	X(VECT_Z,          z)

// vects 1-8
#define VECT_ITERATE   1
#define VECT_A_ITERATE 2
#define VECT_GRADIENT  3
#define VECT_P         4
#define VECT_OLD_P     5
#define VECT_A_P       6
#define VECT_Z         7

// vect copies for checkpoints 9-16
#define SAVE_ITERATE   VECT_ITERATE   +8
#define SAVE_GRADIENT  VECT_GRADIENT  +8
#define SAVE_P         VECT_P         +8
#define SAVE_A_P       VECT_A_P       +8

// tasks needed, but these are for both reductions that we have 17-24
#define NORM_GRADIENT  17
#define NORM_A_P       18
#define RHO            19
#define RECOVERY       20

void solve_pcg(const Matrix *A, const double *b, double *iterate, double convergence_thres);

void allocate_preconditioner(Precond **M, const int maxblocks, char **wait_for_precond);
void make_blockedjacobi_preconditioner(Precond *M, const Matrix *A, char **wait_for_precond);
void deallocate_preconditioner(Precond *M, char **wait_for_precond);

void factorize_jacobiblock(const int block, const Matrix *A, css **S, csn **N);
void apply_preconditioner(const double *g, double *z, Precond *M, char **wait_for_precond);

// all the algorithmical steps of CG that will be subdivided into tasks : 
void update_gradient(double *gradient, double *Ap, double *alpha, char *wait_for_iterate);
void recompute_gradient(double *gradient, const Matrix *A, double *iterate, char *wait_for_iterate , char *wait_for_mvm , double *Aiterate, const double *b);
void update_p(double *p, double *old_p, char *wait_for_p, double *gradient, double *beta);
void update_iterate(double *iterate, char *wait_for_iterate, double *p, double *alpha);
void compute_Ap(const Matrix *A, double *p, char *wait_for_p, char *wait_for_mvm, double *Ap);

void scalar_product_task(const double *u, const double *v, double* r, const int task_name);
void norm_task(const double *v, double* r);

void compute_beta(const double *rho, const double *old_rho, double *beta);
void compute_alpha(double *normA_p_sq, double *rho, double *old_rho, double *alpha, char *wait_for_alpha);

void force_checkpoint(checkpoint_data *ckpt_data, double *iterate, double *gradient, double *p, double *Ap);
void due_checkpoint(checkpoint_data *ckpt_data, double *iterate, double *gradient, double *p, double *Ap);
void force_rollback(checkpoint_data *ckpt_data, double *iterate, double *gradient, double *p, double *Ap);
void checkpoint_vectors(checkpoint_data *ckpt_data, int *behaviour, double *iterate, double *gradient, double *p, double *Ap);

void recover_rectify_xk(const int n, magic_pointers *mp, double *x, char *wait_for_iterate);
void recover_rectify_g_z(const int n, magic_pointers *mp, const double *p, double *Ap, double *gradient, double *z, double *err_sq, double *rho, char *wait_for_iterate);
void recover_rectify_x_g_z(const int n, magic_pointers *mp, double *x, double *gradient, double *z, double *err_sq, double *rho, char *wait_for_mvm);
void recover_rectify_p_Ap(const int n, magic_pointers *mp, double *p, double *old_p, double *Ap, double *normA_p_sq, char *wait_for_mvm);

#endif // CG_H_INCLUDED
