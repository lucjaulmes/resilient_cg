#ifndef DEBUG_H_INCLUDED
#define DEBUG_H_INCLUDED

#include <stdarg.h>
#include "global.h"
#include "csparse.h"

// if we want to use several levels of verbosity
#define SHOW_DBGINFO  1
#define SHOW_FAILINFO 2
#define SHOW_TASKINFO 3
#define SHOW_TOOMUCH  4

// if we defined PERFORMANCE we are going to be very silent
// if we defined VERBOSE we are going to be selectively talkative

#ifndef PERFORMANCE

// this is all a bit manual, but necessary to remove side-effects of the VA_ARGS
// if we don't #define the unused log_err to {} (e.g. declaring to a function that does nothing)
// then the compiler will keep whatever's in the call for side-effects. BAD !
// i.e. log_err(SHOW_TOOMUCH, "%e\n", norm(vect)) will still spend time & FLOPS computing norm(vect) but never show it

#if VERBOSE >= 1
#define log_err_1(...) fprintf(stderr, __VA_ARGS__)
#endif

#if VERBOSE >= 2
#define log_err_2(...) fprintf(stderr, __VA_ARGS__)
#endif

#if VERBOSE >= 3
#define log_err_3(...) fprintf(stderr, __VA_ARGS__)
#endif

#if VERBOSE >= 4
#define log_err_4(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef log_err_1
#define log_err_1(...) {}
#endif

#ifndef log_err_2
#define log_err_2(...) {}
#endif

#ifndef log_err_3
#define log_err_3(...) {}
#endif

#ifndef log_err_4
#define log_err_4(...) {}
#endif


#define log_out(...) printf(__VA_ARGS__)

// and now refer to those nice log_err_X
#define PASTE(a,b) a ## b
#define log_err(level, ...) PASTE(log_err_, level)(__VA_ARGS__)

static inline void print_csparse_mat(cs *M)
{
	/*
	typedef struct cs_sparse    // matrix in compressed-column or triplet form
	{
	int nzmax ;	    // maximum number of entries
	int m ;	    // number of rows
	int n ;	    // number of columns
	int *p ;	    // column pointers (size n+1) or col indices (size nzmax)
	int *i ;	    // row indices, size nzmax
	double *x ;	    // numerical values, size nzmax
	int nz ;	    // # of entries in triplet matrix, -1 for compressed-col
	} cs ;
	*/
	if(M->nz < 0 ) { int i,j = 0; printf("\tp = ");
		for(i=0; i < M->nzmax && j <= M->m; i++) if(M->p[j] == i) {printf("%-8d ", j); j++;} else printf("         "); printf("\n\ti = ");
		for(i=0; i < M->p[M->m]; i++) printf("%-8d ", M->i[i]); printf("\n\tx = ");
		for(i=0; i < M->p[M->m]; i++) printf("%1.2e ", M->x[i]); printf("\n");
	} else { int i; printf("\tp = ");
		for(i=0; i < M->p[M->nz]; i++) printf("%-8d ", M->p[i]); printf("\n\ti = ");
		for(i=0; i < M->p[M->nz]; i++) printf("%-8d ", M->i[i]); printf("\n\tx = ");
		for(i=0; i < M->p[M->nz]; i++) printf("%1.2e ", M->x[i]); printf("\n"); }
}

static inline void print_jacobiblock(int n, css *S, csn *N)
{
	int i;
	/*
	   typedef struct cs_symbolic  // symbolic Cholesky, LU, or QR analysis
	   {
	   int *Pinv ;	    // inverse row perm. for QR, fill red. perm for Chol
	   int *Q ;	    // fill-reducing column permutation for LU and QR
	   int *parent ;   // elimination tree for Cholesky and QR
	   int *cp ;	    // column pointers for Cholesky, row counts for QR
	   int m2 ;	    // # of rows for QR, after adding fictitious rows
	   int lnz ;	    // # entries in L for LU or Cholesky; in V for QR
	   int unz ;	    // # entries in U for LU; in R for QR
	   } css ;
	 */
	printf("\nJACOBI PRECONDITIONING BLOCK #%d\n", n);
	printf("S : m2 = %d\n", S->m2);
	printf("    lnz = %d\n", S->lnz);
	printf("    unz = %d\n", S->unz);
	if( ! S->Pinv ) printf("    Pinv = NULL\n"); else { printf("    Pinv ="); for(i=0; i < S->lnz; i++)printf(" %2d", S->Pinv[i]); printf("\n"); }
	if( ! S->Q ) printf("    Q = NULL\n"); else { printf("    Q ="); for(i=0; i < S->lnz; i++)printf(" %2d", S->Q[i]); printf("\n"); }
	if( ! S->parent ) printf("    parent = NULL\n"); else { printf("    parent ="); for(i=0; i < S->lnz; i++)printf(" %2d", S->parent[i]); printf("\n"); }
	if( ! S->cp ) printf("    cp = NULL\n"); else { printf("    cp ="); for(i=0; i < S->lnz; i++)printf(" %2d", S->cp[i]); printf("\n"); }

	/*
	   typedef struct cs_numeric   // numeric Cholesky, LU, or QR factorization
	   {
	   cs *L ;	    // L for LU and Cholesky, V for QR
	   cs *U ;	    // U for LU, R for QR, not used for Cholesky
	   int *Pinv ;	    // partial pivoting for LU
	   double *B ;	    // beta [0..n-1] for QR
	   } csn ;
	 */
	if( ! N->Pinv ) printf("N : Pinv = NULL\n"); else { printf("N : Pinv ="); for(i=0; i < S->lnz; i++)printf(" %2d", N->Pinv[i]); printf("\n"); }
	if( ! N->B ) printf("    B = NULL\n"); else { printf("    B ="); for(i=0; i < S->lnz; i++)printf(" %1.2e", N->B[i]); printf("\n"); }
	if( ! N->L ) printf("    L = NULL\n"); else { printf("    L = {%d x %d : %d} fmt=%d\n", N->L->m, N->L->n, N->L->nzmax, N->L->nz);  print_csparse_mat(N->L); }
	if( ! N->U ) printf("    U = NULL\n"); else { printf("    U = {%d x %d : %d} fmt=%d\n", N->U->m, N->U->n, N->U->nzmax, N->U->nz);  print_csparse_mat(N->U); }
}
#else

#undef VERBOSE

#define log_out(...) {}
#define log_err(...) {}

#endif

#endif // DEBUG_H_INCLUDED

