#ifndef DEBUG_H_INCLUDED
#define DEBUG_H_INCLUDED

#include <stdarg.h>

// if we want to use several levels of verbosity
#define FULL_VERBOSE 5
#define LIGHT_VERBOSE 1
#define SHOW_DBGINFO 2
#define SHOW_FULLDBG 3
#define SHOW_FAILINFO 4
#define SHOW_TASKINFO 5

#ifdef PERFORMANCE
#undef VERBOSE
#endif

// if we defined PERFORMANCE we are going to be very silent
// if we defiend VERBOSE we are going to be very talkative

static inline void log_out(const char* fmt, ...)
{
	#ifndef PERFORMANCE
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
	#endif
}

static inline void log_err(const int level, const char* fmt, ...)
{
	#ifdef VERBOSE
	if( level >= VERBOSE )
		return;

	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
	#endif
}

#include "csparse.h"

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

#endif // DEBUG_H_INCLUDED

