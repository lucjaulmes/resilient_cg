#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "global.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"
#include "counters.h"
#include "csparse.h"

#include "pcg.h"

void make_blockedjacobi_preconditioner( Precond *M, const Matrix *A )
{
	allocate_preconditioner(M, get_nb_failblocks());

	int i;
	for(i=0; i<get_nb_failblocks(); i++)
		factorize_jacobiblock( A->n, i, A, M->S + i, M->N + i );
}

void factorize_jacobiblock( const int n, const int block, const Matrix *A, css **S, csn **N )
{
	int fbs = get_failblock_size();
	Matrix sm;

	int pos = block * fbs, max = pos + fbs;
	if( max > n )
	{
		max = n;
		fbs = n - pos;
	}
	#ifndef MATRIX_DENSE
	int nnz = A->r[ max ] - A->r[ max-fbs ];
	#endif

	allocate_matrix( fbs, fbs, nnz, &sm );

	// get the submatrix for those lines
	get_submatrix(A, &pos, 1, &pos, 1, fbs, &sm);

	// from csparse
	cs submatrix;
	submatrix.m = sm.m ;
	submatrix.n = sm.n ;
	submatrix.nzmax = sm.nnz ;
	submatrix.nz = -1 ;
	// don't work with triplets, so has to be compressed column even though we work with compressed row
	// but since here the matrix is symmetric they are interchangeable
	submatrix.p = sm.r;
	submatrix.i = sm.c;
	submatrix.x = sm.v;

	*S = cs_schol (&submatrix, 0) ; /* ordering and symbolic analysis */
	*N = cs_chol (&submatrix, *S) ; /* numeric Cholesky factorization */

	deallocate_matrix(&sm);
}

void apply_preconditioner(const int n, const double *g, double *z, Precond *M)
{
	int i, fbs = get_failblock_size(), pos = 0;

	double *x = malloc( fbs * sizeof(double) );

	for(i = 0; i<get_nb_failblocks(); i++)
	{
		if( pos + fbs > n )
			fbs = n - pos;

		cs_ipvec (fbs, M->S[i]->Pinv, &g[pos], x) ;	/* x = P*b */
		cs_lsolve (M->N[i]->L, x) ;		/* x = L\x */
		cs_ltsolve (M->N[i]->L, x) ;		/* x = L'\x */
		cs_pvec (fbs, M->S[i]->Pinv, x, &z[pos]) ;	/* b = P'*x */

		pos += fbs;
	}
	// assert ( pos == n )
	free(x);
}

void deallocate_preconditioner(Precond *M)
{
	int i;
	for(i=0; i<get_nb_failblocks(); i++)
	{
		cs_sfree(M->S[i]);
		cs_nfree(M->N[i]);
	}
	free(M->N);
	free(M->S);
}

void allocate_preconditioner(Precond *M, const int maxblocks)
{
	M->N = (csn**)malloc(maxblocks * sizeof(csn*));
	M->S = (css**)malloc(maxblocks * sizeof(css*));
}

void solve_pcg( const int n, const Matrix *A, const double *b, double *iterate, double thres )
{
	// do some memory allocations
	double norm_b, thres_sq;
	int i, r, failures, total_failures = 0, update_gradient = 0;
	double *p, *Ap, normA_p_sq, *gradient, *z, err_sq, rho, old_rho = INFINITY, alpha, beta = 0.0;

	Precond M;
	make_blockedjacobi_preconditioner(&M, A);

	p = (double*)calloc( n, sizeof(double) );
	z = (double*)calloc( n, sizeof(double) );
	Ap = (double*)malloc( n * sizeof(double) );
	gradient = (double*)malloc( n * sizeof(double) );

	// some parameters pre-computed, and show some informations
	norm_b = scalar_product(n, b, b);
	thres_sq = thres * thres * norm_b;
	log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);

	printf("%d blocks of size %d for preconditioning\n", get_nb_failblocks(), get_failblock_size());

	// real work starts here
	start_measure();
	for(r=0; r < (MAGIC_MAXITERATION <= 0 ? 100*n : MAGIC_MAXITERATION); r++)
	{
		start_iteration();

		// update gradient to solution (a.k.a. error) : b - A * it
		// every now and then, recompute properly to remove rounding errors
		// NB. do this on first iteration where alpha Ap and gradient aren't defined
		if( update_gradient-- )
			daxpy(n, -alpha, Ap, gradient, gradient);
		else
		{
			mult(A, iterate, gradient);
			daxpy(n, -1.0, gradient, b, gradient);

			// do this computation, say, every 50 iterations
			update_gradient = 50 -1;
		}

		err_sq = scalar_product(n, gradient, gradient);

		apply_preconditioner(n, gradient, z, &M);

		rho = scalar_product(n, gradient, z);

		// we've got the gradient to get next direction (= error vector)
		// make it orthogonal to the last direction (it already is to all the previous ones)

		// Initially beta is 0 and p unimportant : p <- gradient (on (re)starts, old_rho = +infinity)
		if( old_rho == 0.0 )
			beta = scalar_product(n, gradient, Ap) / normA_p_sq;
		else
			beta = rho / old_rho;

		//daxpy(n, beta, p, gradient, p);
		daxpy(n, beta, p, z, p);

		// store A*p_r
		mult(A, p, Ap);

		// get the norm for A of this new direction vector
		normA_p_sq = scalar_product(n, p, Ap);

		alpha = rho / normA_p_sq ;

		log_err(FULL_VERBOSE, "||p_%d||_A = %e ; alpha = %e \n", r, normA_p_sq, alpha );

		// update iterate with contribution along new direction
		daxpy(n, alpha, p, iterate, iterate);

		old_rho = rho;

		// this to break when we have errors
		stop_iteration();

		if( r == MAGIC_ITERATION && 0)
		{
			for(i=0; i<MAGIC_NBSIMULTANEOUSFAULTS; i++)
				report_failed_block(MAGIC_BLOCKTORECOVER+i);
			update_gradient = 0;
		}

		failures = get_nb_failed_blocks();

		log_out("%4d, % e %d\n", r, err_sq, failures );

		if ( failures > 0 )
		{
			total_failures += failures;

			// do the recovery here
			// recover by interpolation since our submatrix is always spd

			// this recover implies a full restart
			if( failures > 1 )
			{
				recover( A, b, iterate, &M, fault_strat );
				old_rho = INFINITY;
				update_gradient = 0;
			}
			else
				// this one doesn't
				recover_xk( A, b, gradient, iterate, &M, fault_strat );

			#if VERBOSE > SHOW_FAILINFO
				mult(A, iterate, gradient);
				daxpy(n, -1.0, gradient, b, gradient);

				log_err(SHOW_FAILINFO, "Recovered : moved from % 1.2e to % 1.2e\n", err_sq, scalar_product(n, gradient, gradient));
			#endif
		}
		if (err_sq <= thres_sq)
			break;
	}

	// end of the math, showing infos
	stop_measure();

	log_out("\n\n------\nConverged at rank %d\n------\n\n", r);
	printf("CG method finished iterations:%d with error:%e (failures:%d)\n", r, sqrt(err_sq/norm_b), total_failures);

	free(p);
	free(z);
	free(Ap);
	free(gradient);
	deallocate_preconditioner(&M);
}

