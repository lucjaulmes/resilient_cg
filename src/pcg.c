#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"
#include "counters.h"

#include "pcg.h"

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
	free(M);
}

void allocate_preconditioner(Precond **M, const int maxblocks)
{
	*M = (Precond*)malloc( sizeof(Precond) );
	(*M)->N = (csn**)malloc(maxblocks * sizeof(csn*));
	(*M)->S = (css**)malloc(maxblocks * sizeof(css*));
}

void factorize_jacobiblock( const int n, const int block, const Matrix *A, css **S, csn **N)
{
	int fbs = get_failblock_size();
	Matrix sm;

	int pos = block * fbs, max = pos + fbs;
	if( max > n )
	{
		max = n;
		fbs = n - pos;
	}

	allocate_matrix( fbs, fbs, A->nnz, &sm );

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

void make_blockedjacobi_preconditioner(Precond *M, const Matrix *A, char **wait_for_precond)
{
	int i, j, n = A->n, fbs = get_failblock_size();
	
	if( get_block_end(0) / fbs > nb_blocks ) 
		// biggish blocks, need to make them smaller for more load balance - however needs to be a partition of the parallel blocks
		for(i=0; i < nb_blocks; i ++ )
		{
			int s = get_block_start(i), e = get_block_end(i), bs, be, k, sub_block;
			if( e > n )
				e = n;

			bs = s / fbs;
			be = ((e + fbs - 1) / fbs) - 1;
			sub_block = (be + 1 - bs + nb_blocks/2) / nb_blocks;

			if(sub_block < 1)
				sub_block = 1;

			for(j=bs; j <= be; j+=sub_block)
			{
				int l = j + sub_block;
				if( l > be )
					l = be;

				#pragma omp task out(M->S[j:l], M->N[j:l]) concurrent(*(wait_for_precond[i])) firstprivate(i, j, l) private(k) label(cholesky_diag_blocks) priority(10) no_copy_deps
				for(k=j; k <= l; k++)
					factorize_jacobiblock( A->n, k, A, M->S + k, M->N + k );
			}
		}
	else
		for(i=0; i < nb_blocks; i ++ )
		{
			int s = get_block_start(i), e = get_block_end(i), bs, be;
			if( e > n )
				e = n;
			bs = s / fbs;
			be = ((e + fbs - 1) / fbs) - 1;

			#pragma omp task out(M->S[bs:be], M->N[bs:be], *(wait_for_precond[i])) firstprivate(i, bs, be) private(j) label(cholesky_diag_blocks) priority(10) no_copy_deps
			for(j=bs; j <= be; j++)
				factorize_jacobiblock( A->n, j, A, M->S + j, M->N + j );
		}
}

void apply_preconditioner(const int n, const double *g, double *z, Precond *M, char **wait_for_precond)
{
	int b, s, e, bs, be, fbs = get_failblock_size();

	for(b=0; b < nb_blocks; b ++ )
	{
		s = get_block_start(b);
		e = get_block_end(b);
		bs = s / fbs;
		be = (e + fbs - 1) / fbs;

		#pragma omp task in(g[s:e], M->S[bs:be], M->N[bs:be], *(wait_for_precond[b])) out(z[s:e]) firstprivate(b, s, e, bs, be, n) label(precondition) priority(50) no_copy_deps
		{
			double *x = malloc( fbs * sizeof(double) );

			for(; bs < be ; s += fbs, bs++)
			{
				if( s + fbs > n )
					fbs = n - s;

				cs_ipvec (fbs, M->S[bs]->Pinv, &g[s], x) ;	// x = P*g
				cs_lsolve (M->N[bs]->L, x) ;		// x = L\x
				cs_ltsolve (M->N[bs]->L, x) ;		// x = L'\x
				cs_pvec (fbs, M->S[bs]->Pinv, x, &z[s]) ;	// z = P'*x
			}
			
			free(x);

			// remove preconditioning :
			//for(i=s; i<e; i++) z[i] = g[i];
		}
	}	
}

void scalar_product_task(const int n, const double *v, const double *w, double* r)
{
	int i;
	/*
	// This is read by the main task, but r must not be set to 0 by it since some tasks consumer of r my not have been executed yet.
	// Let us rely on the program to send only 0 values into here. Alternatively, use the task below to generate proper anti-dependencies.
	#pragma omp task out(*r) label(prepare_dotp)
	{
		*r = 0;
	}
	*/

	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- <v, w>
		#pragma omp task in(v[s:e-1], w[s:e-1]) concurrent(*r) firstprivate(s, e) label(dotp) priority(10) no_copy_deps
		{
			double local_r = scalar_product( e-s, &(v[s]), &(w[s]) );

			#pragma omp atomic
				*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow scalar product <p, Ap> block %d finished = %e\n", i, local_r);
		}
	}
}

void norm_task( const int n, const double *v, double* r )
{
	int i;
	/*
	// This is read by the main task, but r must not be set to 0 by it since some tasks consumer of r my not have been executed yet.
	// Let us rely on the program to send only 0 values into here. Alternatively, use the task below to generate proper anti-dependencies.
	#pragma omp task out(*r) label(prepare_norm)
	{
		*r = 0;
	}
	*/

	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- || v ||
		#pragma omp task in(v[s:e-1]) concurrent(*r) firstprivate(s, e) label(norm) priority(0) no_copy_deps
		{
			double local_r = norm( e-s, &(v[s]) );

			#pragma omp atomic
			*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow square norm || g || part %d finished = %e\n", i, local_r);
		}
	}
}

void update_gradient(const int n, double *gradient, double *Ap, double *alpha)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		#pragma omp task in(*alpha, Ap[s:e-1]) inout(gradient[s:e-1]) firstprivate(s, e) label(update_gradient) priority(100) no_copy_deps
		{
			daxpy(e-s, -(*alpha), &(Ap[s]), &(gradient[s]), &(gradient[s]));
		}
	}
}

void recompute_gradient(const int n, double *gradient, const Matrix *A, double *iterate, char *wait_for_iterate, double *Aiterate, const double *b)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// Aiterate <- A * iterate
		#pragma omp task in(iterate[s:e-1], *wait_for_iterate) out(Aiterate[s:e-1]) firstprivate(s, e) label(AxIt) priority(10) no_copy_deps
		{
			Matrix local;
			local.m = A->m;
			local.n = e-s;

			local.r = & ( A->r[s] );
			local.v = A->v;
			local.c = A->c;

			mult(&local, iterate, &(Aiterate[s]) );
		}

		// gradient <- b - Aiterate
		#pragma omp task in(Aiterate[s:e-1]) out(gradient[s:e-1]) firstprivate(s, e) label(b-AxIt) priority(100) no_copy_deps
		{
			// there is not really any multiplication here
			//daxpy(e-s, -1, &(Aiterate[s]), &(b[s]), &(gradient[s]));
			int j;
			for (j=s; j<e; j++)
				gradient[j] = b[j] - Aiterate[j] ;
		}
	}
}

void update_p(const int n, double *p, double *old_p, char *wait_for_p, double *gradient, double *beta)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// p <- beta * old_p + gradient
		#pragma omp task in(*beta, gradient[s:e-1]) in(old_p[s:e-1]) out(p[s:e-1]) concurrent(*wait_for_p) firstprivate(s, e) label(update_p) priority(10) no_copy_deps
		{
			daxpy(e-s, *beta, &(old_p[s]), &(gradient[s]), &(p[s]));
		}
	}
}

void compute_Ap(const int n, const Matrix *A, double *p, char *wait_for_p, double *Ap)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// Ap <- A * p
		#pragma omp task in(p[s:e-1], *wait_for_p) out(Ap[s:e-1]) firstprivate(s, e) label(Axp) priority(20) no_copy_deps
		{
			Matrix local;
			local.m = A->m;
			local.n = e-s;

			local.r = & ( A->r[s] );
			local.v = A->v;
			local.c = A->c;

			mult(&local, p, &(Ap[s]) );
		}
	}
}

void update_iterate(const int n, double *iterate, char *wait_for_iterate, double *p, double *alpha)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// iterate <- iterate - alpha * p
		#pragma omp task in(*alpha, p[s:e-1]) inout(iterate[s:e-1]) concurrent(*wait_for_iterate) firstprivate(s, e) label(update_iterate) priority(0) no_copy_deps
		{
			daxpy(e-s, *alpha, &(p[s]), &(iterate[s]), &(iterate[s]));
		}
	}
}

#pragma omp task in(*rho, *old_rho) out(*beta) label(compute_beta) priority(100) no_copy_deps
void compute_beta(const double *rho, const double *old_rho, double *beta)
{
	// on first iterations of a (re)start, old_rho should be INFINITY so that beta = 0
	*beta = *rho / *old_rho;

	log_err(SHOW_TASKINFO, "Computing beta finished : rho = %e ; old_rho = %e ; beta = %e \n", *rho, *old_rho, *beta);
}

#pragma omp task inout(*normA_p_sq, *rho, *err_sq) out(*alpha, *old_rho, *old_err_sq) label(compute_alpha) priority(100) no_copy_deps
void compute_alpha(double *rho, double *normA_p_sq, double *old_rho, double *err_sq, double *old_err_sq, double *alpha)
{
	*alpha = *rho / *normA_p_sq ;
	*old_rho = *rho;
	*old_err_sq = *err_sq;

	log_err(SHOW_TASKINFO, "Computing alpha finished : normA_p_sq = %+e ; rho = %e ; alpha = %e\n", *normA_p_sq, *old_rho, *alpha);

	// last consumer of these values : let's 0 them so the scalar product doesn't need to
	*err_sq = 0.0;
	*rho = 0.0;
	*normA_p_sq = 0.0;
}

#pragma omp task in([n]gradient) inout([n]iterate, *wait_for_iterate) out(*failures) label(recovery) priority(100) no_copy_deps
void recover_task( const int r, const int n, const Matrix *A, const Precond *M, const double *b, const double *gradient, double *iterate, char *wait_for_iterate, int *failures )
{
	// debug replacement to have deterministic errors
	if((r % 50) == 25 && get_strategy() != NOFAULT )
	{
		int nb_errs = 0, start_errs, i;
		if( get_strategy() == SINGLEFAULT )
		{
			if( r > 25 )
			{
				nb_errs = 1;
				start_errs = (r - 25) / 50 - 1;
			}
		}
		// 0 at 25, 1 err at 75, 2 at 125 etc.
		else
		{
			nb_errs = (r - 25) / 50;
			start_errs = ((nb_errs * (nb_errs - 1)) / 2) % get_nb_failblocks();
		}

		for(i=0; i<nb_errs; i++)
			report_failure( (start_errs + i) % get_nb_failblocks() );

		if( nb_errs > 0 )
			printf("Iteration %d : Simulating %d errs starting at block %d\n", r, nb_errs, start_errs);

		if( nb_errs > 0 && start_errs < get_nb_failblocks() && (start_errs + i) > get_nb_failblocks() )
			fprintf(stderr, "iteration %d, had to wrap around to generate failures on %d consecutive blocks starting at %d (maxblocks is %d)\n",
				r, nb_errs, start_errs, get_nb_failblocks());
	}

	// normal function from here on
	//check_errors();

	int flb = get_nb_failed_blocks();

	if( flb > 0 )
	{
		*failures = flb;
		
		int id, lost[flb];
		// do recovery
		// recover by interpolation since our submatrix is always spd

		// fill lost with all the failed blocks
		for(id=0; id<flb; id++)
			lost[id] = pull_failed_block();

		//recover( A, b, iterate, M, get_strategy(), lost, flb );
		recover_xk( A, b, gradient, iterate, M, get_strategy(), lost, flb );

		log_err(SHOW_DBGINFO, "Recovered from fault.\n");
	}
}

void solve_pcg( const int n, const Matrix *A, Precond *M, const double *b, double *iterate, double thres )
{
	// do some memory allocations
	double norm_b, thres_sq;
	int i, r = -1, do_update_gradient = 0;
	double *p, *old_p, *z, *Ap, normA_p_sq, *gradient, *Aiterate, rho = 0.0, old_rho = INFINITY, err_sq = 0.0, old_err_sq = INFINITY, alpha, beta = 0.0;
	char *wait_for_p = alloc_deptoken(), *wait_for_iterate = alloc_deptoken(), *wait_for_precond[nb_blocks], alloc_M = 0;

	for(i=0; i<nb_blocks; i++)
		wait_for_precond[i] = alloc_deptoken();

	if( M == NULL )
	{
		allocate_preconditioner(&M, get_nb_failblocks());
		alloc_M++;
	}

	p = (double*)calloc( n, sizeof(double) );
	z = (double*)calloc( n, sizeof(double) );
	old_p = (double*)calloc( n, sizeof(double) );
	Ap = (double*)calloc( n, sizeof(double) );
	gradient = (double*)calloc( n, sizeof(double) );
	Aiterate = (double*)calloc( n, sizeof(double) );

	// some parameters pre-computed, and show some informations
	norm_b = scalar_product(n, b, b);
	thres_sq = thres * thres * norm_b;
	log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);


	if( alloc_M )
	{
		// real work starts here
		start_measure();
		make_blockedjacobi_preconditioner(M, A, wait_for_precond);
	}

	start_iterations();

	for(r=0; r < MAXIT ; r++)
	{

		if( do_update_gradient-- )
			update_gradient(n, gradient, Ap, &alpha);
		else
		{
			recompute_gradient(n, gradient, A, iterate, wait_for_iterate, Aiterate, b);
			// do this direct computation, say, every 50 iterations
			do_update_gradient = RECOMPUTE_GRADIENT_FREQ - 1;
		}

		norm_task(n, gradient, &err_sq);

		apply_preconditioner(n, gradient, z, M, wait_for_precond);

		scalar_product_task(n, gradient, z, &rho);

		compute_beta(&rho, &old_rho, &beta);

		update_p(n, p, old_p, wait_for_p, z, &beta);

		compute_Ap(n, A, p, wait_for_p, Ap);

		scalar_product_task(n, p, Ap, &normA_p_sq);

		// when reaching this point, all tasks of loop should be created.
		// then waiting start : should be released halfway through the loop.
		// We want this to be after alpha on normal iterations, after AxIt on recompute iterations
		// but it should not wait for recovery to finish on recovery iterations
		if( do_update_gradient == RECOMPUTE_GRADIENT_FREQ - 1 )
		{
			#pragma omp taskwait on(old_err_sq, *wait_for_iterate)
		}
		else
		{
			#pragma omp taskwait on(old_err_sq)
		}

		// swapping p's so we reduce pressure on the execution of the update_iterate tasks
		// now output-dependencies is not conflicting with the next iteration but the one after
		{
			double *swap_p = p;
			p = old_p;
			old_p = swap_p;
			
			if( r > 0 )
				log_convergence(r-1, old_err_sq, 0);

			if( old_err_sq <= thres_sq )
				break;
		}

		compute_alpha(&rho, &normA_p_sq, &old_rho, &err_sq, &old_err_sq, &alpha);
		
		update_iterate(n, iterate, wait_for_iterate, old_p, &alpha);
	}

	#pragma omp taskwait 
	// end of the math, showing infos
	if( alloc_M )
		stop_measure();
	
	err_sq = norm(n, gradient);
	log_convergence(r-1, err_sq, 0);

	log_out("\n\n------\nConverged at rank %d\n------\n\n", r);

	printf("PCG method finished iterations:%d with error:%e (failures:%d)\n", r, sqrt((err_sq==0.0?old_err_sq:err_sq)/norm_b), 0);


	free(p);
	free(z);
	free(old_p);
	free(Ap);
	free(gradient);
	free(Aiterate);
	free(wait_for_p);
	free(wait_for_iterate);
	if( alloc_M )
		deallocate_preconditioner(M);
	for(i=0;i<nb_blocks;i++)
		free(wait_for_precond[i]);
}

