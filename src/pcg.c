#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"
#include "counters.h"
#include "csparse.h"

#include "pcg.h"

magic_pointers mp;

#if DUE && DUE != DUE_ROLLBACK
#include "pcg_resilient_tasks.c"
#include "pcg_recovery_tasks.c"
#else
#include "pcg_normal_tasks.c"
#endif

#if CKPT
#include "pcg_checkpoint.c"
#endif

void deallocate_preconditioner(Precond *M, char **wait_for_precond)
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

	for(i=0; i<nb_blocks; i++)
		free(wait_for_precond[i]);
}

void allocate_preconditioner(Precond **M, const int maxblocks, char **wait_for_precond)
{
	*M = (Precond*)malloc( sizeof(Precond) );
	(*M)->N = (csn**)malloc(maxblocks * sizeof(csn*));
	(*M)->S = (css**)malloc(maxblocks * sizeof(css*));

	int i;
	for(i=0; i<nb_blocks; i++)
		wait_for_precond[i] = alloc_deptoken();
}

void factorize_jacobiblock(const int block, const Matrix *A, css **S, csn **N)
{
	const int page_bytes = get_failblock_size() * sizeof(double);
	Matrix sm;

	int fbs = get_failblock_size(), pos = block * fbs, max = pos + fbs;
	if( max > A->n )
	{
		max = A->n;
		fbs = A->n - pos;
	}

	int nnz = A->r[ max ] - A->r[ max-fbs ];

	allocate_matrix( fbs, fbs, nnz, &sm, page_bytes);

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

void make_blockedjacobi_preconditioner(Precond *M, const Matrix *A, char **wait_for_precond UNUSED)
{
	int i, j, n = A->n, fbs = get_failblock_size(), log2fbs = get_log2_failblock_size();
	
	if( get_block_end(0) >> log2fbs > nb_blocks ) 
		// biggish blocks, need to make them smaller for more load balance - however needs to be a partition of the parallel blocks
		for(i=0; i < nb_blocks; i ++ )
		{
			int s = get_block_start(i), e = get_block_end(i), bs, be, k, sub_block;
			if( e > n )
				e = n;

			bs = s >> log2fbs;
			be = ((e + fbs - 1) >> log2fbs) - 1;
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
					factorize_jacobiblock(k, A, M->S + k, M->N + k );
			}
		}
	else
		for(i=0; i < nb_blocks; i ++ )
		{
			int s = get_block_start(i), e = get_block_end(i), bs, be;
			if( e > n )
				e = n;
			bs = s >> log2fbs;
			be = ((e + fbs - 1) >> log2fbs) - 1;

			#pragma omp task out(M->S[bs:be], M->N[bs:be], *(wait_for_precond[i])) firstprivate(i, bs, be) private(j) label(cholesky_diag_blocks) priority(10) no_copy_deps
			for(j=bs; j <= be; j++)
				factorize_jacobiblock(j, A, M->S + j, M->N + j );
		}
}

#pragma omp task in(*rho, *old_rho) out(*beta) label(compute_beta) priority(100) no_copy_deps
void compute_beta(const double *rho, const double *old_rho, double *beta)
{
	// on first iterations of a (re)start, old_rho should be INFINITY so that beta = 0
	*beta = *rho / *old_rho;

	#if DUE
	int state = aggregate_skips();
	if( state & (MASK_GRADIENT | MASK_NORM_G | MASK_RECOVERY) )
		fprintf(stderr, "ERROR SUBSISTED PAST RECOVERY restart needed. At beta, g:%d, ||g||:%d\n", (state & MASK_GRADIENT) > 0, (state & MASK_NORM_G) > 0);
	#endif

	log_err(SHOW_TASKINFO, "Computing beta finished : rho = %e ; old_rho = %e ; beta = %e\n", *rho, *old_rho, *beta);
}

#pragma omp task inout(*normA_p_sq, *rho) out(*alpha, *old_rho, *wait_for_alpha) label(compute_alpha) priority(100) no_copy_deps
void compute_alpha(double *normA_p_sq, double *rho, double *old_rho, double *alpha, char *wait_for_alpha UNUSED)
{
	*alpha = *rho / *normA_p_sq ;
	*old_rho = *rho;

	#if DUE
	int state = aggregate_skips();
	#if DUE == DUE_LOSSY
	if( state )
	{
		log_err(SHOW_FAILINFO, "There was an error, restarting (eventual lossy x interpolation)");
		hard_reset(&mp);
	}
	#else
	if( state & (MASK_ITERATE | MASK_P | MASK_OLD_P | MASK_A_P | MASK_NORM_A_P | MASK_RECOVERY) )
		fprintf(stderr, "ERROR SUBSISTED PAST RECOVERY restart needed. At alpha, x:%d, p:%d, p':%d, Ap:%d, <p,Ap>:%d\n", (state & MASK_ITERATE) > 0, (state & MASK_P) > 0, (state & MASK_OLD_P) > 0, (state & MASK_A_P) > 0, (state & MASK_NORM_A_P) > 0);
	#endif
	#endif

	log_err(SHOW_TASKINFO, "Computing alpha finished : normA_p_sq = %+e ; rho = %e ; alpha = %e\n", *normA_p_sq, *rho, *alpha);

	// last consumer of these values : let's 0 them so the scalar product doesn't need to
	*rho = 0.0;
	*normA_p_sq = 0.0;
}

static inline void swap(double **v, double **w)
{
	double *swap = *v;
	*v = *w;
	*w = swap;
}

void solve_pcg(const Matrix *A, const double *b, double *iterate, double convergence_thres)
{
	// do some memory allocations
	double norm_b, thres_sq;
	const int n = A->n;
	int r = -1, total_failures = 0, failures = 0;
	int do_update_gradient = 0;
	double *p, *old_p, *Ap, normA_p_sq, *gradient, *z, *Aiterate, rho = 0.0, old_rho = INFINITY, err_sq = 0.0, old_err_sq = DBL_MAX, alpha, beta = 0.0;
	char *wait_for_p = alloc_deptoken(), *wait_for_iterate = alloc_deptoken(), *wait_for_mvm = alloc_deptoken(), *wait_for_alpha = alloc_deptoken(), *wait_for_precond[nb_blocks];
	
	Precond *M;
	allocate_preconditioner(&M, get_nb_failblocks(), wait_for_precond);

	p        = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	old_p    = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	Ap       = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	gradient = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	z        = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	Aiterate = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));

	#if CKPT == CKPT_IN_MEMORY
	save_it  = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	save_g   = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	save_p   = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	#endif

	// some parameters pre-computed, and show some informations
	norm_b = norm(n, b);
	thres_sq = convergence_thres * convergence_thres * norm_b;
	log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);

	mp = (magic_pointers){.A = A, .M = M, .b = b, .x = iterate, .p = p, .old_p = old_p, .g = gradient, .z = z, .Ap = Ap, .Ax = Aiterate,
							.alpha = &alpha, .beta = &beta, .err_sq = &err_sq, .rho = &rho, .old_rho = &old_rho, .normA_p_sq = &normA_p_sq};
	#if CKPT
	detect_error_data err_data = (detect_error_data) {
		.save_rho = &save_rho, .save_alpha = &save_alpha,
		#if CKPT == CKPT_IN_MEMORY
		.save_x = save_it, .save_g = save_g, .save_p = save_p, .save_rho = &save_rho, .save_alpha = &save_alpha
		#endif
	};
	mp.err_data = err_data;
	#endif
	

	setup_resilience(A, 7, &mp);
	start_measure();

	// real work starts here

	make_blockedjacobi_preconditioner(M, A, wait_for_precond);

	for(r=0; r < MAXIT ; r++)
	{
		if( --do_update_gradient > 0 )
		{
			update_gradient(gradient, Ap, &alpha, wait_for_iterate);

			apply_preconditioner(gradient, z, M, wait_for_precond);

			scalar_product_task(gradient, z, &rho, RHO);

			#if DUE
			recover_rectify_g_z(n, &mp, old_p, Ap, gradient, z, &err_sq, &rho, wait_for_iterate);
			#endif

			compute_beta(&rho, &old_rho, &beta);

			update_iterate(iterate, wait_for_iterate, old_p, &alpha);
			#if DUE
			recover_rectify_xk(n, &mp, iterate, wait_for_iterate);
			#endif
		}
		else
		{
			if( r > 0 )
				update_iterate(iterate, wait_for_iterate, old_p, &alpha);

			recompute_gradient(gradient, A, iterate, wait_for_iterate, wait_for_mvm, Aiterate, b);

			apply_preconditioner(gradient, z, M, wait_for_precond);

			scalar_product_task(gradient, z, &rho, RHO);

			#if DUE
			recover_rectify_x_g_z(n, &mp, iterate, gradient, z, &err_sq, &rho, wait_for_mvm);
			#endif

			compute_beta(&rho, &old_rho, &beta);
		}

		update_p(p, old_p, wait_for_p, z, &beta);

		compute_Ap(A, p, wait_for_p, wait_for_mvm, Ap);

		scalar_product_task(p, Ap, &normA_p_sq, NORM_A_P);

		#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
		recover_rectify_p_Ap(n, &mp, p, old_p, Ap, &normA_p_sq, wait_for_mvm);
		#endif

		// when reaching this point, all tasks of loop should be created.
		// then waiting start : should be released halfway through the loop.
		// We want this to be after alpha on normal iterations, after AxIt
		// but it should not wait for recovery to finish on recovery iterations
		if( !do_update_gradient )
		{
			#pragma omp taskwait on(*wait_for_alpha, *wait_for_iterate)
		}
		else
		{
			#pragma omp taskwait on(*wait_for_alpha)
		}

		// swapping p's so we reduce pressure on the execution of the update iterate tasks
		// now output-dependencies is not conflicting with the next iteration but the one after
		{
			swap(&p, &old_p);
			
			failures = check_errors_signaled();

			if( r > 0 )
				log_convergence(r-1, old_err_sq, failures);

			log_err(SHOW_TASKINFO, "\n\n");

			total_failures += failures;

			if( old_err_sq <= thres_sq )
				break;

			if( do_update_gradient <= 0 )
				do_update_gradient = RECOMPUTE_GRADIENT_FREQ;
		}

		norm_task(gradient, &err_sq);

		#pragma omp task in(err_sq) out(old_err_sq) label(reset_err) priority(100) no_copy_deps
		{
			old_err_sq = err_sq;
			err_sq = 0.0;
		}

		compute_alpha(&normA_p_sq, &rho, &old_rho, &alpha, wait_for_alpha);
	}

	#pragma omp taskwait 
	// end of the math, showing infos
	stop_measure();
	
	failures = check_errors_signaled();
	log_convergence(r-1, old_err_sq, failures);

	printf("CG method finished iterations:%d with error:%e (failures:%d)\n", r, sqrt((err_sq==0.0?old_err_sq:err_sq)/norm_b), total_failures);

	// stop resilience stuff that's still going on
	unset_resilience();

	free(p);
	free(old_p);
	free(z);
	free(Ap);
	free(gradient);
	free(Aiterate);
	free(wait_for_p);
	free(wait_for_mvm);
	free(wait_for_alpha);
	free(wait_for_iterate);
	deallocate_preconditioner(M, wait_for_precond);
}

