#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"
#include "counters.h"

#include "cg.h"

magic_pointers mp;

#if DUE && DUE != DUE_ROLLBACK
#include "cg_resilient_tasks.c"
#include "cg_recovery_tasks.c"
#else
#include "cg_normal_tasks.c"
#endif

#if CKPT
#include "cg_checkpoint.c"
#endif

#pragma omp task in(*err_sq, *old_err_sq) out(*beta) label(compute_beta) priority(100) no_copy_deps
void compute_beta(const double *err_sq, const double *old_err_sq, double *beta)
{
	// on first iterations of a (re)start, old_err_sq should be INFINITY so that beta = 0
	*beta = *err_sq / *old_err_sq;

	#if DUE
	int state = aggregate_skips();
	if(state & (MASK_GRADIENT | MASK_NORM_G | MASK_RECOVERY))
	{
		clear_failed(MASK_GRADIENT | MASK_NORM_G | MASK_RECOVERY);
		fprintf(stderr, "ERROR SUBSISTED PAST RECOVERY restart needed. At beta, g:%d, ||g||:%d\n", (state & MASK_GRADIENT) > 0, (state & MASK_NORM_G) > 0);
	}
	#endif

	log_err(SHOW_TASKINFO, "Computing beta finished : err_sq = %e ; old_err_sq = %e ; beta = %e \n", *err_sq, *old_err_sq, *beta);
}

#pragma omp task inout(*normA_p_sq, *err_sq) out(*alpha, *old_err_sq, *old_err_sq2) label(compute_alpha) priority(100) no_copy_deps
void compute_alpha(double *err_sq, double *normA_p_sq, double *old_err_sq, double *old_err_sq2, double *alpha)
{
	*alpha = *err_sq / *normA_p_sq ;
	*old_err_sq = *err_sq;
	*old_err_sq2 = *err_sq;

	#if DUE
	int state = aggregate_skips();
	#if DUE == DUE_LOSSY
	if(state)
	{
		log_err(SHOW_FAILINFO, "There was an error, restarting (eventual lossy x interpolation)");
		hard_reset(&mp);
	}
	#else
	if(state & (MASK_ITERATE | MASK_P | MASK_OLD_P | MASK_A_P | MASK_NORM_A_P | MASK_RECOVERY))
	{
		clear_failed(MASK_ITERATE | MASK_P | MASK_OLD_P | MASK_A_P | MASK_NORM_A_P | MASK_RECOVERY);
		fprintf(stderr, "ERROR SUBSISTED PAST RECOVERY restart needed. At alpha, x:%d, p:%d, p':%d, Ap:%d, <p,Ap>:%d\n", (state & MASK_ITERATE) > 0, (state & MASK_P) > 0, (state & MASK_OLD_P) > 0, (state & MASK_A_P) > 0, (state & MASK_NORM_A_P) > 0);
	}
	#endif
	#endif

	log_err(SHOW_TASKINFO, "Computing alpha finished : normA_p_sq = %+e ; err_sq = %e ; alpha = %e\n", *normA_p_sq, *err_sq, *alpha);

	// last consumer of these values : let's 0 them so the scalar product doesn't need to
	*err_sq = 0.0;
	*normA_p_sq = 0.0;
}

static inline void swap(double **v, double **w)
{
	double *swap = *v;
	*v = *w;
	*w = swap;
}

void solve_cg(const Matrix *A, const double *b, double *iterate, double convergence_thres)
{
	// do some memory allocations
	double norm_b, thres_sq;
	const int n = A->n;
	int r = -1, total_failures = 0, failures = 0;
	int do_update_gradient = 0;
	double *p, *old_p, *Ap, normA_p_sq, *gradient, *Aiterate, err_sq = 0.0, old_err_sq = INFINITY, old_err_sq2 = DBL_MAX, alpha = 0.0, beta = 0.0;
	char *wait_for_p = alloc_deptoken(), *wait_for_iterate = alloc_deptoken(), *wait_for_mvm = alloc_deptoken();
	#if CKPT == CKPT_IN_MEMORY
	double *save_it, *save_g, *save_p, *save_Ap, save_err_sq, save_alpha;
	int do_checkpoint = 0;
	#elif CKPT == CKPT_TO_DISK
	double save_err_sq, save_alpha;
	int do_checkpoint = 0;
	#endif

	p        = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	old_p    = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	Ap       = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	gradient = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	Aiterate = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));

	#if CKPT == CKPT_IN_MEMORY
	save_it  = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	save_g   = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	save_p   = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	save_Ap  = (double*)aligned_calloc(sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	#endif

	// some parameters pre-computed, and show some informations
	norm_b = norm(n, b);
	thres_sq = convergence_thres * convergence_thres * norm_b;
	log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);

	mp = (magic_pointers){.A = A, .b = b, .x = iterate, .p = p, .old_p = old_p, .g = gradient, .Ap = Ap, .Ax = Aiterate,
							.alpha = &alpha, .beta = &beta, .err_sq = &err_sq, .old_err_sq = &old_err_sq, .normA_p_sq = &normA_p_sq};
	#if CKPT
	checkpoint_data ckpt_data = (checkpoint_data) {
		#if CKPT == CKPT_IN_MEMORY
		.save_x = save_it, .save_g = save_g, .save_p = save_p, .save_Ap = save_Ap,
		#endif
		.instructions = SAVE_CHECKPOINT, .save_err_sq = &save_err_sq, .save_alpha = &save_alpha
	};
	mp.ckpt_data = &ckpt_data;
	#endif

	setup_resilience(A, 6, &mp);
	start_measure();

	// real work starts here

	for(r=0; r < MAXIT ; r++)
	{
		if(--do_update_gradient > 0)
		{
			update_iterate(iterate, wait_for_iterate, old_p, &alpha);

			update_gradient(gradient, Ap, &alpha, wait_for_iterate);

			norm_task(gradient, &err_sq);

			// at this point, Ap = A * old_p
			#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
			recover_rectify_g(n, &mp, old_p, Ap, gradient, &err_sq, wait_for_iterate);
			#endif

			compute_beta(&err_sq, &old_err_sq, &beta);

			#if DUE == DUE_ASYNC
			recover_rectify_xk(n, &mp, iterate, wait_for_iterate);
			#endif
		}
		else
		{
			if(r > 0)
				update_iterate(iterate, wait_for_iterate, old_p, &alpha);

			recompute_gradient_mvm(A, iterate, wait_for_iterate, wait_for_mvm, Aiterate);

			recompute_gradient_update(gradient, wait_for_mvm, Aiterate, b);

			norm_task(gradient, &err_sq);

			#if CKPT
			if(r == 0)
				force_checkpoint(&ckpt_data, iterate, gradient, old_p, Ap);
			#endif

			#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
			recover_rectify_x_g(n, &mp, iterate, gradient, &err_sq, wait_for_mvm);
			#endif

			compute_beta(&err_sq, &old_err_sq, &beta);

			// after first beta, we are sure to have the first x, g, and checkpoint
			// so we can start injecting errors
			if(r == 0)
				#pragma omp task in(beta) label(start_injection)
				start_error_injection();
		}

		update_p(p, old_p, wait_for_p, gradient, &beta);

		compute_Ap(A, p, wait_for_p, wait_for_mvm, Ap);

		scalar_product_task(p, Ap, &normA_p_sq);

		#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
		recover_rectify_p_Ap(n, &mp, p, old_p, Ap, &normA_p_sq, wait_for_mvm, wait_for_iterate);
		#endif

		// when reaching this point, all tasks of loop should be created.
		// then waiting start : should be released halfway through the loop.
		// We want this to be after alpha on normal iterations, after AxIt
		// but it should not wait for recovery to finish on recovery iterations
		if(!do_update_gradient)
		{
			#pragma omp taskwait on(old_err_sq2, *wait_for_iterate)
		}
		else
		{
			#pragma omp taskwait on(old_err_sq2)
		}

		// swapping p's so we reduce pressure on the execution of the update iterate tasks
		// now output-dependencies is not conflicting with the next iteration but the one after
		{
			swap(&p, &old_p);
			
			failures = check_errors_signaled();

			if(r > 0)
				log_convergence(r-1, old_err_sq2, failures);

			log_err(SHOW_TASKINFO, "\n\n");

			total_failures += failures;

			if(old_err_sq <= thres_sq)
				break;

			if(do_update_gradient <= 0)
				do_update_gradient = RECOMPUTE_GRADIENT_FREQ;
			#if DUE == DUE_IN_PATH
			else
				recover_rectify_xk(n, &mp, iterate, (char*)&normA_p_sq);
			#endif
			#if CKPT
			if(do_checkpoint <= 0)
				do_checkpoint = CHECKPOINT_FREQ;
			#endif
		}

		compute_alpha(&err_sq, &normA_p_sq, &old_err_sq, &old_err_sq2, &alpha);

		// should happen after p, Ap are ready and before (post-alpha) iterate and gradient updates
		// so just after (or just before) alpha basically
		#if CKPT // NB. this implies DUE_ROLLBACK
		if(failures)
		{
			do_checkpoint = 0;
			force_rollback(&ckpt_data, iterate, gradient, old_p, Ap);
		}
		else if(--do_checkpoint == 0)
			due_checkpoint(&ckpt_data, iterate, gradient, old_p, Ap);
		#endif
	}

	#pragma omp taskwait 
	// end of the math, showing infos
	stop_measure();
	
	failures = check_errors_signaled();
	log_convergence(r-1, old_err_sq2, failures);

	printf("CG method finished iterations:%d with error:%e (failures:%d)\n", r, sqrt((err_sq==0.0?old_err_sq:err_sq)/norm_b), total_failures);

	// stop resilience stuff that's still going on
	unset_resilience();

	free(p);
	free(old_p);
	free(Ap);
	free(gradient);
	free(Aiterate);
	free(wait_for_p);
	free(wait_for_mvm);
	free(wait_for_iterate);

	#if CKPT == CKPT_IN_MEMORY
	free(save_it);
	free(save_g);
	free(save_p);
	free(save_Ap);
	#endif
}

