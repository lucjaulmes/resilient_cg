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

// too much work to repeat this, especially ALL_BLOCKS(...)
#define ALL_BLOCKS(__vector__) {__vector__[get_block_start(_blk):get_block_end(_blk)-1], _blk=0:nb_blocks-1}
#define PRAGMA(__str__) _Pragma(#__str__)
#define PRAGMA_TASK(__deps__, __lab__, __prio__) PRAGMA(omp task __deps__ label(__lab__) priority(__prio__) no_copy_deps)


#if DUE && DUE != DUE_ROLLBACK
#include "cg_resilient_tasks.c"
#include "cg_recovery_tasks.c"
#else
#include "cg_normal_tasks.c"
#endif

#if CKPT
#include "cg_checkpoint.c"
#endif

PRAGMA_TASK(in(*err_sq, *old_err_sq) out(*beta), compute_beta, 50)
void compute_beta(double *err_sq, const double *old_err_sq, double *beta, double *old_p UNUSED, int recomputed UNUSED)
{
	// on first iterations of a (re)start, old_err_sq should be INFINITY so that beta = 0
	*beta = *err_sq / *old_err_sq;

	#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
	int state = aggregate_skips();
	const int check_masks = MASK_GRADIENT | MASK_NORM_G | MASK_RECOVERY | (recomputed ? MASK_ITERATE : 0);
	if (state & check_masks)
	{
		if (recomputed)
			recover_rectify_g(mp.n, &mp, old_p, mp.Ap, mp.g, err_sq);
		else
			recover_rectify_x_g(mp.n, &mp, mp.x, mp.g, err_sq);

		state = aggregate_skips();
	}

	if (state & check_masks)
	{
		if (state & MASK_RECOVERY)
			fprintf(stderr, "Error discovered during recovery\n");
		else
			fprintf(stderr, "Error remaining or discovered post recovery\n");

		clear_failed(check_masks);
		log_err(SHOW_DBGINFO, "ERROR SUBSISTED PAST RECOVERY restart needed. At beta, g:%d, ||g||:%d\n", (state & MASK_GRADIENT) > 0, (state & MASK_NORM_G) > 0);
	}

	memset(mp.shared_page_reductions, 0, 2 * nb_blocks * sizeof(*mp.shared_page_reductions));
	#endif

	log_err(SHOW_TASKINFO, "Computing beta finished : err_sq = %e ; old_err_sq = %e ; beta = %e\n", *err_sq, *old_err_sq, *beta);
}

PRAGMA_TASK(inout(*normA_p_sq, *err_sq) out(*alpha, *old_err_sq, *old_err_sq2), compute_alpha, 50)
void compute_alpha(double *err_sq, double *normA_p_sq, double *old_err_sq, double *old_err_sq2, double *alpha, double *p UNUSED, double *old_p UNUSED)
{
	*alpha = *err_sq / *normA_p_sq ;
	*old_err_sq = *err_sq;
	*old_err_sq2 = *err_sq;

	#if DUE
	int state = aggregate_skips();
	#if DUE == DUE_LOSSY
	if (state)
	{
		log_err(SHOW_FAILINFO, "There was an error, restarting (eventual lossy x interpolation)");
		hard_reset(&mp);

		// refresh to check if some happened during recovery
		state = aggregate_skips() > 0 ? (aggregate_skips() | MASK_RECOVERY) : 0;
	}
	#endif
	const int check_masks = MASK_ITERATE | MASK_P | MASK_OLD_P | MASK_A_P | MASK_NORM_A_P | MASK_RECOVERY;

	#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
	if (state & check_masks)
	{
		// Errors not corrected (after async task if it exists)
		if (state &	MASK_ITERATE)
			recover_rectify_xk(mp.n, &mp, mp.x);
		if (state & (MASK_P | MASK_OLD_P | MASK_A_P | MASK_NORM_A_P))
			recover_rectify_p_Ap(mp.n, &mp, p, old_p, mp.Ap, normA_p_sq);

		// Now verify all has been corrected
		state = aggregate_skips();
	}

	memset(mp.shared_page_reductions, 0, 2 * nb_blocks * sizeof(*mp.shared_page_reductions));
	#endif

	if (state & check_masks)
	{
		if (state & MASK_RECOVERY)
			fprintf(stderr, "Error discovered during recovery\n");
		else
			fprintf(stderr, "Error remaining or discovered post recovery\n");

		clear_failed(check_masks);
		log_err(SHOW_DBGINFO, "ERROR SUBSISTED PAST RECOVERY restart needed. At alpha, x:%d, p:%d, p':%d, Ap:%d, <p,Ap>:%d\n", (state & MASK_ITERATE) > 0, (state & MASK_P) > 0, (state & MASK_OLD_P) > 0, (state & MASK_A_P) > 0, (state & MASK_NORM_A_P) > 0);
	}

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
	#if CKPT == CKPT_IN_MEMORY
	double *save_it, *save_p, save_err_sq, save_alpha;
	int do_checkpoint = 0;
	#elif CKPT == CKPT_TO_DISK
	double save_err_sq, save_alpha;
	int do_checkpoint = 0;
	#endif

	p        = (double*)aligned_calloc(failblock_size_bytes, n * sizeof(double));
	old_p    = (double*)aligned_calloc(failblock_size_bytes, n * sizeof(double));
	Ap       = (double*)aligned_calloc(failblock_size_bytes, n * sizeof(double));
	gradient = (double*)aligned_calloc(failblock_size_bytes, n * sizeof(double));
	Aiterate = (double*)aligned_calloc(failblock_size_bytes, n * sizeof(double));

	#if CKPT == CKPT_IN_MEMORY
	save_it  = (double*)aligned_calloc(failblock_size_bytes, n * sizeof(double));
	save_p   = (double*)aligned_calloc(failblock_size_bytes, n * sizeof(double));
	#endif

	// some parameters pre-computed, and show some informations
	norm_b = norm(n, b);
	thres_sq = convergence_thres * convergence_thres * norm_b;
	log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);

	mp = (magic_pointers){.A = A, .b = b, .x = iterate, .p = p, .old_p = old_p, .g = gradient, .Ap = Ap, .Ax = Aiterate, .n = n,
		.alpha = &alpha, .beta = &beta, .err_sq = &err_sq, .old_err_sq = &old_err_sq, .old_err_sq2 = &old_err_sq2, .normA_p_sq = &normA_p_sq};
	#if CKPT
	checkpoint_data ckpt_data = (checkpoint_data) {
		#if CKPT == CKPT_IN_MEMORY
		.save_x = save_it, .save_p = save_p,
		#endif
		.instructions = SAVE_CHECKPOINT, .save_err_sq = &save_err_sq, .save_alpha = &save_alpha
	};
	mp.ckpt_data = &ckpt_data;
	#endif

	#ifdef _OMPSS
	// make sure main WD (a.k.a task creation) has highest priority to start creating tasks ASAP
	nanos_set_wd_priority(nanos_current_wd(), 100);
	#endif

	setup_resilience(A, 6, &mp);
	start_measure();

	// real work starts here

	for (r = 0; r < MAXIT ; r++)
	{
		if (--do_update_gradient > 0)
		{

			update_gradient(gradient, Ap, &alpha);

			norm_task(gradient, &err_sq);

			// at this point, Ap = A * old_p
			#if DUE == DUE_ASYNC
			PRAGMA_TASK(concurrent(err_sq, ALL_BLOCKS(gradient)) inout(ALL_BLOCKS(Ap), ALL_BLOCKS(p)) in([n]iterate), recover_g, 5)
			recover_rectify_g(n, &mp, old_p, Ap, gradient, &err_sq);
			#endif

			compute_beta(&err_sq, &old_err_sq, &beta, old_p, 0);

			// in the algorithm, update_iterate is here, but we can delay it (for performance)
		}
		else
		{
			if (r > 0)
				update_iterate(iterate, old_p, &alpha);

			recompute_gradient_mvm(A, iterate, Aiterate);

			recompute_gradient_update(gradient, Aiterate, b);

			norm_task(gradient, &err_sq);

			#if CKPT
			if (r == 0)
				force_checkpoint(&ckpt_data, iterate, old_p);
			#endif

			#if DUE == DUE_ASYNC
			PRAGMA_TASK(concurrent(err_sq, ALL_BLOCKS(gradient)) inout([n]iterate) in([n]Ap), recover_xk_g, 5)
			recover_rectify_x_g(n, &mp, iterate, gradient, &err_sq);
			#endif

			compute_beta(&err_sq, &old_err_sq, &beta, old_p, 1);

			// after first beta, we are sure to have the first x, g, and checkpoint
			// so we can start injecting errors
			if (r == 0)
				#pragma omp task in(beta) label(start_injection)
				start_error_injection();
		}

		update_p(p, old_p, gradient, &beta);

		compute_Ap(A, p, Ap);

		scalar_product_task(p, Ap, &normA_p_sq);

		if (do_update_gradient > 0)
		{
			update_iterate(iterate, old_p, &alpha);
			#if DUE == DUE_ASYNC
			PRAGMA_TASK(inout(ALL_BLOCKS(iterate), alpha) in(ALL_BLOCKS(gradient)), recover_xk, 20)
			recover_rectify_xk(n, &mp, iterate);
			#endif
		}

		#if DUE == DUE_ASYNC
		PRAGMA_TASK(concurrent(normA_p_sq, ALL_BLOCKS(p), ALL_BLOCKS(Ap)) inout([n]old_p, ALL_BLOCKS(gradient)) in(ALL_BLOCKS(iterate)), recover_p_Ap, 5)
		recover_rectify_p_Ap(n, &mp, p, old_p, Ap, &normA_p_sq);
		#endif

		// when reaching this point, all tasks of loop should be created.
		// then waiting start : should be released halfway through the loop.
		// We want this to be after alpha on normal iterations, after AxIt
		// but it should not wait for recovery to finish on recovery iterations
		#pragma omp taskwait on(old_err_sq2)

		// swapping p's so we reduce pressure on the execution of the update iterate tasks
		// now output-dependencies are not conflicting with the next iteration but the one after
		{
			swap(&p, &old_p);

			failures = check_errors_signaled();

			if (r > 0)
				log_convergence(r-1, old_err_sq2, failures);

			log_err(SHOW_TASKINFO, "\n\n");

			total_failures += failures;

			if (old_err_sq <= thres_sq)
				break;

			if (do_update_gradient <= 0)
				do_update_gradient = RECOMPUTE_GRADIENT_FREQ;
			#if CKPT
			if (do_checkpoint <= 0)
				do_checkpoint = get_ckpt_freq();
			#endif
		}

		compute_alpha(&err_sq, &normA_p_sq, &old_err_sq, &old_err_sq2, &alpha, old_p, p);

		#if CKPT // NB. this implies DUE_ROLLBACK
		// should happen after p, Ap are ready and before (post-alpha) iterate and gradient updates
		// so just after alpha basically
		if (failures)
		{
			do_checkpoint = 0;
			do_update_gradient = 0;
			force_rollback(&ckpt_data, iterate, old_p);
		}
		else if (--do_checkpoint == 0)
		{
			do_update_gradient = 0;
			due_checkpoint(&ckpt_data, iterate, old_p);
		}
		#endif
	}

	#pragma omp taskwait
	// end of the math, showing infos
	stop_measure();

	failures = check_errors_signaled();
	log_convergence(r-1, old_err_sq2, failures);

	printf("CG method finished iterations:%d with error:%e (failures:%d injected:%d)\n", r, sqrt((err_sq == 0.0 ? old_err_sq : err_sq) / norm_b), total_failures, get_inject_count());

	// stop resilience stuff that's still going on
	unset_resilience(&mp);

	free(p);
	free(old_p);
	free(Ap);
	free(gradient);
	free(Aiterate);

	#if CKPT == CKPT_IN_MEMORY
	free(save_it);
	free(save_p);
	#endif
}

