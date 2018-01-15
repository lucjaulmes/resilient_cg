
void hard_reset(magic_pointers *mp)
{
	// This method can be used as a fallback when DUE techniques don't work.
	// Primary use is to implement other techniques (against which to compare)
	#if CKPT

	force_rollback(mp->A->n, mp->ckpt_data, mp->x, mp->g, mp->old_p, mp->Ap);
	#pragma omp taskwait

	#else
	// here we are called at alpha (the function will finish executing normally)
	// we want ||p||_A = INF to have alpha = 0, err_sq = INF so that next beta = 0
	*(mp->beta) = 0.0;
	*(mp->alpha) = 0.0;
	*(mp->err_sq) = DBL_MAX;
	*(mp->old_err_sq) = INFINITY;
	*(mp->normA_p_sq) = INFINITY;

	recover_x_lossy(mp, mp->x);

	clear_failed(~0);

	recompute_gradient_mvm(mp->A, mp->x, mp->Ax);
	recompute_gradient_update(mp->g, mp->Ax, mp->b);
	#pragma omp taskwait

	#endif
}

void recover_rectify_xk(const int n UNUSED, magic_pointers *mp, double *x UNUSED)
{
	// g used for recovery, also waiting for all its blocks makes sure it's not updating
	// normA_p_sq is used artificially to force recovery in the critical path
	// alpha is used to force recovery before next compute_alpha (as well as after g and x)
#if DUE == DUE_IN_PATH
	PRAGMA_TASK(inout(ALL_BLOCKS(x), mp->normA_p_sq) in(ALL_BLOCKS(mp->g)), recover_xk, 20)
#else
	PRAGMA_TASK(inout(ALL_BLOCKS(x), mp->alpha) in(ALL_BLOCKS(mp->g)), recover_xk, 20)
#endif
	{
		int faults = get_nb_failed_blocks();
		if (!faults)
		{
			log_err(SHOW_FAILINFO, "Skipping x recovery task cause nothing failed\n");
			return;
		}

		int failed_recovery = 0;

		enter_task(RECOVERY);

		log_err(SHOW_FAILINFO, "Recovery task for x (faults:%d) started\n", has_skipped_blocks(MASK_ITERATE));

		if (has_skipped_blocks(MASK_ITERATE))
			failed_recovery += abs(recover_full_xk(mp, x, REMOVE_FAULTS));

		if (failed_recovery)
		{
			// ouch.
			fprintf(stderr, "Impossible xk recovery\n");
		}

		exit_task();
	}
}

void recover_rectify_g(const int n UNUSED, magic_pointers *mp, const double *p, double *Ap, double *gradient, double *err_sq UNUSED)
{
	// p and x may be used for recovering
	// Also, using inout(p) forces update_it(in:p inout:x) to wait, so we get consistent recoveries
#if DUE == DUE_IN_PATH
	PRAGMA_TASK(inout(*err_sq, ALL_BLOCKS(gradient), ALL_BLOCKS(Ap), ALL_BLOCKS(p)) in([n]mp->x), recover_g, 0)
#else
	PRAGMA_TASK(concurrent(*err_sq, ALL_BLOCKS(gradient)) inout(ALL_BLOCKS(Ap), ALL_BLOCKS(p)) in([n]mp->x), recover_g, 5)
#endif
	{
		int faults = get_nb_failed_blocks();
		if (!faults)
		{
			log_err(SHOW_FAILINFO, "Skipping g recovery task cause nothing failed\n");
			return;
		}

		// this should happen concurrently to norm_task's, and split work onto more tasks if there is too much work ; however before update_it
		int failed_recovery = 0, error_types;

		enter_task(RECOVERY);

		error_types = aggregate_skips();

		log_err(SHOW_FAILINFO, "Recovery task for g (faults:%d), ||g|| (faults:%d) depends on Ap (faults:%d) started\n",
				(error_types & MASK_GRADIENT) > 0, (error_types & MASK_NORM_G) > 0, (error_types & MASK_A_P) > 0);

		if (error_types & MASK_A_P)
			failed_recovery += abs(recover_full_Ap(mp, Ap, p, REMOVE_FAULTS));

		if (error_types & MASK_GRADIENT)
		{
			failed_recovery += abs(recover_full_g_recompute(mp, gradient, KEEP_FAULTS));
			failed_recovery += abs(recover_full_g_update(mp, gradient, REMOVE_FAULTS));
		}

		if (failed_recovery)
		{
			fprintf(stderr, "Impossible g recovery\n");
			exit_task();
			return;
		}

		// now that we did 'full' recoveries, correct error propagation
		// in here it can (luckily) only happen to some blocks of the norm
		int i, j;
		double local_r = 0.0, page_r;

		#if VERBOSE >= SHOW_FAILINFO
		char str[100+20*faults];
		sprintf(str, "\tAdding blocks that were skipped in reduction:");
		#endif

		for (i = 0; i < nb_failblocks; i++)
		{
			if (!is_skipped_block(i, MASK_NORM_G))
				continue;

			page_r = 0.0;

			// block skipped by reduction
			for (j = i * failblock_size_dbl; j < (i+1) * failblock_size_dbl && j < n; j++)
				page_r += gradient[j] * gradient[j];

			#if VERBOSE >= SHOW_FAILINFO
			sprintf(str+strlen(str), " %d ; %e", i, page_r);
			#endif

			// if it was a shared block, remove all (potential) partial contributions
			#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
			if (is_shared_block(i))
			{
				for (j = 0; j < nb_blocks - 1; j++)
				{
					if ((get_block_end(j) / failblock_size_dbl) == i)
					{
						// end of block and start of next
						page_r -= mp->shared_page_reductions[2 * j + 1];
						page_r -= mp->shared_page_reductions[2 * j + 2];
					}
					else if ((get_block_end(j) / failblock_size_dbl) > i)
						break;
				}
			}
			#endif

			mark_corrected(i, MASK_NORM_G);
			local_r += page_r;
		}

		log_err(SHOW_FAILINFO, "%s; total recovery contribution is %e\n", str, local_r);

		#pragma omp atomic
			*err_sq += local_r;

		// just to clear 'skipped mvm' items - we didn't do any
		clear_mvm();

		exit_task();
	}
}

void recover_rectify_x_g(const int n UNUSED, magic_pointers *mp, double *x, double *gradient, double *err_sq UNUSED)
{
#if DUE == DUE_IN_PATH
	PRAGMA_TASK(inout(*err_sq, ALL_BLOCKS(gradient), [n]x) in([n]mp->Ap), recover_xk_g, 0)
#else
	PRAGMA_TASK(concurrent(*err_sq, ALL_BLOCKS(gradient)) inout([n]x) in([n]mp->Ap), recover_xk_g, 5)
#endif
	{
		int faults = get_nb_failed_blocks();
		if (!faults)
		{
			log_err(SHOW_FAILINFO, "Skipping x_g recovery task cause nothing failed\n");
			return;
		}

		// this should happen concurrently to norm_task's, and split work onto more tasks if there is too much work
		int failed_recovery = 0, error_types;

		enter_task(RECOVERY);

		error_types = aggregate_skips();
		log_err(SHOW_FAILINFO, "Recovery task for x (faults:%d), g (faults:%d), Ax (faults:%d) and ||g|| (faults:%d) started\n",
			(error_types & MASK_ITERATE) > 0, (error_types & MASK_GRADIENT) > 0, (error_types & MASK_A_ITERATE) > 0, (error_types & NORM_GRADIENT) > 0);

		// gradient skipped everything that was contaminated.
		// however if the g marked as 'skipped' are actually failed, a recover_update does not make sense
		// and we destroyed any chance of recovering by updating x.

		if (error_types & MASK_ITERATE)
		{
			recover_mvm_skips_g(mp, gradient, REMOVE_FAULTS); // not from gradient, only from the 'skipped' items
			failed_recovery += abs(recover_full_xk(mp, x, REMOVE_FAULTS));
		}

		if (error_types & MASK_GRADIENT)
			failed_recovery += abs(recover_full_g_recompute(mp, gradient, REMOVE_FAULTS));

		// just to clean the 'skipped mvm' items -- they were all corrected by re-updating g
		clear_failed(MASK_A_ITERATE);
		clear_mvm();

		if (failed_recovery)
		{
			// ouch.
			fprintf(stderr, "Impossible g & x recovery\n");
			exit_task();
			return;
		}

		// apply contributions to scalar product
		int i, j;
		double local_r = 0.0, page_r;

		#if VERBOSE >= SHOW_FAILINFO
		char str[100+20*faults];
		sprintf(str, "\tAdding blocks that were skipped in reduction:");
		#endif

		for (i = 0; i < nb_failblocks; i++)
		{
			if (!is_skipped_block(i, MASK_NORM_G))
				continue;

			page_r = 0.0;
			// block skipped by reduction
			for (j = i * failblock_size_dbl; j < (i+1) * failblock_size_dbl && j < n; j++)
				page_r += gradient[j] * gradient[j];

			#if VERBOSE >= SHOW_FAILINFO
			sprintf(str+strlen(str), " %d ; %e", i, page_r);
			#endif

			#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
			// if it was a shared block, remove all (potential) partial contributions
			if (is_shared_block(i))
			{
				for (j = 0; j < nb_blocks - 1; j++)
				{
					if ((get_block_end(j) / failblock_size_dbl) == i)
					{
						// end of block and start of next
						page_r -= mp->shared_page_reductions[2 * j + 1];
						page_r -= mp->shared_page_reductions[2 * j + 2];
					}
					else if ((get_block_end(j) / failblock_size_dbl) > i)
						break;
				}
			}
			#endif

			mark_corrected(i, MASK_NORM_G);
			local_r += page_r;
		}

		log_err(SHOW_FAILINFO, "%s; total recovery contribution is %e\n", str, local_r);

		#pragma omp atomic
			*err_sq += local_r;

		exit_task();
	}
}

void recover_rectify_p_Ap(const int n UNUSED, magic_pointers *mp, double *p, double *old_p, double *Ap, double *normA_p_sq UNUSED)
{
	// both old_p and g used for recoveries (and maybe themselves repaired) -- thus requiring x
#if DUE == DUE_IN_PATH
	PRAGMA_TASK(inout(*normA_p_sq, ALL_BLOCKS(p), ALL_BLOCKS(Ap), [n]old_p, ALL_BLOCKS(mp->g)) in(ALL_BLOCKS(mp->x)), recover_p_Ap, 0)
#else
	PRAGMA_TASK(concurrent(*normA_p_sq, ALL_BLOCKS(p), ALL_BLOCKS(Ap)) inout([n]old_p, ALL_BLOCKS(mp->g)) in(ALL_BLOCKS(mp->x)), recover_p_Ap, 5)
#endif
	{
		int faults = get_nb_failed_blocks();
		if (!faults)
		{
			log_err(SHOW_FAILINFO, "Skipping p_Ap recovery task cause nothing failed\n");
			return;
		}

		// this should happen concurrently to norm_task's, and split work onto more tasks if there is too much work
		const int mask_p = 1 << get_data_vectptr(p), mask_old_p = 1 << get_data_vectptr(old_p);
		int failed_recovery = 0, error_types;
		// are we sure g is already recomputed ?

		enter_task(RECOVERY);

		error_types = aggregate_skips();

		log_err(SHOW_FAILINFO, "Recovery task for p (faults:%d), Ap (faults:%d) and <p,Ap> (faults:%d) depending on g (faults:%d) and old_p (faults:%d) started\n",
			(error_types & mask_p) > 0, (error_types & MASK_A_P) > 0, (error_types & MASK_NORM_A_P) > 0, (error_types & MASK_GRADIENT) > 0, (error_types & mask_old_p) > 0);

		if (error_types & MASK_GRADIENT)
			failed_recovery += abs(recover_full_g_recompute(mp, mp->g, REMOVE_FAULTS));

		if (error_types & mask_old_p)
			failed_recovery += abs(recover_full_old_p_invert(mp, old_p, REMOVE_FAULTS));

		if (error_types & mask_p)
			failed_recovery += abs(recover_full_p_repeat(mp, p, old_p, REMOVE_FAULTS));

		if (error_types & MASK_A_P)
			failed_recovery += abs(recover_full_Ap(mp, Ap, p, REMOVE_FAULTS));

		if (failed_recovery)
		{
			// ouch.
			fprintf(stderr, "Impossible p & Ap recovery\n");
			exit_task();
			return;
		}

		// now add contributions to scalar product.
		int i, j;
		double local_r = 0.0, page_r;

		#if VERBOSE >= SHOW_FAILINFO
		char str[100+20*faults];
		sprintf(str, "\tAdding blocks that were skipped in reduction:");
		#endif

		for (i = 0; i < nb_failblocks; i++)
		{
			if (!is_skipped_block(i, MASK_NORM_A_P))
				continue;

			#if VERBOSE >= SHOW_FAILINFO
			if (is_skipped_block(i, ~MASK_NORM_A_P))
				fprintf(stderr, "!![%d]!!", is_skipped_block(i, -1));
			#endif

			page_r = 0.0;

			// block skipped by reduction, recompute
			for (j = i * failblock_size_dbl; j < (i+1) * failblock_size_dbl && j < n; j++)
				page_r += p[j] * Ap[j];


			#if VERBOSE >= SHOW_FAILINFO
			sprintf(str+strlen(str), " %d ; %e", i, page_r);
			#endif

			#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
			// if it was a shared block, remove all (potential) partial contributions
			if (is_shared_block(i))
			{
				for (j = 0; j < nb_blocks - 1; j++)
				{
					if ((get_block_end(j) / failblock_size_dbl) == i)
					{
						// end of block and start of next
						page_r -= mp->shared_page_reductions[2 * j + 1];
						page_r -= mp->shared_page_reductions[2 * j + 2];
					}
					else if ((get_block_end(j) / failblock_size_dbl) > i)
						break;
				}
			}
			#endif

			local_r += page_r;
			mark_corrected(i, MASK_NORM_A_P);
		}

		log_err(SHOW_FAILINFO, "%s; total recovery contribution is %e\n", str, local_r);

		#pragma omp atomic
			*normA_p_sq += local_r;

		exit_task();
	}
}

