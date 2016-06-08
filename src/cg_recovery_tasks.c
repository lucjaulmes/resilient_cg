
void hard_reset(magic_pointers *mp)
{
	// This method can be used as a fallback when DUE techniques don't work.
	// Primary use is to implement other techniques (against which to compare)
	#if CKPT
	force_rollback(mp->A->n, mp->ckpt_data, mp->x, mp->g, mp->old_p, mp->Ap);
	#else
	// here we are called at alpha (the function will finish executing normally)
	// we want ||p||_A = INF to have alpha = 0, err_sq = INF so that next beta = 0
	*(mp->beta) = 0.0;
	*(mp->alpha) = 0.0;
	*(mp->err_sq) = DBL_MAX;
	*(mp->old_err_sq) = INFINITY;
	*(mp->normA_p_sq) = INFINITY;

	recover_x_lossy(mp, mp->x);

	recompute_gradient_mvm(mp->A, mp->x, NULL, NULL, mp->Ax);
	recompute_gradient_update(mp->g, NULL, mp->Ax, mp->b);
	clear_failed(~0);
	#endif
}

#pragma omp task inout([n]x, *wait_for_iterate) label(recover_xk) priority(20) no_copy_deps
void recover_rectify_xk(const int n UNUSED, magic_pointers *mp, double *x, char *wait_for_iterate UNUSED)
{
	int faults = get_nb_failed_blocks();
	if( !faults )
	{
		log_err(SHOW_FAILINFO, "Skipping x recovery task cause nothing failed\n");
		return;
	}

	int failed_recovery = 0;

	enter_task(RECOVERY);

	log_err(SHOW_FAILINFO, "Recovery task for x (faults:%d) started\n", has_skipped_blocks(MASK_ITERATE));

	if(has_skipped_blocks(MASK_ITERATE))
		failed_recovery += abs(recover_full_xk(mp, x, REMOVE_FAULTS));

	if(failed_recovery)
	{
		// ouch.
		fprintf(stderr, "Impossible xk recovery, forced restart !\n");
	}

	exit_task();
}

#if DUE == DUE_IN_PATH
#pragma omp task inout(*err_sq, [n]Ap) out(*wait_for_iterate) concurrent([n]gradient) label(recover_g) priority(0) no_copy_deps
#else
#pragma omp task inout([n]Ap) out(*wait_for_iterate) concurrent(*err_sq, [n]gradient) label(recover_g) priority(5) no_copy_deps
#endif
void recover_rectify_g(const int n UNUSED, magic_pointers *mp, const double *p, double *Ap, double *gradient, double *err_sq, char *wait_for_iterate UNUSED)
{
	int faults = get_nb_failed_blocks();
	if( !faults )
	{
		log_err(SHOW_FAILINFO, "Skipping g recovery task cause nothing failed\n");
		return;
	}

	// this should happen concurrently to norm_task's, and split work onto more tasks if there is too much work ; however before update_it
	const int log2fbs = get_log2_failblock_size();
	int failed_recovery = 0, error_types;

	enter_task(RECOVERY);

	error_types = aggregate_skips();

	log_err(SHOW_FAILINFO, "Recovery task for g (faults:%d), ||g|| (faults:%d) depends on Ap (faults:%d) started\n",
			(error_types & MASK_GRADIENT) > 0, (error_types & NORM_GRADIENT) > 0, (error_types & MASK_A_P) > 0);

	if(error_types & MASK_A_P)
		failed_recovery += abs(recover_full_Ap(mp, Ap, p, REMOVE_FAULTS));

	if(error_types & MASK_GRADIENT)
	{
		failed_recovery += abs(recover_full_g_recompute(mp, gradient, KEEP_FAULTS));
		failed_recovery += abs(recover_full_g_update(mp, gradient, REMOVE_FAULTS));
	}

	if(failed_recovery)
	{
		fprintf(stderr, "Impossible g recovery, forced restart !\n");
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

	for(i=0; i<get_nb_failblocks(); i++)
	{
		if(!is_skipped_block(i, MASK_NORM_G))
			continue;

		page_r = 0.0;

		// block skipped by reduction
		for(j=i<<log2fbs; j<(i+1)<<log2fbs && j<n; j++)
			page_r += gradient[j] * gradient[j];

		#if VERBOSE >= SHOW_FAILINFO
		sprintf(str+strlen(str), " %d ; %e", i, page_r);
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

#if DUE == DUE_IN_PATH
#pragma omp task inout(*err_sq, [n]x) in(*wait_for_mvm) concurrent([n]gradient) label(recover_xk_g) priority(0) no_copy_deps
#else
#pragma omp task inout([n]x) in(*wait_for_mvm) concurrent(*err_sq, [n]gradient) label(recover_xk_g) priority(5) no_copy_deps
#endif
void recover_rectify_x_g(const int n UNUSED, magic_pointers *mp, double *x, double *gradient, double *err_sq, char *wait_for_mvm UNUSED)
{
	int faults = get_nb_failed_blocks();
	if( !faults )
	{
		log_err(SHOW_FAILINFO, "Skipping x_g recovery task cause nothing failed\n");
		return;
	}

	// this should happen concurrently to norm_task's, and split work onto more tasks if there is too much work
	const int log2fbs = get_log2_failblock_size();
	int failed_recovery = 0, error_types;

	enter_task(RECOVERY);

	error_types = aggregate_skips();
	log_err(SHOW_FAILINFO, "Recovery task for x (faults:%d), g (faults:%d), Ax (faults:%d) and ||g|| (faults:%d) started\n", (error_types & MASK_ITERATE) > 0, (error_types & MASK_GRADIENT) > 0, (error_types & MASK_A_ITERATE) > 0, (error_types & NORM_GRADIENT) > 0);

	// gradient skipped everything that was contaminated.
	// however if the g marked as 'skipped' are actually failed, a recover_update does not make sense
	// and we destroyed any chance of recovering by updating x.

	if(error_types & MASK_ITERATE)
	{
		recover_mvm_skips_g(mp, gradient, REMOVE_FAULTS); // not from gradient, only from the 'skipped' items
		failed_recovery += abs(recover_full_xk(mp, x, REMOVE_FAULTS));
	}

	if(error_types & MASK_GRADIENT)
		failed_recovery += abs(recover_full_g_recompute(mp, gradient, REMOVE_FAULTS));

	// just to clean the 'skipped mvm' items -- they were all corrected by re-updating p
	clear_failed(MASK_A_ITERATE);
	clear_mvm();

	if(failed_recovery)
	{
		// ouch.
		fprintf(stderr, "Impossible g & x recovery, forced restart !\n");
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

	for(i=0; i<get_nb_failblocks(); i++)
	{
		if(!is_skipped_block(i, MASK_NORM_G))
			continue;

		page_r = 0.0;
		// block skipped by reduction
		for(j=i<<log2fbs; j<(i+1)<<log2fbs && j<n; j++)
			page_r += gradient[j] * gradient[j];

		#if VERBOSE >= SHOW_FAILINFO
		sprintf(str+strlen(str), " %d ; %e", i, page_r);
		#endif

		mark_corrected(i, MASK_NORM_G);
		local_r += page_r;
	}

	log_err(SHOW_FAILINFO, "%s; total recovery contribution is %e\n", str, local_r);

	#pragma omp atomic
		*err_sq += local_r;

	exit_task();
}

#if DUE == DUE_IN_PATH
#pragma omp task in(*wait_for_mvm, *wait_for_iterate) inout(*normA_p_sq) concurrent([n]p, [n]Ap) label(recover_p_Ap) priority(0) no_copy_deps
#else
#pragma omp task in(*wait_for_mvm, *wait_for_iterate) concurrent(*normA_p_sq, [n]p, [n]Ap) label(recover_p_Ap) priority(5) no_copy_deps
#endif
void recover_rectify_p_Ap(const int n UNUSED, magic_pointers *mp, double *p, double *old_p, double *Ap, double *normA_p_sq, char *wait_for_mvm UNUSED, char *wait_for_iterate UNUSED)
{
	int faults = get_nb_failed_blocks();
	if( !faults )
	{
		log_err(SHOW_FAILINFO, "Skipping p_Ap recovery task cause nothing failed\n");
		return;
	}

	// this should happen concurrently to norm_task's, and split work onto more tasks if there is too much work
	const int log2fbs = get_log2_failblock_size(), mask_p = 1 << get_data_vectptr(p), mask_old_p = 1 << get_data_vectptr(old_p);
	int failed_recovery = 0, error_types;
	// are we sure g is already recomputed ?

	enter_task(RECOVERY);

	error_types = aggregate_skips();

	log_err(SHOW_FAILINFO, "Recovery task for p (faults:%d), Ap (faults:%d) and <p,Ap> (faults:%d) depending on g (faults:%d) and old_p (faults:%d) started\n", (error_types & mask_p) > 0, (error_types & MASK_A_P) > 0, (error_types & MASK_NORM_A_P) > 0, (error_types & MASK_GRADIENT) > 0, (error_types & mask_old_p) > 0);

	if(error_types & MASK_GRADIENT)
		failed_recovery += abs(recover_full_g_recompute(mp, mp->g, REMOVE_FAULTS));

	if(error_types & mask_old_p)
		failed_recovery += abs(recover_full_old_p_invert(mp, old_p, REMOVE_FAULTS));

	if(error_types & mask_p)
		failed_recovery += abs(recover_full_p_repeat(mp, p, old_p, REMOVE_FAULTS));

	if(error_types & MASK_A_P)
		failed_recovery += abs(recover_full_Ap(mp, Ap, p, REMOVE_FAULTS));

	if(failed_recovery)
	{
		// ouch.
		fprintf(stderr, "Impossible p & Ap recovery, forced restart !\n");
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

	for(i=0; i<get_nb_failblocks(); i++)
	{
		if(!is_skipped_block(i, MASK_NORM_A_P))
			continue;

		#if VERBOSE >= SHOW_FAILINFO
		if(is_skipped_block(i, ~MASK_NORM_A_P))
			fprintf(stderr, "!![%d]!!", is_skipped_block(i, -1));
		#endif

		page_r = 0.0;

		// block skipped by reduction
		for(j=i<<log2fbs; j<(i+1)<<log2fbs && j<n; j++)
			page_r += p[j] * Ap[j];

		local_r += page_r;

		#if VERBOSE >= SHOW_FAILINFO
		sprintf(str+strlen(str), " %d ; %e", i, page_r);
		#endif

		mark_corrected(i, MASK_NORM_A_P);
	}

	log_err(SHOW_FAILINFO, "%s; total recovery contribution is %e\n", str, local_r);

	#pragma omp atomic
		*normA_p_sq += local_r;

	exit_task();
}
