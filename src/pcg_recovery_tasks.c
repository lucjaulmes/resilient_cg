
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
	*(mp->rho) = DBL_MAX;
	*(mp->old_rho) = INFINITY;
	*(mp->normA_p_sq) = INFINITY;

	recover_x_lossy(mp, mp->x);

	recompute_gradient(mp->g, mp->A, mp->x, NULL, NULL, mp->Ax, mp->b);
	reset_failed_skipped_blocks();
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
	
	if( has_skipped_blocks(MASK_ITERATE) )
		failed_recovery += abs(recover_full_xk(mp, x, REMOVE_FAULTS));
	
	if( failed_recovery )
	{
		// ouch.
		fprintf(stderr, "Impossible xk recovery, forced restart !\n");
	}

	exit_task();
}

#if DUE == DUE_IN_PATH
#pragma omp task inout(*err_sq, [n]Ap) out(*wait_for_iterate) concurrent(*rho, [n]gradient, [n]z) label(recover_g) priority(0) no_copy_deps
#else
#pragma omp task inout([n]Ap) out(*wait_for_iterate) concurrent(*err_sq, *rho, [n]gradient, [n]z) label(recover_g) priority(0) no_copy_deps
#endif
void recover_rectify_g_z(const int n UNUSED, magic_pointers *mp, const double *p, double *Ap, double *gradient, double *z, double *err_sq, double *rho, char *wait_for_iterate UNUSED)
{
	int faults = get_nb_failed_blocks();
	if( !faults )
	{
		log_err(SHOW_FAILINFO, "Skipping g_z recovery task cause nothing failed\n");
		return;
	}

	// this should happen concurrently to norm_task's, and split work onto more tasks if there is too much work ; however before update_it
	const int log2fbs = get_log2_failblock_size();
	int failed_recovery = 0, error_types;

	enter_task(RECOVERY);

	error_types = aggregate_skips();

	log_err(SHOW_FAILINFO, "Recovery task for g (faults:%d), z (faults:%d), ||g|| (faults:%d), <z,g> (faults:%d), depends on Ap (faults:%d) started\n",
			(error_types & MASK_GRADIENT) > 0, (error_types & MASK_Z) > 0, (error_types & MASK_NORM_G) > 0, (error_types & MASK_RHO) > 0, (error_types & MASK_A_P) > 0);
	
	if( error_types & MASK_A_P )
		failed_recovery += abs(recover_full_Ap(mp, Ap, p, REMOVE_FAULTS));
	
	if( error_types & MASK_GRADIENT )
	{
		failed_recovery += abs(recover_full_g_recompute(mp, gradient, KEEP_FAULTS));
		failed_recovery += abs(recover_full_g_update(mp, gradient, REMOVE_FAULTS));
	}

	if( error_types & MASK_Z )
		failed_recovery += abs(recover_full_z(mp, z, REMOVE_FAULTS));

	if( failed_recovery )
	{
		fprintf(stderr, "Impossible g recovery, forced restart !\n");
		exit_task();
		return;
	}

	// now that we did 'full' recoveries, correct error propagation
	// in here it can (luckily) only happen to some blocks of the norm
	int i, j;
	double local_err = 0.0, local_rho = 0.0, page_r;

	#if VERBOSE >= SHOW_FAILINFO
	char str[100+20*faults];
	sprintf(str, "\tAdding blocks that were skipped in reduction:");
	#endif

	for(i=0; i<get_nb_failblocks(); i++)
	{
		if( is_skipped_block(i, MASK_NORM_G) )
		{
			page_r = 0.0;

			// block skipped by reduction
			for(j=i<<log2fbs; j<(i+1)<<log2fbs && j<n; j++)
				page_r += gradient[j] * gradient[j];

			#if VERBOSE >= SHOW_FAILINFO
			sprintf(str+strlen(str), "||g||_%d ; %e", i, page_r);
			#endif

			mark_corrected(i, MASK_NORM_G);
			local_err += page_r;
		}

		if( is_skipped_block(i, MASK_RHO) )
		{
			page_r = 0.0;

			// block skipped by reduction
			for(j=i<<log2fbs; j<(i+1)<<log2fbs && j<n; j++)
				page_r += gradient[j] * z[j];

			#if VERBOSE >= SHOW_FAILINFO
			sprintf(str+strlen(str), "<g,z>_%d ; %e", i, page_r);
			#endif

			mark_corrected(i, MASK_RHO);
			local_rho += page_r;
		}
	}

	log_err(SHOW_FAILINFO, "%s; total recovery contribution are  ||g|| += %e, <g,z> += %e\n", str, local_err, local_rho);

	#pragma omp atomic
		*err_sq += local_err;

	#pragma omp atomic
		*rho += local_rho;

	// just to clear 'skipped mvm' items - we didn't do any
	clear_mvm();

	exit_task();
}

#if DUE == DUE_IN_PATH
#pragma omp task inout(*err_sq, *rho, [n]x) in(*wait_for_mvm) concurrent([n]gradient, [n]z) label(recover_xk_g) priority(0) no_copy_deps
#else
#pragma omp task inout([n]x) in(*wait_for_mvm) concurrent(*err_sq, *rho, [n]gradient, [n]z) label(recover_xk_g) priority(0) no_copy_deps
#endif
void recover_rectify_x_g_z(const int n, magic_pointers *mp, double *x, double *gradient, double *z, double *err_sq, double *rho, char *wait_for_mvm UNUSED)
{
	int faults = get_nb_failed_blocks();
	if( !faults )
	{
		log_err(SHOW_FAILINFO, "Skipping x_g_z recovery task cause nothing failed\n");
		return;
	}

	// this should happen concurrently to norm_task's, and split work onto more tasks if there is too much work
	const int log2fbs = get_log2_failblock_size();
	int failed_recovery = 0, error_types;

	enter_task(RECOVERY);

	error_types = aggregate_skips();
	log_err(SHOW_FAILINFO, "Recovery task for x (faults:%d), g (faults:%d), z (faults:%d), Ax (faults:%d), ||g|| (faults:%d), <z,g> (faults:%d) started\n",
			(error_types & MASK_ITERATE) > 0, (error_types & MASK_GRADIENT) > 0, (error_types & MASK_Z) > 0, (error_types & MASK_A_ITERATE) > 0, (error_types & MASK_NORM_G) > 0, (error_types & MASK_RHO) > 0);

	// gradient skipped everything that was contaminated.
	// however if the g marked as 'skipped' are actually failed, a recover_update does not make sense
	// and we destroyed any chance of recovering by updating x.

	if( error_types & MASK_ITERATE )
	{
		recover_mvm_skips_g(mp, gradient, REMOVE_FAULTS); // not from gradient, only from the 'skipped' items
		failed_recovery += abs(recover_full_xk(mp, x, REMOVE_FAULTS));
	}

	if( error_types & MASK_GRADIENT )
		failed_recovery += abs(recover_full_g_recompute(mp, gradient, REMOVE_FAULTS));
	
	if( error_types & MASK_Z )
		failed_recovery += abs(recover_full_z(mp, z, REMOVE_FAULTS));
	
	// just to clean the 'skipped mvm' items -- they were all corrected by re-updating p
	clear_failed(MASK_A_ITERATE);
	clear_mvm();

	if( failed_recovery )
	{
		// ouch.
		fprintf(stderr, "Impossible g & x recovery, forced restart !\n");
		exit_task();
		return;
	}

	// apply contributions to scalar product
	int i, j;
	double local_err = 0.0, local_rho = 0.0, page_r;

	#if VERBOSE >= SHOW_FAILINFO
	char str[100+20*faults];
	sprintf(str, "\tAdding blocks that were skipped in reduction:");
	#endif

	for(i=0; i<get_nb_failblocks(); i++)
	{
		if( is_skipped_block(i, MASK_NORM_G) )
		{
			page_r = 0.0;
			// block skipped by reduction
			for(j=i<<log2fbs; j<(i+1)<<log2fbs && j<n; j++)
				page_r += gradient[j] * gradient[j];

			#if VERBOSE >= SHOW_FAILINFO
			sprintf(str+strlen(str), "||g||_%d ; %e", i, page_r);
			#endif

			mark_corrected(i, MASK_NORM_G);
			local_err += page_r;
		}
		if( is_skipped_block(i, MASK_RHO) )
		{
			page_r = 0.0;
			// block skipped by reduction
			for(j=i<<log2fbs; j<(i+1)<<log2fbs && j<n; j++)
				page_r += gradient[j] * z[j];

			#if VERBOSE >= SHOW_FAILINFO
			sprintf(str+strlen(str), "<g,z>_%d ; %e", i, page_r);
			#endif

			mark_corrected(i, MASK_RHO);
			local_rho += page_r;
		}
	}

	log_err(SHOW_FAILINFO, "%s; total recovery contribution are ||g|| += %e, <g,z> += %e\n", str, local_err, local_rho);

	#pragma omp atomic
		*err_sq += local_err;

	#pragma omp atomic
		*rho += local_rho;

	exit_task();
}

#if DUE == DUE_IN_PATH
#pragma omp task in(*wait_for_mvm) inout(*normA_p_sq) concurrent([n]p, [n]Ap) label(recover_p_Ap) priority(0) no_copy_deps
#else
#pragma omp task in(*wait_for_mvm) concurrent(*normA_p_sq, [n]p, [n]Ap) label(recover_p_Ap) priority(0) no_copy_deps
#endif
void recover_rectify_p_Ap(const int n, magic_pointers *mp, double *p, double *old_p, double *Ap, double *normA_p_sq, char *wait_for_mvm UNUSED)
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

	log_err(SHOW_FAILINFO, "Recovery task for p (faults:%d), Ap (faults:%d) and <p,Ap> (faults:%d) depending on z (faults:%d) and old_p (faults:%d) started\n",
		(error_types & mask_p) > 0, (error_types & MASK_A_P) > 0, (error_types & MASK_NORM_A_P) > 0, (error_types & MASK_Z) > 0, (error_types & mask_old_p) > 0);

	if( error_types & MASK_Z )
		failed_recovery += abs(recover_full_z(mp, mp->z, REMOVE_FAULTS));

	if( error_types & mask_old_p )
		failed_recovery += abs(recover_full_old_p_invert(mp, old_p, REMOVE_FAULTS));

	if( error_types & mask_p )
		failed_recovery += abs(recover_full_p_repeat(mp, p, old_p, REMOVE_FAULTS));

	if( error_types & MASK_A_P )
		failed_recovery += abs(recover_full_Ap(mp, Ap, p, REMOVE_FAULTS));
//		// leaving the skips
//		failed_recovery += abs(recover_only_fails_Ap(mp, Ap, p, REMOVE_FAULTS));

	if( failed_recovery )
	{
		// ouch.
		fprintf(stderr, "Impossible p & Ap recovery, forced restart !\n");
		exit_task();
		return;
	}

	int i, j;
//	// now that we did 'full' recoveries, correct error propagation : so make Ap completely correct
//	int j, k, *skipped_blocks_mvm, nb_skipped_mvm, start_out, end_out, start_in, end_in;
//	nb_skipped_mvm = get_all_skipped_mvm(&skipped_blocks_mvm);
//
//	// rectify matrix_vector_multiplication
//	for(i=0; i<nb_skipped_mvm; i+=2)
//	{
//		// blocks that have been failed + fully recovered will not be marked 'skipped_not_failed' anymore
//		if( !is_skipped_not_failed_block(skipped_blocks_mvm[i+1], MASK_A_P) )
//			continue;
//
//		start_in  =  skipped_blocks_mvm[i  ]     << log2fbs;
//		end_in    = (skipped_blocks_mvm[i  ] +1) << log2fbs;
//		start_out =  skipped_blocks_mvm[i+1]    << log2fbs;
//		end_out   = (skipped_blocks_mvm[i+1]+1) << log2fbs;
//
//		if( end_in  > n ) end_in  = n;
//		if( end_out > n ) end_out = n;
//
//		log_err(SHOW_FAILINFO, "Complementing MVM with block [%d, %d]\n", skipped_blocks_mvm[i], skipped_blocks_mvm[i+1]);
//
//		for(j = start_out; j < end_out; j++)
//		{
//			for(k = mp->A->r[j]; k < mp->A->r[j+1]; k++)
//			{
//				if( mp->A->c[k] < start_in )
//					continue;
//
//				else if( mp->A->c[k] >= end_in )
//					break;
//
//				Ap[j] += mp->A->v[k] * p[ mp->A->c[k] ];
//			}
//		}
//	}
//
//	clear_mvm_skips();

	// now add contributions to scalar product.
	double local_r = 0.0, page_r;

	#if VERBOSE >= SHOW_FAILINFO
	char str[100+20*faults];
	sprintf(str, "\tAdding blocks that were skipped in reduction:");
	#endif

	for(i=0; i<get_nb_failblocks(); i++)
	{
		if( !is_skipped_block(i, MASK_NORM_A_P) )
			continue;

		#if VERBOSE >= SHOW_FAILINFO
		if( is_skipped_block(i, ~MASK_NORM_A_P) )
			log_err(SHOW_FAILINFO, "!![%d]!!", is_skipped_block(i, -1));
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

