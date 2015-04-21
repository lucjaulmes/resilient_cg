
void hard_reset(magic_pointers *mp, int *do_update_gradient, int failures, int first_n, int last_n, int *need_x, MPI_Request *need_x_req, MPI_Request *x_req)
{
	// here we are called just after alpha
	// we want have alpha = 0 to avoid any invalid updates, and err_sq = INF so that next beta = 0
	*(mp->beta) = 0.0;
	*(mp->alpha) = 0.0;
	*(mp->err_sq) = 0.0;
	*(mp->normA_p_sq) = 0.0;
	*(mp->old_err_sq) = INFINITY;

	exchange_x_for_recovery(failures & MASK_ITERATE, first_n, last_n, need_x, need_x_req, x_req);

	if( failures & MASK_ITERATE )
		recover_x_lossy(mp, mp->x);

	if( failures )
		clear_failed( ~0 );

	// if this task is not instantly followed by a recomputation, do it here manually
	if( *do_update_gradient != RECOMPUTE_GRADIENT_FREQ )
	{
		int n UNUSED = mp->A->n, count_mpix = last_n - first_n; // last+1, -self
		double *it_glob = mp->x - mpi_zonestart[mpi_rank];
		#pragma omp task inout(it_glob[0:n-1]) firstprivate(x_req, count_mpix) label(exchange_x) priority(100) no_copy_deps
		{
			enter_task(MPI_X_EXCHANGE);
			//MPI_Allgatherv(MPI_IN_PLACE, 0/*ignored*/, MPI_DOUBLE, it_glob, mpi_zonesize, mpi_zonestart, MPI_DOUBLE, MPI_COMM_WORLD);

			MPI_Startall(2*count_mpix, x_req);
			MPI_Waitall(2*count_mpix, x_req, MPI_STATUSES_IGNORE);

			exit_task();
		}

		recompute_gradient_mvm(mp->A, it_glob, NULL, NULL, mp->Ax);
		recompute_gradient_update(mp->g, NULL, mp->Ax, mp->b);
		#pragma omp taskwait

		*do_update_gradient = RECOMPUTE_GRADIENT_FREQ;
	}
}

void exchange_x_for_recovery(int check_need, int first_n, int last_n, int *need_x, MPI_Request *need_x_req, MPI_Request *x_req)
{
	int nn = last_n-first_n, nx = 0;
	MPI_Request exchanges_needed[2*nn];

	MPI_Request *x_recv = x_req, *x_send = x_req+nn;

	// NB ; the following seems reverse because of double communication pattern :
	// we first send whether we need to recieve, and recieve whether we need to send
	int *need_x_send = need_x, *need_x_recv = need_x+nn;
	
	// if we have failures, set 1 in every mpi_neighbour of a failed page, otherwise 0
	if(check_need)
	{
		int i, first_n_page, last_n_page, first_n_needed = mpi_size, last_n_needed = 0;
		const int log2fbs = get_log2_failblock_size();

		for(i=0; i < get_nb_failblocks(); i++)
		{
			if( !is_skipped_block(i, MASK_GRADIENT|MASK_ITERATE) )
				continue;

			determine_mpi_neighbours(mp.A, i << log2fbs, (i+1) << log2fbs, mpi_rank, mpi_size, &first_n_page, &last_n_page);

			if( first_n_needed > first_n_page )
				first_n_needed = first_n_page;
			if(  last_n_needed <  last_n_page )
				 last_n_needed =  last_n_page;
		}

		int j=0; // j = i - first_n - (i > mpi_rank ? 1 : 0)
		for(i=first_n; i<=last_n; i++)
			if( i != mpi_rank )
			{
				if(first_n_needed <= i && i <= last_n_needed)
				{
					need_x_recv[j] = 1;
					exchanges_needed[nx++] = x_recv[j];
				}
				else
					need_x_recv[j] = 0;

				j++;
			}
	}
	else
		memset(need_x_recv, 0, nn*sizeof(int));

	// exchange flags to know which real MPI exchanges will be needed
	MPI_Startall(2*nn, need_x_req);
	#if VERBOSE < SHOW_FAILINFO
	MPI_Waitall(2*nn, need_x_req, MPI_STATUSES_IGNORE);
	#else
	MPI_Status mpi_status[2*nn];
	MPI_Waitall(2*nn, need_x_req, mpi_status);
	char print_status[25+75*nn];
	sprintf(print_status, "MPI_Status for need_x_req :");
	for(int z=0; z<2*nn; z++)
		sprintf(print_status+strlen(print_status), " %d:{SOURCE=%d, TAG=%d, ERROR=%d}", z, mpi_status[z].MPI_SOURCE, mpi_status[z].MPI_TAG, mpi_status[z].MPI_ERROR);
	log_err(SHOW_FAILINFO, "%s\n", print_status);
	#endif

	#if VERBOSE >= SHOW_FAILINFO
	int n_recv = nx;
	#endif

	// add needed x requests and start/wait on them
	int r;
	for(r=0; r<nn; r++)
		if(need_x_send[r])
			exchanges_needed[nx++] = x_send[r];

	if( nx )
	{
		log_err(SHOW_FAILINFO, "\tNeed for x exchange for recovery : %d/%d receives, %d/%d sends\n", n_recv, nn, nx-n_recv, nn);
		MPI_Startall(nx, exchanges_needed);
		#if VERBOSE < SHOW_FAILINFO
		MPI_Waitall(nx, exchanges_needed, MPI_STATUSES_IGNORE);
		#else
		MPI_Waitall(nx, exchanges_needed, mpi_status);
		sprintf(print_status, "MPI_Status for need_x_req :");
		for(int z=0; z<nx; z++)
			sprintf(print_status+strlen(print_status), " %d:{SOURCE=%d, TAG=%d, ERROR=%d}", z, mpi_status[z].MPI_SOURCE, mpi_status[z].MPI_TAG, mpi_status[z].MPI_ERROR);
		log_err(SHOW_FAILINFO, "%s\n", print_status);
		#endif
	}
	else
		log_err(SHOW_FAILINFO, "\tNo need for x exchange for recovery\n");
}

#if DUE == DUE_IN_PATH
#pragma omp task inout([n]x, *wait_for_iterate, *wait_for_prev) label(recover_xk) priority(0) no_copy_deps
#else
#pragma omp task inout([n]x, *wait_for_iterate) concurrent(*wait_for_prev) label(recover_xk) priority(5) no_copy_deps
#endif
void recover_rectify_xk(const int n UNUSED, magic_pointers *mp, double *x, char *wait_for_prev UNUSED, char *wait_for_iterate UNUSED, int first_n, int last_n, int *need_x, MPI_Request *need_x_req, MPI_Request *x_req)
{
	int failed_recovery = 0;

	enter_task(RECOVERY);
	
	int x_failed = has_skipped_blocks(MASK_ITERATE);

	log_err(SHOW_FAILINFO, "Recovery task x for x (faults:%d) started\n", x_failed);

	exchange_x_for_recovery(x_failed, first_n, last_n, need_x, need_x_req, x_req);

	if( x_failed )
		failed_recovery += abs(recover_full_xk(mp, x, REMOVE_FAULTS));
	
	if( failed_recovery )
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

	log_err(SHOW_FAILINFO, "Recovery task g for g (faults:%d), ||g|| (faults:%d) depends on Ap (faults:%d) started\n",
			(error_types & MASK_GRADIENT) > 0, (error_types & MASK_GRADIENT) > 0, (error_types & MASK_A_P) > 0);
	
	if( error_types & MASK_A_P )
		failed_recovery += abs(recover_full_Ap(mp, Ap, p, REMOVE_FAULTS));
	
	if( error_types & MASK_GRADIENT )
	{
		failed_recovery += abs(recover_full_g_from_p_diff(mp, mp->g, *(mp->beta), p, p==mp->p?mp->old_p:mp->p, KEEP_FAULTS));
		failed_recovery += abs(recover_full_g_update(mp, gradient, REMOVE_FAULTS));
	}

	if( failed_recovery )
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
		if( !is_skipped_block(i, MASK_NORM_G) )
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
void recover_rectify_x_g(const int n, magic_pointers *mp, double *x, double *gradient, double *err_sq, char *wait_for_mvm UNUSED)
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
	log_err(SHOW_FAILINFO, "Recovery task x_g for x (faults:%d), g (faults:%d), Ax (faults:%d) and ||g|| (faults:%d) started\n", (error_types & MASK_ITERATE) > 0, (error_types & MASK_GRADIENT) > 0, (error_types & MASK_A_ITERATE) > 0, (error_types & NORM_GRADIENT) > 0);

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
	double local_r = 0.0, page_r;

	#if VERBOSE >= SHOW_FAILINFO
	char str[100+20*faults];
	sprintf(str, "\tAdding blocks that were skipped in reduction:");
	#endif

	for(i=0; i<get_nb_failblocks(); i++)
	{
		if( !is_skipped_block(i, MASK_NORM_G) )
			continue;

		page_r = 0.0;
		// block skipped by reduction
		for(j=i<<log2fbs; j<(i+1)<<log2fbs && j<n; j++)
			page_r += gradient[j] * gradient[j];

		#if VERBOSE >= SHOW_FAILINFO
		sprintf(str+strlen(str), ", %d: %e", i, page_r);
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
void recover_rectify_p_Ap(const int n, magic_pointers *mp, double *p, double *old_p, double *Ap, double *normA_p_sq, char *wait_for_mvm UNUSED, char *wait_for_iterate UNUSED)
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

	log_err(SHOW_FAILINFO, "Recovery task p_Ap for p (faults:%d), Ap (faults:%d) and <p,Ap> (faults:%d) depending on old_p (faults:%d) started\n", (error_types & mask_p) > 0, (error_types & MASK_A_P) > 0, (error_types & MASK_NORM_A_P) > 0, (error_types & mask_old_p) > 0);

	if( error_types & mask_old_p )
		failed_recovery += abs(recover_full_old_p_invert(mp, old_p, REMOVE_FAULTS));

	if( error_types & mask_p )
		failed_recovery += abs(recover_full_p_repeat(mp, p, old_p, REMOVE_FAULTS));

	if( error_types & MASK_A_P )
		failed_recovery += abs(recover_full_Ap(mp, Ap, p, REMOVE_FAULTS));

	if( failed_recovery )
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
		if( !is_skipped_block(i, MASK_NORM_A_P) )
			continue;

		#if VERBOSE >= SHOW_FAILINFO
		if( is_skipped_block(i, ~MASK_NORM_A_P) )
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

#if DUE == DUE_IN_PATH
#pragma omp task inout(*wait_for_p, *wait_for_p2) concurrent([n]p) label(recover_p_early) priority(0) no_copy_deps
#else
#pragma omp task concurrent(*wait_for_p, *wait_for_p2, [n]p) label(recover_p_early) priority(5) no_copy_deps
#endif
void recover_rectify_p_early(const int n UNUSED, magic_pointers *mp, double *p, double *old_p, char *wait_for_p UNUSED, char *wait_for_p2 UNUSED, int first_n, int last_n, int *need_x, MPI_Request *need_x_req, MPI_Request *x_req)
{
	// this should happen concurrently to norm_task's, and split work onto more tasks if there is too much work
	const int mask_p = 1 << get_data_vectptr(p), mask_old_p = 1 << get_data_vectptr(old_p);
	int failed_recovery = 0, error_types;
	// are we sure g is already recomputed ?

	enter_task(RECOVERY);

	error_types = aggregate_skips();

	log_err(SHOW_FAILINFO, "Recovery task p_early for p (faults:%d) before exchange, depending on g (faults:%d) and old_p (faults:%d) started\n", (error_types & mask_p) > 0, (error_types & MASK_GRADIENT) > 0, (error_types & mask_old_p) > 0);

	exchange_x_for_recovery(error_types & MASK_GRADIENT, first_n, last_n, need_x, need_x_req, x_req);

	// NB force to execute this task before update_it so that iterate is at a coherent state if needed for g recovery
	if( error_types & MASK_GRADIENT )
	{
		// for each page of g failed, if corresponding pages of p are skipped, 
		// we can get the old g from old p and old_old_p (at iteration -2, not overwritten)
		failed_recovery += abs(recover_full_g_recompute(mp, mp->g, KEEP_FAULTS));
		
		failed_recovery += abs(recover_full_g_update(mp, mp->g, REMOVE_FAULTS));
	}

	if( error_types & mask_old_p )
		failed_recovery += abs(recover_early_full_old_p_invert(mp, old_p, REMOVE_FAULTS));

	if( error_types & mask_p )
		failed_recovery += abs(recover_full_p_repeat(mp, p, old_p, REMOVE_FAULTS));

	if( failed_recovery )
	{
		// ouch.
		fprintf(stderr, "Impossible p & Ap recovery, forced restart !\n");
		exit_task();
		return;
	}

	exit_task();
}
