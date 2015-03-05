void check_sdc_recompute_grad(const int save, detect_error_data *err_data, const double *b, double *iterate, double *gradient, double *p, double *Ap, char *wait_for_mvm UNUSED, double *Aiterate, double *err_sq, const double threshold)
{
	double *zero = &(err_data->helper_1), normAb = err_data->helper_4;
	int *behaviour = &(err_data->error_detected), *prev_error = &(err_data->prev_error);
	int i;
	for(i=0; i < nb_blocks; i ++)
	{
		int s = get_block_start(i), e = get_block_end(i);
		if(e > n)
			e = n;

		// gradient <- b - Aiterate
		#pragma omp task in(b[s:e-1], Aiterate[s:e-1]) inout(gradient[s:e-1]) concurrent(*wait_for_mvm, *zero) firstprivate(s, e) label(b-AxIt_check_sdc) priority(10) no_copy_deps
		{
			// Can't afford to propagate errors from Ax to g ; if they come from a block i of x
			// then it will be impossible for x_i to be recovered from g_i, x_j (j!=i)

			double local_zero = 0.0;
			int j;

			enter_task(VECT_GRADIENT);

			for(j=s; j<e; j++)
			{
				double new_gj = b[j] - Aiterate[j];
				local_zero += (new_gj - gradient[j]) * (new_gj - gradient[j]);
				gradient[j] = new_gj;
			}

			#pragma omp atomic
				*zero += local_zero;

			exit_task();
			log_err(SHOW_TASKINFO, "b - Ax with SDC_CHECK part %d finished = %e, check = %e\n", i, norm(e-s, &(gradient[s])), local_zero);
		}
	}

	#pragma omp task in(*zero) out(*behaviour) inout(*prev_error, *err_sq) firstprivate(err_data, save, normAb) label(check_sdc) priority(100) no_copy_deps
	{
		// we should have *bp - *xAp - err_sq = 0
		*zero = sqrt(*zero);
		
		int sdc = !isfinite(*zero) || *zero > threshold * normAb;
		log_sdc(*zero/normAb, sdc);

		*zero  = 0.0;

		if(sdc)
		{
			// if twice we're stuck : restart
			if(*prev_error)
			{
				*behaviour = RESTART_CHECKPOINT;
				*err_sq = INFINITY;
				log_err(SHOW_DBGINFO, "TWO SUCCESSIVE SDCs : RELOAD CHECKPOINT + CG RESTART\n");
			}
			else
			{
				*behaviour = RELOAD_CHECKPOINT;
				*err_sq = *err_data->save_err_sq;
				log_err(SHOW_DBGINFO, "SILENT DATA CORRUPTION DETECTED : RELOAD CHECKPOINT\n");
			}

			*prev_error = !*prev_error;
		}
		else if(save)
		{
			*behaviour = SAVE_CHECKPOINT;
			*prev_error = 0;
			*err_data->save_err_sq = *err_sq;
		}
		else
			*behaviour = DO_NOTHING;
	}

	checkpoint_vectors(n, err_data, behaviour, iterate, gradient, p, Ap);
}

void check_sdc_alpha_invariant(const int save, detect_error_data *err_data, const double *b, double *iterate, double *gradient, double *p, double *Ap, double *err_sq, double *alpha, const double threshold)
{
	int i;
	double *bp = &(err_data->helper_1), *xAp = &(err_data->helper_2);
	int *behaviour = &(err_data->error_detected), *prev_error = &(err_data->prev_error);

	for(i=0; i < nb_blocks; i ++)
	{
		int s = get_block_start(i), e = get_block_end(i);
		if(e > n)
			e = n;

		#pragma omp task in(b[s:e-1], p[s:e-1]) concurrent(*bp) firstprivate(s,e) label(check_error_bp) priority(0) no_copy_deps
		{
			#pragma omp atomic
				*bp += scalar_product(e-s, b+s, p+s);
		}

		#pragma omp task in(iterate[s:e-1], Ap[s:e-1]) concurrent(*xAp) firstprivate(s,e) label(check_error_xAp) priority(0) no_copy_deps
		{
			#pragma omp atomic
				*xAp += scalar_product(e-s, iterate+s, Ap+s);
		}
	}


	#pragma omp task in(*bp, *xAp) out(*behaviour) inout(*prev_error, *alpha, *err_sq) firstprivate(err_data, save) label(check_sdc) priority(100) no_copy_deps
	{
		// we should have *bp - *xAp - err_sq = 0
		
		double zero = fabs(*bp - *xAp - *err_sq), max = fabs(*bp);

		if(fabs(*xAp) > max)
			max = fabs(*xAp);

		if(fabs(*err_sq) > max)
			max = fabs(*err_sq);
		
		int sdc = !isfinite(zero) || zero > threshold * max;
		log_sdc(zero/max, sdc);

		*bp  = 0.0;
		*xAp = 0.0;

		if(sdc)
		{
			// if twice we're stuck : restart
			if(*prev_error)
			{
				*behaviour = RESTART_CHECKPOINT;
				*err_sq = INFINITY;
				*alpha = 0.0;
				log_err(SHOW_DBGINFO, "TWO SUCCESSIVE SDCs : RELOAD CHECKPOINT + CG RESTART\n");
			}
			else
			{
				*behaviour = RELOAD_CHECKPOINT;
				*err_sq = *err_data->save_err_sq;
				*alpha  = *err_data->save_alpha;
				log_err(SHOW_DBGINFO, "SILENT DATA CORRUPTION DETECTED : RELOAD CHECKPOINT\n");
			}

			*prev_error = !*prev_error;
		}
		else if(save)
		{
			*behaviour = SAVE_CHECKPOINT;
			*prev_error = 0;
			*err_data->save_err_sq = *err_sq;
			*err_data->save_alpha = *alpha;
		}
		else
			*behaviour = DO_NOTHING;
	}

	checkpoint_vectors(n, err_data, behaviour, iterate, gradient, p, Ap);
}

void check_sdc_p_Ap_orthogonal(const int save, detect_error_data *err_data, double *iterate, double *gradient, double *p, double *Ap, double *err_sq, const double threshold)
{
	int i;
	double *pAp = &(err_data->helper_1), *np = &(err_data->helper_2), *nAp = &(err_data->helper_3);
	int *behaviour = &(err_data->error_detected), *prev_error = &(err_data->prev_error);

	for(i=0; i < nb_blocks; i ++)
	{
		int s = get_block_start(i), e = get_block_end(i);
		if(e > n)
			e = n;

		#pragma omp task in(Ap[s:e-1]) concurrent(*nAp) firstprivate(s,e) label(check_error_nAp) priority(0) no_copy_deps
		{
			#pragma omp atomic
				*nAp += norm(e-s, Ap+s);
		}

		#pragma omp task in(p[s:e-1], Ap[s:e-1]) concurrent(*np, *pAp) firstprivate(s,e) label(check_error_np_pAp) priority(0) no_copy_deps
		{
			#pragma omp atomic
				*np += norm(e-s, p+s);
			#pragma omp atomic
				*pAp += scalar_product(e-s, p+s, Ap+s);
		}
	}


	#pragma omp task in(*np, *nAp, *pAp) out(*behaviour) inout(*prev_error, *err_sq) firstprivate(err_data, save) label(check_sdc) priority(100) no_copy_deps
	{
		// we should have *bp - *xAp - err_sq = 0
		
		double zero = fabs(*pAp), scale = sqrt(*np * *nAp);

		int sdc = !isfinite(zero) || zero > threshold * scale;
		log_sdc(zero/scale, sdc);

		*np  = 0.0;
		*nAp = 0.0;
		*pAp = 0.0;

		if(sdc)
		{
			*err_sq = *err_data->save_err_sq;
			// if twice we're stuck : restart
			if(*prev_error)
			{
				*behaviour = RESTART_CHECKPOINT;
				log_err(SHOW_DBGINFO, "TWO SUCCESSIVE SDCs : RELOAD CHECKPOINT + CG RESTART\n");
			}
			else
			{
				*behaviour = RELOAD_CHECKPOINT;
				log_err(SHOW_DBGINFO, "SILENT DATA CORRUPTION DETECTED : RELOAD CHECKPOINT\n");
			}

			*prev_error = !*prev_error;
		}
		else if(save)
		{
			*behaviour = SAVE_CHECKPOINT;
			*prev_error = 0;
			*err_data->save_err_sq = *err_sq;
		}
		else
			*behaviour = DO_NOTHING;
	}

	checkpoint_vectors(n, err_data, behaviour, iterate, gradient, p, NULL);
}

