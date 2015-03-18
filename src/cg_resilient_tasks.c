
void scalar_product_task(const double *p, const double *Ap, double* r)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- <p, Ap>
		#pragma omp task concurrent(*r, p[s:e-1], Ap[s:e-1]) firstprivate(s, e) label(dotp) priority(10) no_copy_deps
		{
			double local_r = 0, page_r;
			const int fbs = get_failblock_size(), mask = MASK_A_P | (1 << get_data_vectptr(p));
			int j, k, page;

			enter_task(NORM_A_P);

			for(j=s, page = s >> get_log2_failblock_size(); j<e; j+=fbs, page++)
			{
				if( should_skip_block(page, mask) )
					continue;

				page_r = 0.0;
				for(k=j; k < e && k < j + fbs ; k++)
					page_r += p[k] * Ap[k];

				if( !check_block(page, mask) )
					local_r += page_r;
			}

			#pragma omp atomic
				*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow scalar product <p[%d], Ap> block %d finished = %e\n", get_data_vectptr(p), world_block(i), local_r);
			exit_task();
		}
	}
}

void norm_task(const double *v, double* r)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- || v ||
		#pragma omp task concurrent(*r, v[s:e-1]) firstprivate(s, e) label(norm) priority(10) no_copy_deps
		{
			double local_r = 0, page_r;
			const int fbs = get_failblock_size();
			int j, k, page;

			enter_task(NORM_GRADIENT);

			for(j=s, page = s >> get_log2_failblock_size(); j<e; j+=fbs, page++)
			{
				if( should_skip_block(page, MASK_GRADIENT) )
					continue;

				page_r = 0;
				for(k=j; k < e && k < j + fbs ; k++)
					page_r += v[k] * v[k];

				if( !check_block(page, MASK_GRADIENT) )
					local_r += page_r;
			}

			#pragma omp atomic
				*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow square norm || g || part %d finished = %e\n", world_block(i), local_r);
			exit_task();
		}
	}
}

void update_gradient(double *gradient, double *Ap, double *alpha, char *wait_for_iterate UNUSED)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		#pragma omp task in(*alpha, Ap[s:e-1]) concurrent(*wait_for_iterate) inout(gradient[s:e-1]) firstprivate(s, e) label(update_gradient) priority(10) no_copy_deps
		{
			const int fbs = get_failblock_size();
			int j, k, page;

			enter_task(VECT_GRADIENT);

			for(j=s, page = s >> get_log2_failblock_size(); j<e; j+=fbs, page++)
			{
				if( should_skip_block(page, MASK_GRADIENT | MASK_A_P) )
					continue;
			
				for(k=j; k<j+fbs; k++)
					gradient[k] -= (*alpha) * Ap[k];

				check_block(page, MASK_GRADIENT | MASK_A_P);
			}

			log_err(SHOW_TASKINFO, "Updating gradient part %d finished = %e with alpha = %e\n", world_block(i), norm(e-s, &(gradient[s])), *alpha);
			exit_task();
		}
	}
}

void recompute_gradient_mvm(const Matrix *A, double *iterate, char *wait_for_iterate UNUSED, char *wait_for_mvm UNUSED, double *Aiterate)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// Aiterate <- A * iterate
		#pragma omp task in(iterate[s:e-1], *wait_for_iterate) concurrent(*wait_for_mvm) out(Aiterate[s:e-1]) firstprivate(s, e) label(AxIt) priority(10) no_copy_deps
		{
			int j, k, l, page = s >> get_log2_failblock_size(), skips;
			const int fbs = get_failblock_size();

			enter_task(VECT_A_ITERATE);

			for(j=s; j<e; j+=fbs, page++)
			{
				if( is_skipped_block( page, MASK_A_ITERATE ) )
					continue;

				skips = count_neighbour_faults( page, MASK_ITERATE );
				if( skips )
				{
					mark_to_skip( page, MASK_A_ITERATE );
					continue;
				}

				for(l=j; l<j+fbs && l < e; l++)
				{
					Aiterate[l] = 0;
					
					for(k=A->r[l]; k < A->r[l+1] ; k++)
						Aiterate[l] += A->v[k] * iterate[ A->c[k] ];
				}


				if( skips != count_neighbour_faults( page, MASK_ITERATE ) )
					mark_to_skip( page, MASK_A_ITERATE );
			}

			log_err(SHOW_TASKINFO, "A * x part %d finished = %e\n", world_block(i), norm(e-s, &(Aiterate[s])));
			exit_task();
		}
	}
}

void recompute_gradient_update(double *gradient, char *wait_for_mvm UNUSED, double *Aiterate, const double *b)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);


		// gradient <- b - Aiterate
		#pragma omp task in(Aiterate[s:e-1]) concurrent(*wait_for_mvm) out(gradient[s:e-1]) firstprivate(s, e) label(b-AxIt) priority(10) no_copy_deps
		{
			// Can't afford to propagate errors from Ax to g ; if they come from a block i of x
			// then it will be impossible for x_i to be recovered from g_i, x_j (j!=i)

			const int fbs = get_failblock_size();
			int j, k, page;

			enter_task(VECT_GRADIENT);

			for(j=s, page = s >> get_log2_failblock_size(); j<e; j+=fbs, page++)
			{
				if( should_skip_block(page, MASK_A_ITERATE) )
					continue;

				for (k=j; k< j + fbs; k++)
					gradient[k] = b[k] - Aiterate[k] ;

				check_block(page, MASK_A_ITERATE);
			}

			log_err(SHOW_TASKINFO, "b - Ax part %d finished = %e\n", world_block(i), norm(e-s, &(gradient[s])));
			exit_task();
		}
	}
}

void update_p(double *p, double *old_p, char *wait_for_p UNUSED, char *wait_for_p2 UNUSED, double *gradient, double *beta)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// p <- beta * old_p + gradient
		#pragma omp task in(*beta, gradient[s:e-1]) in(old_p[s:e-1]) out(p[s:e-1]) concurrent(*wait_for_p, *wait_for_p2) firstprivate(s, e) label(update_p) priority(10) no_copy_deps
		{
			const int fbs = get_failblock_size(), mask = MASK_GRADIENT | (1 << get_data_vectptr(old_p));
			int j, k, page, errcount = 0;

			enter_task_vect(p);

			for(j=s, page = s >> get_log2_failblock_size(); j<e; j+=fbs, page++)
			{
				if( should_skip_block(page, mask) )
				{
					errcount ++;
					continue;
				}

				for (k=j; k< j + fbs; k++)
					p[k] = (*beta) * old_p[k] + gradient[k];

				errcount += check_block(page, mask);
			}

			log_err(SHOW_TASKINFO, "Updating p[%d from %d] part %d finished = %e with beta = %e\n", get_data_vectptr(p), get_data_vectptr(old_p), world_block(i), norm(e-s, &(p[s])), *beta);
			exit_task();

			// Ap (= A * old_p at this time) might be needed for old_p recovery, and before Ap contains A * new_p
			if( errcount )
				save_oldAp_for_old_p_recovery(&mp, old_p, s, e);
		}
	}
}

void compute_Ap(const Matrix *A, double *p, char *wait_for_p UNUSED, char *wait_for_mvm UNUSED, double *Ap)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// Ap <- A * p
		#pragma omp task in(p[s:e-1], *wait_for_p) concurrent(*wait_for_mvm) out(Ap[s:e-1]) firstprivate(s, e) label(Axp) priority(20) no_copy_deps
		{
			int j, k, l, page = s >> get_log2_failblock_size(), skips;
			const int fbs = get_failblock_size(), mask = 1 << get_data_vectptr(p);

			enter_task(VECT_A_P);

			for(j=s; j<e; j+=fbs, page++)
			{
				skips = count_neighbour_faults( page, mask );
				if( skips )
					mark_to_skip( page, MASK_A_P );

				for(l=j; l<j+fbs && l < e; l++)
				{
					Ap[l] = 0;
					
					for(k=A->r[l]; k < A->r[l+1] ; k++)
						Ap[l] += A->v[k] * p[ A->c[k] ];
				}
				
				if( skips != count_neighbour_faults( page, mask ) )
					mark_to_skip( page, FAIL_A_P );
			}

			log_err(SHOW_TASKINFO, "A * p[%d] part %d finished = %e\n", get_data_vectptr(p), world_block(i), norm(e-s, &(Ap[s])));
			exit_task();
		}
	}
}

void update_iterate(double *iterate, char *wait_for_iterate UNUSED, double *wait_for_beta, double *p, double *alpha)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// iterate <- iterate - alpha * p
		#pragma omp task in(*alpha, p[s:e-1], *wait_for_beta) inout(iterate[s:e-1]) concurrent(*wait_for_iterate) firstprivate(s, e) label(update_iterate) priority(5) no_copy_deps
		{
			enter_task(VECT_ITERATE);

			const int fbs = get_failblock_size(), mask = (1 << get_data_vectptr(p));
			int j, k, page;

			for(j=s, page = s >> get_log2_failblock_size(); j<e; j+=fbs, page++)
			{
				if( should_skip_block(page, mask) )
					continue;

				for(k=j; k<j+fbs; k++)
					iterate[k] += (*alpha) * p[k];

				check_block(page, mask);
			}

			log_err(SHOW_TASKINFO, "Updating it (from p[%d]) part %d finished = %e with alpha = %e\n", get_data_vectptr(p), world_block(i), norm(e-s, &(iterate[s])), *alpha);
			exit_task();
		}
	}
}

