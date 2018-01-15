
void scalar_product_task(const double *p, const double *Ap, double* r)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- <p, Ap>
		PRAGMA_TASK(concurrent(*r, p[s:e-1], Ap[s:e-1]) firstprivate(i, s, e), dotp, 10)
		{
			double local_r = 0, page_r;
			const int mask = MASK_A_P | (1 << get_data_vectptr(p));
			int j, k, page;
			#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
			int shared_block = 0;
			#endif

			enter_task(NORM_A_P);

			for (j = s, page = s / failblock_size_dbl; j < e; page++, j = k)
			{
				int next_j = round_up(j + 1, failblock_size_dbl);
				if (next_j > e)
					next_j = e;

				if (should_skip_block(page, mask))
				{
					k = next_j;
					continue;
				}

				page_r = 0.0;
				for (k = j; k < next_j; k++)
					page_r += p[k] * Ap[k];

				#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
				if (!check_block(page, mask, &shared_block))
				{
					local_r += page_r;
					if (shared_block)
					{
						if (j == s)
							mp.shared_page_reductions[2 * i] = page_r;
						else if (k == e)
							mp.shared_page_reductions[2 * i + 1] = page_r;
					}
				}
				#else
				if (!check_block(page, mask, NULL))
					local_r += page_r;
				#endif
			}

			#pragma omp atomic
				*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow scalar product <p[%d], Ap> block %d finished = %e\n", get_data_vectptr(p), i, local_r);
			exit_task();
		}
	}
}

void norm_task(const double *v, double* r)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- || v ||
		PRAGMA_TASK(concurrent(*r, v[s:e-1]) firstprivate(i, s, e), norm, 10)
		{
			double local_r = 0, page_r;
			int j, k, page;
			#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
			int shared_block = 0;
			#endif

			enter_task(NORM_GRADIENT);

			for (j = s, page = s / failblock_size_dbl; j < e; page++, j = k)
			{
				int next_j = round_up(j + 1, failblock_size_dbl);
				if (next_j > e)
					next_j = e;

				if (should_skip_block(page, MASK_GRADIENT))
				{
					k = next_j;
					continue;
				}

				page_r = 0;
				for (k = j; k < next_j; k++)
					page_r += v[k] * v[k];

				#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
				if (!check_block(page, MASK_GRADIENT, &shared_block))
				{
					local_r += page_r;
					if (shared_block)
					{
						if (j == s)
							mp.shared_page_reductions[2 * i] = page_r;
						else if (k == e)
							mp.shared_page_reductions[2 * i + 1] = page_r;
					}
				}
				#else
				if (!check_block(page, MASK_GRADIENT, NULL))
					local_r += page_r;
				#endif
			}

			#pragma omp atomic
				*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow square norm || g || part %d finished = %e\n", i, local_r);
			exit_task();
		}
	}
}

void update_gradient(double *gradient, double *Ap, double *alpha UNUSED)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		PRAGMA_TASK(in(*alpha, Ap[s:e-1]) inout(gradient[s:e-1]) firstprivate(s, e), update_gradient, 10)
		{
			int j, k, page;

			enter_task(VECT_GRADIENT);

			for (j = s, page = s / failblock_size_dbl; j < e; page++, j = k)
			{
				int next_j = round_up(j + 1, failblock_size_dbl);
				if (next_j > e)
					next_j = e;

				if (should_skip_block(page, MASK_GRADIENT | MASK_A_P))
				{
					k = next_j;
					continue;
				}

				for (k = j; k < next_j; k++)
					gradient[k] -= (*alpha) * Ap[k];

				check_block(page, MASK_GRADIENT | MASK_A_P, NULL);
			}

			log_err(SHOW_TASKINFO, "Updating gradient part %d finished = %e with alpha = %e\n", i, norm(e-s, &(gradient[s])), *alpha);
			exit_task();
		}
	}
}

void recompute_gradient_mvm(const Matrix *A, double *iterate UNUSED, double *Aiterate)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// Aiterate <- A * iterate
		PRAGMA_TASK(in(ALL_BLOCKS(iterate)) out(Aiterate[s:e-1]) firstprivate(s, e), AxIt, 10)
		{
			int j, k, l, page, skips;

			enter_task(VECT_A_ITERATE);

			for (j = s, page = s / failblock_size_dbl; j < e; page++, j = l)
			{
				int next_j = round_up(j + 1, failblock_size_dbl);
				if (next_j > e)
					next_j = e;

				if (is_skipped_block(page, MASK_A_ITERATE))
				{
					l = next_j;
					continue;
				}

				skips = count_neighbour_faults(page, MASK_ITERATE);
				if (skips)
				{
					mark_to_skip(page, MASK_A_ITERATE);
					l = next_j;
				}

				for (l = j; l < next_j; l++)
				{
					Aiterate[l] = 0;

					for (k = A->r[l]; k < A->r[l+1] ; k++)
						Aiterate[l] += A->v[k] * iterate[A->c[k]];
				}


				if (skips != count_neighbour_faults(page, MASK_ITERATE))
					mark_to_skip(page, MASK_A_ITERATE);
			}

			log_err(SHOW_TASKINFO, "A * x part %d finished = %e\n", i, norm(e-s, &(Aiterate[s])));
			exit_task();
		}
	}
}

void recompute_gradient_update(double *gradient UNUSED, double *Aiterate, const double *b)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// gradient <- b - Aiterate
		PRAGMA_TASK(in(Aiterate[s:e-1]) out(gradient[s:e-1]) firstprivate(s, e), b-AxIt, 10)
		{
			// Can't afford to propagate errors from Ax to g ; if they come from a block i of x
			// then it will be impossible for x_i to be recovered from g_i, x_j (where j != i)

			int j, k, page;

			enter_task(VECT_GRADIENT);

			for (j = s, page = s / failblock_size_dbl; j < e; page++, j = k)
			{
				int next_j = round_up(j + 1, failblock_size_dbl);
				if (next_j > e)
					next_j = e;

				if (should_skip_block(page, MASK_A_ITERATE))
				{
					k = next_j;
					continue;
				}

				for (k = j; k < next_j; k++)
					gradient[k] = b[k] - Aiterate[k] ;

				check_block(page, MASK_A_ITERATE, NULL);
			}

			log_err(SHOW_TASKINFO, "b - Ax part %d finished = %e\n", i, norm(e-s, &(gradient[s])));
			exit_task();
		}
	}
}

void update_p(double *p, double *old_p UNUSED, double *gradient, double *beta)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// p <- beta * old_p + gradient
		PRAGMA_TASK(in(*beta, gradient[s:e-1], old_p[s:e-1]) out(p[s:e-1]) firstprivate(s, e), update_p, 10)
		{
			const int mask = MASK_GRADIENT | (1 << get_data_vectptr(old_p));
			int j, k, page, errcount = 0;

			enter_task_vect(p);

			for (j = s, page = s / failblock_size_dbl; j < e; page++, j = k)
			{
				int next_j = round_up(j + 1, failblock_size_dbl);
				if (next_j > e)
					next_j = e;

				if (should_skip_block(page, mask))
				{
					errcount ++;
					k = next_j;
					continue;
				}

				for (k = j; k < next_j; k++)
					p[k] = (*beta) * old_p[k] + gradient[k];

				errcount += check_block(page, mask, NULL);
			}

			log_err(SHOW_TASKINFO, "Updating p[%d from %d] part %d finished = %e with beta = %e\n", get_data_vectptr(p), get_data_vectptr(old_p), i, norm(e-s, &(p[s])), *beta);
			exit_task();

			// Ap (= A * old_p at this time) might be needed for old_p recovery, and before Ap contains A * new_p
			if (errcount)
				save_oldAp_for_old_p_recovery(&mp, old_p, s, e);
		}
	}
}

void compute_Ap(const Matrix *A, double *p UNUSED, double *Ap)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// Ap <- A * p
		PRAGMA_TASK(in(ALL_BLOCKS(p)) out(Ap[s:e-1]) firstprivate(s, e), Axp, 20)
		{
			int j, k, l, page, skips;
			const int mask = 1 << get_data_vectptr(p);

			enter_task(VECT_A_P);

			for (j = s, page = s / failblock_size_dbl; j < e; page++, j = l)
			{
				int next_j = round_up(j + 1, failblock_size_dbl);
				if (next_j > e)
					next_j = e;

				skips = count_neighbour_faults(page, mask);
				if (skips)
				{
					mark_to_skip(page, MASK_A_P);
					k = next_j;
				}

				for (l = j; l < next_j; l++)
				{
					Ap[l] = 0;

					for (k = A->r[l]; k < A->r[l+1] ; k++)
						Ap[l] += A->v[k] * p[A->c[k]];
				}

				if (skips != count_neighbour_faults(page, mask))
					mark_to_skip(page, FAIL_A_P);
			}

			log_err(SHOW_TASKINFO, "A * p[%d] part %d finished = %e\n", get_data_vectptr(p), i, norm(e-s, &(Ap[s])));
			exit_task();
		}
	}
}

void update_iterate(double *iterate UNUSED, double *p, double *alpha)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// iterate <- iterate - alpha * p
		PRAGMA_TASK(in(*alpha, p[s:e-1]) inout(iterate[s:e-1]) firstprivate(s, e), update_iterate, 5)
		{
			enter_task(VECT_ITERATE);

			const int mask = (1 << get_data_vectptr(p));
			int j, k, page;

			for (j = s, page = s / failblock_size_dbl; j < e; page++, j = k)
			{
				int next_j = round_up(j + 1, failblock_size_dbl);
				if (next_j > e)
					next_j = e;

				if (should_skip_block(page, mask))
				{
					k = next_j;
					continue;
				}

				for (k = j; k < next_j; k++)
					iterate[k] += (*alpha) * p[k];

				check_block(page, mask, NULL);
			}

			log_err(SHOW_TASKINFO, "Updating it (from p[%d]) part %d finished = %e with alpha = %e\n", get_data_vectptr(p), i, norm(e-s, &(iterate[s])), *alpha);
			exit_task();
		}
	}
}

