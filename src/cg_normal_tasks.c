

void scalar_product_task(const double *p, const double *Ap, double* r)
{
	int i;
	for (i = 0; i < nb_blocks; i++)
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- <p, Ap>
		PRAGMA_TASK(concurrent(*r) in(p[s:e-1], Ap[s:e-1]) firstprivate(s, e), dotp, 10)
		{
			double local_r = 0;
			int k;
			for (k = s; k < e; k++)
				local_r += p[k] * Ap[k];

			#pragma omp atomic
				*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow scalar product <p[%d], Ap> block %d finished = %e\n", get_data_vectptr(p), i, local_r);
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
		PRAGMA_TASK(concurrent(*r) in(v[s:e-1]) firstprivate(s, e), norm, 10)
		{
			double local_r = 0;
			int k;
			for (k = s; k < e; k++)
				local_r += v[k] * v[k];

			#pragma omp atomic
				*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow square norm || g || part %d finished = %e\n", i, local_r);
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
			int k;
			for (k = s; k < e; k++)
				gradient[k] -= (*alpha) * Ap[k];

			log_err(SHOW_TASKINFO, "Updating gradient part %d finished = %e with alpha = %e\n", i, norm(e-s, &(gradient[s])), *alpha);
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
			int k, l;
			for (l = s; l < e; l++)
			{
				Aiterate[l] = 0;

				for (k = A->r[l]; k < A->r[l+1] ; k++)
					Aiterate[l] += A->v[k] * iterate[A->c[k]];
			}

			log_err(SHOW_TASKINFO, "A * x part %d finished = %e\n", i, norm(e-s, &(Aiterate[s])));
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
			int k;
			for (k = s; k < e; k++)
				gradient[k] = b[k] - Aiterate[k] ;

			log_err(SHOW_TASKINFO, "b - Ax part %d finished = %e\n", i, norm(e-s, &(gradient[s])));
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
			int k;
			for (k = s; k < e; k++)
				p[k] = (*beta) * old_p[k] + gradient[k];

			log_err(SHOW_TASKINFO, "Updating p[%d from %d] part %d finished = %e with beta = %e\n", get_data_vectptr(p), get_data_vectptr(old_p), i, norm(e-s, &(p[s])), *beta);
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
			int k, l;
			for (l = s; l < e; l++)
			{
				Ap[l] = 0;

				for (k = A->r[l]; k < A->r[l+1] ; k++)
					Ap[l] += A->v[k] * p[A->c[k]];
			}

			log_err(SHOW_TASKINFO, "A * p[%d] part %d finished = %e\n", get_data_vectptr(p), i, norm(e-s, &(Ap[s])));
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
			int k;
			for (k = s; k < e; k++)
				iterate[k] += (*alpha) * p[k];

			log_err(SHOW_TASKINFO, "Updating it (from p[%d]) part %d finished = %e with alpha = %e\n", get_data_vectptr(p), i, norm(e-s, &(iterate[s])), *alpha);
		}
	}
}

