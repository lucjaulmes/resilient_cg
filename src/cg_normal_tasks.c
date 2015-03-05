
void scalar_product_task(const double *p, const double *Ap, double* r)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- <p, Ap>
		#pragma omp task concurrent(*r, p[s:e-1], Ap[s:e-1]) firstprivate(s, e) label(dotp) priority(10) no_copy_deps
		{
			double local_r = 0;
			int k;
			for(k=s; k < e; k++)
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
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- || v ||
		#pragma omp task concurrent(*r, v[s:e-1]) firstprivate(s, e) label(norm) priority(10) no_copy_deps
		{
			double local_r = 0;
			int k;
			for(k=s; k<e; k++)
				local_r += v[k] * v[k];

			#pragma omp atomic
				*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow square norm || g || part %d finished = %e\n", i, local_r);
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
			int k;
			for(k=s; k<e; k++)
				gradient[k] -= (*alpha) * Ap[k];

			log_err(SHOW_TASKINFO, "Updating gradient part %d finished = %e with alpha = %e\n", i, norm(e-s, &(gradient[s])), *alpha);
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
			int k, l;
			for(l=s; l<e; l++)
			{
				Aiterate[l] = 0;
				
				for(k=A->r[l]; k < A->r[l+1] ; k++)
					Aiterate[l] += A->v[k] * iterate[ A->c[k] ];
			}

			log_err(SHOW_TASKINFO, "A * x part %d finished = %e\n", i, norm(e-s, &(Aiterate[s])));
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
			int k;
			for (k=s; k<e; k++)
				gradient[k] = b[k] - Aiterate[k] ;

			log_err(SHOW_TASKINFO, "b - Ax part %d finished = %e\n", i, norm(e-s, &(gradient[s])));
		}
	}
}

void update_p(double *p, double *old_p, char *wait_for_p UNUSED, double *gradient, double *beta)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// p <- beta * old_p + gradient
		#pragma omp task in(*beta, gradient[s:e-1]) in(old_p[s:e-1]) out(p[s:e-1]) concurrent(*wait_for_p) firstprivate(s, e) label(update_p) priority(10) no_copy_deps
		{
			int k;
			for (k=s; k<e; k++)
				p[k] = (*beta) * old_p[k] + gradient[k];

			log_err(SHOW_TASKINFO, "Updating p[%d from %d] part %d finished = %e with beta = %e\n", get_data_vectptr(p), get_data_vectptr(old_p), i, norm(e-s, &(p[s])), *beta);
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
			int k, l;
			for(l=s; l<e; l++)
			{
				Ap[l] = 0;
				
				for(k=A->r[l]; k < A->r[l+1] ; k++)
					Ap[l] += A->v[k] * p[ A->c[k] ];
			}

			log_err(SHOW_TASKINFO, "A * p[%d] part %d finished = %e\n", get_data_vectptr(p), i, norm(e-s, &(Ap[s])));
		}
	}
}

void update_iterate(double *iterate, char *wait_for_iterate UNUSED, double *p, double *alpha)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// iterate <- iterate - alpha * p
		#pragma omp task in(*alpha, p[s:e-1]) inout(iterate[s:e-1]) concurrent(*wait_for_iterate) firstprivate(s, e) label(update_iterate) priority(5) no_copy_deps
		{
			int k;
			for(k=s; k<e; k++)
				iterate[k] += (*alpha) * p[k];

			log_err(SHOW_TASKINFO, "Updating it (from p[%d]) part %d finished = %e with alpha = %e\n", get_data_vectptr(p), i, norm(e-s, &(iterate[s])), *alpha);
		}
	}
}

