
void apply_preconditioner(const double *g, double *z, Precond *M, char **wait_for_precond UNUSED)
{
	int i, fbs = get_failblock_size(), log2fbs = get_log2_failblock_size();

	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		#pragma omp task in(g[s:e-1], M->S[s>>log2fbs:(e>>log2fbs)-1], M->N[s>>log2fbs:(e>>log2fbs)-1], *(wait_for_precond[i])) out(z[s:e-1]) \
					firstprivate(i, s, e) label(precondition) priority(50) no_copy_deps
		{
			int j, page;
			double *x = malloc(fbs * sizeof(double));

			if( ! x )
			{
				fprintf(stderr, "Malloc of size %d B failed in preconditioner (parallel block %d)\n", (int)(fbs * sizeof(double)), i);
				exit(EXIT_FAILURE);
			}

			for(j=s, page=s>>log2fbs; j < e ; j += fbs, page++)
			{
				if( j + fbs > e )
					fbs = e - j;

				cs_ipvec(fbs, M->S[page]->Pinv, &g[j], x) ;	// x = P*g
				cs_lsolve(M->N[page]->L, x) ;		// x = L\x
				cs_ltsolve(M->N[page]->L, x) ;		// x = L'\x
				cs_pvec(fbs, M->S[page]->Pinv, x, &z[j]) ;	// z = P'*x
			}

			free(x);

			// remove preconditioning :
			//for(i=s; i<e; i++) z[i] = g[i];
			log_err(SHOW_TASKINFO, "Preconditioning block %d [%d,%d] finished\n", i, s>>log2fbs, (e>>log2fbs)-1);
		}
	}
}

void scalar_product_task(const double *v, const double *u, double* r, const int task_name UNUSED)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// r <- <v, u>
		#pragma omp task concurrent(*r, v[s:e-1], u[s:e-1]) firstprivate(s, e) label(dotp) priority(10) no_copy_deps
		{
			double local_r = 0;
			int k;
			for(k=s; k<e; k++)
				local_r += v[k] * u[k];

			#pragma omp atomic
				*r += local_r;

			log_err(SHOW_TASKINFO, "Blockrow scalar product <%s, %s> block %d finished = %e\n", vect_name(get_data_vectptr(u)), vect_name(get_data_vectptr(v)), i, local_r);
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

void recompute_gradient(double *gradient, const Matrix *A, double *iterate, char *wait_for_iterate UNUSED, char *wait_for_mvm UNUSED, double *Aiterate, const double *b)
{
	int i;
	for(i=0; i < nb_blocks; i ++ )
	{
		int s = get_block_start(i), e = get_block_end(i);

		// Aiterate <- A * iterate
		#pragma omp task in(iterate[0:A->m-1], *wait_for_iterate) concurrent(*wait_for_mvm) out(Aiterate[s:e-1]) firstprivate(s, e) label(AxIt) priority(10) no_copy_deps
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

		// gradient <- b - Aiterate
		#pragma omp task in(Aiterate[s:e-1]) concurrent(*wait_for_mvm) out(gradient[s:e-1]) firstprivate(s, e) label(b-AxIt) priority(10) no_copy_deps
		{
			int k;
			for(k=s; k<e; k++)
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
		#pragma omp task in(p[0:A->m-1], *wait_for_p) concurrent(*wait_for_mvm) out(Ap[s:e-1]) firstprivate(s, e) label(Axp) priority(20) no_copy_deps
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

