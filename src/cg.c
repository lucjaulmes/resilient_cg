#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include "mpi.h"

#include "global.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"
#include "counters.h"

#include "cg.h"

magic_pointers mp;

void determine_mpi_neighbours(const Matrix *A, const int mpi_rank, const int mpi_size, int *first, int *last)
{
	// find our furthest neighbour in both direction
	int max_col = 0, min_col = A->n, i;
	for(i=0; i<mpi_zonesize[mpi_rank]; i++)
	{
		// using the fact that each A->c is sorted on A->r[i]..A->r[i+1]-1
		if( A->c[A->r[i]] < min_col )
			min_col = A->c[A->r[i]];

		if( A->c[A->r[i+1]-1] > max_col )
			max_col = A->c[A->r[i+1]-1];
	}

	// now check with which MPI block we communicate, depending on its rows
	// this is reflexive since the matrix is symmetric
	*first = -1;
	for(i=0; i<mpi_size; i++)
		// if self or there is an overlap, we are 'neighbours'
		if( i==mpi_rank || (! (mpi_zonestart[i] + mpi_zonesize[i] < min_col || max_col < mpi_zonestart[i]) ))
		{
			if( *first < 0 )
				*first = i;

			*last = i;
		}
}

void setup_exchange(const int mpi_rank, const int first, const int last, const int tag, double *v, MPI_Request v_req[])
{
	int i, j=0;

	// now check with which MPI block we communicate, depending on its rows
	// this is reflexive since the matrix is symmetric
	for(i=first; i<=last; i++)
		if(i!=mpi_rank)
		{
			// sends are always from mpi_rank start,size
			MPI_Send_init(v + mpi_zonestart[mpi_rank], mpi_zonesize[mpi_rank], MPI_DOUBLE, i, tag, MPI_COMM_WORLD, v_req+(j++));

			// recvs are always from i start,size
			MPI_Recv_init(v + mpi_zonestart[   i    ], mpi_zonesize[   i    ], MPI_DOUBLE, i, tag, MPI_COMM_WORLD, v_req+(j++));
		}
}
 
#if DUE && DUE != DUE_ROLLBACK
#include "cg_resilient_tasks.c"
#include "cg_recovery_tasks.c"
#else
#include "cg_normal_tasks.c"
#endif

#if SDC
#include "cg_sdc_checks.c"
#endif

#if CKPT
#include "cg_checkpoint.c"
#endif

#pragma omp task in(*old_err_sq) out(*beta) inout(*err_sq) label(compute_beta) priority(100) no_copy_deps
void compute_beta(double *err_sq, const double *old_err_sq, double *beta)
{
	// on first iterations of a (re)start, old_err_sq should be INFINITY so that beta = 0
	double loc_err_sq = *err_sq;
	MPI_Allreduce(&loc_err_sq, err_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	*beta = *err_sq / *old_err_sq;

	#if DUE
	int state = aggregate_skips();
	if( state & (MASK_GRADIENT | MASK_NORM_G | MASK_RECOVERY) )
		fprintf(stderr, "ERROR SUBSISTED PAST RECOVERY restart needed. At beta, g:%d, ||g||:%d\n", (state & MASK_GRADIENT) > 0, (state & MASK_NORM_G) > 0);
	#endif

	log_err(SHOW_TASKINFO, "Computing beta finished : err_sq = %e ; old_err_sq = %e ; beta = %e \n", *err_sq, *old_err_sq, *beta);
}

#pragma omp task inout(*normA_p_sq, *err_sq) out(*alpha, *old_err_sq) label(compute_alpha) priority(100) no_copy_deps
void compute_alpha(double *err_sq, double *normA_p_sq, double *old_err_sq, double *alpha)
{
	double loc_normA_p_sq = *normA_p_sq;
	MPI_Allreduce(&loc_normA_p_sq, normA_p_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	*alpha = *err_sq / *normA_p_sq ;
	*old_err_sq = *err_sq;

	#if DUE
	int state = aggregate_skips();
	#if DUE == DUE_LOSSY
	if( state )
	{
		log_err(SHOW_FAILINFO, "There was an error, restarting (eventual lossy x interpolation)");
		hard_reset(&mp);
	}
	#else
	if( state & (MASK_ITERATE | MASK_P | MASK_OLD_P | MASK_A_P | MASK_NORM_A_P | MASK_RECOVERY) )
		fprintf(stderr, "ERROR SUBSISTED PAST RECOVERY restart needed. At alpha, x:%d, p:%d, p':%d, Ap:%d, <p,Ap>:%d\n", (state & MASK_ITERATE) > 0, (state & MASK_P) > 0, (state & MASK_OLD_P) > 0, (state & MASK_A_P) > 0, (state & MASK_NORM_A_P) > 0);
	#endif
	#endif

	log_err(SHOW_TASKINFO, "Computing alpha finished : normA_p_sq = %+e ; err_sq = %e ; alpha = %e\n", *normA_p_sq, *err_sq, *alpha);

	// last consumer of these values : let's 0 them so the scalar product doesn't need to
	*err_sq = 0.0;
	*normA_p_sq = 0.0;
}

static inline void swap(double **v, double **w)
{
	double *swap = *v;
	*v = *w;
	*w = swap;
}

void solve_cg(const Matrix *A, const double *b, double *it_glob, double convergence_thres, double error_thres UNUSED)
{
	// do some memory allocations
	double norm_b, thres_sq;
	const int n = A->n;
	int r = -1, total_failures = 0, failures = 0;
	int do_update_gradient = 0;
	double *iterate, *p_glob, *old_p_glob, *p, *old_p, *Ap, normA_p_sq, *gradient, *Aiterate, err_sq = 0.0, old_err_sq = INFINITY, alpha = 0.0, beta = 0.0;
	char *wait_for_p = alloc_deptoken(), *wait_for_iterate = alloc_deptoken(), *wait_for_mvm = alloc_deptoken(), *start_rt_work = alloc_deptoken();
	#if CKPT == CKPT_IN_MEMORY
	double *save_it, *save_g, *save_p, *save_Ap, save_err_sq, save_alpha;
	int do_checkpoint = 0;
	#elif CKPT == CKPT_TO_DISK
	double save_err_sq, save_alpha;
	int do_checkpoint = 0;
	#endif
	#if SDC
	int do_check_sdc = CHECK_SDC_FREQ;
	#endif
	
	int first_mpix, last_mpix, count_mpix;
	determine_mpi_neighbours(A, mpi_rank, mpi_size, &first_mpix, &last_mpix);
	count_mpix = last_mpix - first_mpix;

	p_glob     = (double*)aligned_calloc( sizeof(double) << get_log2_failblock_size(), n * sizeof(double));
	old_p_glob = (double*)aligned_calloc( sizeof(double) << get_log2_failblock_size(), n * sizeof(double));

	p          = p_glob 	+ mpi_zonestart[mpi_rank];
	old_p      = old_p_glob	+ mpi_zonestart[mpi_rank];
	iterate    = it_glob	+ mpi_zonestart[mpi_rank];

	Ap         = (double*)aligned_calloc( sizeof(double) << get_log2_failblock_size(), mpi_zonesize[mpi_rank] * sizeof(double));
	gradient   = (double*)aligned_calloc( sizeof(double) << get_log2_failblock_size(), mpi_zonesize[mpi_rank] * sizeof(double));
	Aiterate   = (double*)aligned_calloc( sizeof(double) << get_log2_failblock_size(), mpi_zonesize[mpi_rank] * sizeof(double));

	#if CKPT == CKPT_IN_MEMORY
	save_it  = (double*)aligned_calloc( sizeof(double) << get_log2_failblock_size(), mpi_zonesize[mpi_rank] * sizeof(double));
	save_g   = (double*)aligned_calloc( sizeof(double) << get_log2_failblock_size(), mpi_zonesize[mpi_rank] * sizeof(double));
	save_p   = (double*)aligned_calloc( sizeof(double) << get_log2_failblock_size(), mpi_zonesize[mpi_rank] * sizeof(double));
	#if SDC == SDC_ORTHO
	save_Ap  = NULL;
	#else
	save_Ap  = (double*)aligned_calloc( sizeof(double) << get_log2_failblock_size(), mpi_zonesize[mpi_rank] * sizeof(double));
	#endif
	#endif

	// setting up communications for x and p exchanges
	MPI_Request x_req[2*count_mpix], p1_req[2*count_mpix], p2_req[2*count_mpix], *p_req;
	setup_exchange(mpi_rank, first_mpix, last_mpix, 1, it_glob,		x_req);
	setup_exchange(mpi_rank, first_mpix, last_mpix, 2, p_glob,		p1_req);
	setup_exchange(mpi_rank, first_mpix, last_mpix, 3, old_p_glob,	p2_req);

	p_req = p1_req;

	// some parameters pre-computed, and show some informations (borrow thres_sq to be out_buf, get norm in norm_b)
	thres_sq = norm(mpi_zonesize[mpi_rank], b);
    MPI_Allreduce(&thres_sq, &norm_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	log_err(SHOW_FAILINFO, "||b||=%f\n", norm_b);

	thres_sq = convergence_thres * convergence_thres * norm_b;
	{}//log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);

	detect_error_data err_data = (detect_error_data) {.error_detected = SAVE_CHECKPOINT, .prev_error = 0, .helper_1 = 0.0, .helper_2 = 0.0, .helper_3 = 0.0, .helper_4 = 0.0,
	#if CKPT == CKPT_IN_MEMORY
		.save_x = save_it, .save_g = save_g, .save_p = save_p, .save_Ap = save_Ap, .save_err_sq = &save_err_sq, .save_alpha = &save_alpha
	#elif CKPT == CKPT_TO_DISK
		.save_err_sq = &save_err_sq, .save_alpha = &save_alpha
	#endif
	};
	mp = (magic_pointers){.A = A, .b = b, .x = iterate, .p = p, .old_p = old_p, .g = gradient, .Ap = Ap, .Ax = Aiterate, .err_data = &err_data,
							.alpha = &alpha, .beta = &beta, .err_sq = &err_sq, .old_err_sq = &old_err_sq, .normA_p_sq = &normA_p_sq};

	#if SDC == SDC_GRADIENT
	int i;
	double norm_A = 0.0; // A spd : row norm <=> col norm , ||A|| = max || A_col i || forall i
	for(i=0; i<A->n; i++)
		norm_A = fmax(norm_A, sqrt(norm( A->r[i+1] - A->r[i], A->v + A->r[i] )));
		//norm_A += sqrt(norm( A->r[i+1] - A->r[i], A->v + A->r[i] ));

	MPI_Allreduce(&norm_A, &(err_data.helper_4), 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	err_data.helper_4 *= sqrt(norm_b);
	#endif

	setup_resilience(A, 6, &mp);
	start_measure();

	// real work starts here

	for(r=0; r < MAXIT ; r++)
	{
		if( --do_update_gradient > 0 )
		{
			// wait_for_iterate postpones recovery tasks until all update_x,g are done
			update_gradient(gradient, Ap, &alpha, wait_for_iterate);

			norm_task(gradient, &err_sq);

			// at this point, Ap = A * old_p
			#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
			recover_rectify_g(n, &mp, old_p, Ap, gradient, &err_sq, wait_for_iterate);
			#endif

			compute_beta(&err_sq, &old_err_sq, &beta);

			#if DUE == DUE_ASYNC
			recover_rectify_xk(n, &mp, iterate, wait_for_iterate);
			#endif
		}
		else
		{
			// wait_for_mvm postpones recovery tasks until all update_x,g are done
			// wait_for_iterate allows to wait for all parts of x before doing A*x

			// our initial guess is always 0, don't bother updating and exchanging it
			if( r > 0 )
			{
				update_iterate(iterate, wait_for_iterate, (char*)&alpha/* anything that's in(), won't be accessed*/, old_p, &alpha);

				#pragma omp task inout(it_glob[0:n-1]) in(*wait_for_iterate) out(*wait_for_mvm) firstprivate(x_req, count_mpix) label(exchange_x) priority(100) no_copy_deps
				{
					enter_task(MPI_X_EXCHANGE);
					//MPI_Allgatherv(MPI_IN_PLACE, 0/*ignored*/, MPI_DOUBLE, it_glob, mpi_zonesize, mpi_zonestart, MPI_DOUBLE, MPI_COMM_WORLD);

					MPI_Startall(2*count_mpix, x_req);
					MPI_Waitall(2*count_mpix, x_req, MPI_STATUSES_IGNORE);

					exit_task();
				}
			}

			// first part of recompute g : A*x
			recompute_gradient_mvm(A, it_glob, wait_for_iterate, wait_for_mvm, Aiterate);

			// second part of recompute g : b - Ax, sometimes more complicated or replaced due to SDC/DUE
			#if SDC == SDC_GRADIENT
			do_check_sdc -= RECOMPUTE_GRADIENT_FREQ;
			#if DUE == DUE_ROLLBACK
			if(failures)
			{
				do_update_gradient = do_checkpoint = do_check_sdc = 0;
				force_rollback(&err_data, iterate, gradient, old_p, Ap);
			}
			else
			#endif
			if( r > 0 && do_check_sdc == 0 )
			{
				do_checkpoint -= CHECK_SDC_FREQ;

				update_gradient(gradient, Ap, &alpha, wait_for_iterate);

				check_sdc_recompute_grad(n, do_checkpoint == 0, &err_data, b, iterate, gradient, old_p, Ap, wait_for_mvm, Aiterate, &old_err_sq, error_thres);
			}
			else
				recompute_gradient_update(gradient, wait_for_mvm, Aiterate, b);
			#else
			recompute_gradient_update(gradient, wait_for_mvm, Aiterate, b);
			#endif

			norm_task(gradient, &err_sq);

			#if CKPT
			if( r == 0 )
				force_checkpoint(&err_data, iterate, gradient, old_p, Ap);
			#endif

			#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
			recover_rectify_x_g(n, &mp, iterate, gradient, &err_sq, wait_for_mvm);
			#endif

			compute_beta(&err_sq, &old_err_sq, &beta);

			// after first beta, we are sure to have the first x, g, and checkpoint
			// so we can start injecting errors
			if( r == 0 )
				#pragma omp task in(beta) label(start_injection)
				start_error_injection();
		}

		// wait_for_p postpones communications 
		update_p(p, old_p, wait_for_p, start_rt_work, gradient, &beta);

		#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
		recover_rectify_p_early(n, &mp, p, old_p, wait_for_p, wait_for_iterate);
		#endif

		// if possible, execute the update iterate really late
		if( do_update_gradient > 0 )
			update_iterate(iterate, wait_for_iterate, wait_for_p, old_p, &alpha);

		#pragma omp task inout(p_glob[0:n-1]) in(*wait_for_p) out(*wait_for_mvm) firstprivate(p_req, count_mpix) label(exchange_p) priority(100) no_copy_deps
		{
			enter_task(MPI_P_EXCHANGE);
			//MPI_Allgatherv(MPI_IN_PLACE, 0/*ignored*/, MPI_DOUBLE, p_glob, mpi_zonesize, mpi_zonestart, MPI_DOUBLE, MPI_COMM_WORLD);

			MPI_Startall(2*count_mpix, p_req);
			MPI_Waitall(2*count_mpix, p_req, MPI_STATUSES_IGNORE);

			exit_task();
		}

		// In the hybrid version, there is an overlapping opportunity for the runtime work
		// which is during the communication (MPI_Allgartherv)
		//#pragma omp taskwait on(*start_rt_work, *wait_for_iterate)

		#if SDC == SDC_ORTHO
		// should happen in between p and Ap, to check if the new p and old Ap are orthogonal
		#if DUE == DUE_ROLLBACK
		if(failures)
		{
			do_checkpoint = do_check_sdc = 0;
			force_rollback(&err_data, iterate, gradient, p, Ap);
		}
		else
		#endif
		if( -- do_check_sdc == 0 )
		{
			do_checkpoint -= CHECK_SDC_FREQ ;
			check_sdc_p_Ap_orthogonal(n, do_checkpoint == 0, &err_data, iterate, gradient, p, Ap, &err_sq, error_thres);
		}
		#endif

		compute_Ap(A, p_glob, wait_for_p, wait_for_mvm, Ap);

		scalar_product_task(p, Ap, &normA_p_sq);

		#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
		recover_rectify_p_Ap(n, &mp, p, old_p, Ap, &normA_p_sq, wait_for_mvm, wait_for_iterate);
		#endif

		// swapping p's so we reduce pressure on the execution of the update iterate tasks
		// now output-dependencies is not conflicting with the next iteration but the one after
		#pragma omp taskwait on(alpha)
		{
			swap(&p, &old_p);
			swap(&p_glob, &old_p_glob);
			p_req = (p_req == p1_req) ? p2_req : p1_req;
			
			failures = check_errors_signaled();

			if( r > 0 )
				log_convergence(r-1, old_err_sq, failures);

			log_err(SHOW_TASKINFO, "\n\n");

			total_failures += failures;

			if( old_err_sq <= thres_sq )
				break;

			if( do_update_gradient <= 0 )
				do_update_gradient = RECOMPUTE_GRADIENT_FREQ;
			#if DUE == DUE_IN_PATH
			else
				recover_rectify_xk(n, &mp, iterate, (char*)&normA_p_sq);
			#endif
			#if CKPT
			if( do_checkpoint <= 0 )
				do_checkpoint = CHECKPOINT_FREQ;
			#endif
			#if SDC
			if( do_check_sdc <= 0 )
				do_check_sdc = CHECK_SDC_FREQ;
			#endif
		}

		// if we will recompute the gradient, prepare to listen for incoming iterate exchanges in compute_alpha
		compute_alpha(&err_sq, &normA_p_sq, &old_err_sq, &alpha);

		// should happen after p, Ap are ready and before (post-alpha) iterate and gradient updates
		// so just after (or just before) alpha basically
		#if CKPT && SDC == SDC_NONE // NB. this implies DUE_ROLLBACK
		if(failures)
		{
			do_checkpoint = 0;
			force_rollback(&err_data, iterate, gradient, old_p, Ap);
		}
		else if( --do_checkpoint == 0 )
			due_checkpoint(&err_data, iterate, gradient, old_p, Ap);
		#elif SDC == SDC_ALPHA
		#if DUE == DUE_ROLLBACK // ALPHA + ROLLBACK
		if(failures)
		{
			do_checkpoint = do_check_sdc = 0;
			force_rollback(&err_data, iterate, gradient, old_p, Ap);
		}
		else 
		#endif
		if( -- do_check_sdc == 0 )
		{
			do_checkpoint -= CHECK_SDC_FREQ;
			check_sdc_alpha_invariant(n, do_checkpoint == 0, &err_data, b, iterate, gradient, old_p, Ap, &old_err_sq, &alpha, error_thres);
		}
		#endif
	}

	#pragma omp taskwait 
	// end of the math, showing infos
	stop_measure();
	
	failures = check_errors_signaled();
	log_convergence(r-1, old_err_sq, failures);

	{}//log_out("\n\n------\nConverged at rank %d\n------\n\n", r);

	printf("CG method finished iterations:%d with error:%e (failures:%d)\n", r, sqrt((err_sq==0.0?old_err_sq:err_sq)/norm_b), total_failures);

	// stop resilience stuff that's still going on
	unset_resilience();

	// This is after solving, to be able to compute the verification later on
	//MPI_Allgatherv(MPI_IN_PLACE, 0/*ignored*/, MPI_DOUBLE, it_glob, mpi_zonesize, mpi_zonestart, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Startall(2*count_mpix, x_req);
	MPI_Waitall(2*count_mpix, x_req, MPI_STATUSES_IGNORE);

	for(r=0; r<2*count_mpix; r++)
	{
		MPI_Request_free(x_req+r);
		MPI_Request_free(p1_req+r);
		MPI_Request_free(p2_req+r);
	}

	free(p_glob);
	free(old_p_glob);
	free(Ap);
	free(gradient);
	free(Aiterate);
	free(wait_for_p);
	free(wait_for_mvm);
	free(start_rt_work);
	free(wait_for_iterate);

	#if CKPT == CKPT_IN_MEMORY
	free(save_it);
	free(save_g);
	free(save_p);
	free(save_Ap);
	#endif
}

