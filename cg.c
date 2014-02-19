#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"
#include "counters.h"

#include "cg.h"


// defining recovery strats
#define SEARCH_V_0	1
#define SEARCH_V_INF	2
#define SEARCH_ORTG_0	3
#define SEARCH_ORTG_INF	4
#define SEARCH_VxORTG_0	5
#define SEARCH_VxORTG_INF	6
#define SEARCH_ORT_GV_0	7
#define SEARCH_ORT_GV_INF	8
#define RECUP_X	9
#define RECUP_X_NO_RECOMPUTE_G	11
#define RECUP_X_NO_P	12
#define RECUP_X_NO_RECOMPUTE_G_NO_P	13
#define ORTH_G_BY_Q	14


void solve_cg( const int n, const void *A, const double *b, double *iterate, double thres )
{
	// do some memory allocations
	double norm_b, thres_sq;
	int r, failures, total_failures = 0, update_gradient = 0;
	double *p, *Ap, normA_p_sq, *gradient, err_sq, old_err_sq = INFINITY, alpha, beta = 0.0;

	const double *V = b;

	p = (double*)calloc( n, sizeof(double) );
	Ap = (double*)malloc( n * sizeof(double) );
	gradient = (double*)malloc( n * sizeof(double) );

	// some parameters pre-computed, and show some informations
	norm_b = scalar_product(n, b, b);
	thres_sq = thres * thres * norm_b;
	log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);

	int recovery_iteration = 0;

	// real work starts here
	start_measure();
	for(r=0; r < 100*n ; r++)
	{
		start_iteration();
		if( recovery_iteration )
		{
			int start = MAGIC_BLOCKTORECOVER, end = start + get_failblock_size(), i;
			if ( end > n )
				end = n;

			int block[] = {MAGIC_BLOCKTORECOVER};

			//do_interpolation(A, b, iterate, 1, block);
			//do_interpolation_with_grad(A, b, g, iterate, 1, block);
			//do_interpolation_q(A, gradient, q, 1, block);
			//do_interpolation_x_and_q(A, b, gradient, iterate, q, 1, block);

			if( recovery_iteration == RECUP_X || recovery_iteration == RECUP_X_NO_RECOMPUTE_G || recovery_iteration == RECUP_X_NO_P || recovery_iteration == RECUP_X_NO_RECOMPUTE_G_NO_P )
			{
				daxpy(n, -alpha, Ap, gradient, gradient);

				err_sq = scalar_product(n, gradient, gradient);
				beta = err_sq / old_err_sq;

				// variants where p is lost
				if( recovery_iteration == RECUP_X_NO_P || recovery_iteration == RECUP_X_NO_RECOMPUTE_G_NO_P )
					beta = 0.0;

				// ~ add some noise to x
				do_interpolation_with_grad(A, b, gradient, iterate, 1, block);

				// THE NORMAL ITERATION FOLLOWS :
				daxpy(n, beta, p, gradient, p);

				// store A*p_r
				mult(A, p, Ap);

				// get the norm for A of this new direction vector
				normA_p_sq = scalar_product(n, p, Ap);

				alpha = err_sq / normA_p_sq ;

				// update iterate with contribution along new direction
				daxpy(n, alpha, p, iterate, iterate);

				old_err_sq = err_sq;

				if( recovery_iteration != RECUP_X_NO_RECOMPUTE_G || recovery_iteration != RECUP_X_NO_RECOMPUTE_G_NO_P )
					update_gradient = 0;

				log_out("%4d, % e %d\n", r, err_sq, 0 );
			}
			else if( recovery_iteration == ORTH_G_BY_Q )
			{
				double *q = (double*)calloc( n, sizeof(double) ), *Aq = (double*)calloc( n, sizeof(double) ), *AAq = (double*)calloc( n, sizeof(double) );
				daxpy(n, -alpha, Ap, gradient, gradient);
				//int block[] = {MAGIC_BLOCKTORECOVER};
				//do_interpolation_q(A, gradient, q, 1, block);

				for(i=0; i<n; i++)
					Aq[i] = gradient[i];
				
				// gradient <- b - A * x
				mult(A, iterate, gradient);
				daxpy(n, -1.0, gradient, b, gradient);

				for(i=0; i<n; i++)
					Aq[i] -= gradient[i];

				// now Aq = old_grad - new_grad
				mult(A, q, AAq);

				printf("pos  \tdiff_Aq\tdirect_Aq\tdiff\n");
				double err = 0.0;

				for(i=0; i<n; i++)
				{
					double e = (Aq[i] - AAq[i]) * (Aq[i] - AAq[i]);
					printf("%5d\t% e\t% e\t% e\t% e\n", i, q[i], Aq[i], AAq[i], sqrt(e));
					err += e;
				}
				printf("|| diff_Aq - direct_Aq || = %e", sqrt(err));

				free(q);
				free(Aq);
				free(AAq);
			}
			else if( recovery_iteration < SEARCH_ORT_GV_0 ) 
			{
				for(i=0;i<n;i++)
				{
					if( recovery_iteration == SEARCH_V_0 || recovery_iteration == SEARCH_V_INF )
						p[i] = V[i];
					else if( recovery_iteration == SEARCH_ORTG_0 || recovery_iteration == SEARCH_ORTG_INF )
						p[i] = ( i >= start && i < end ) ;
					else if( recovery_iteration == SEARCH_VxORTG_0 || recovery_iteration == SEARCH_VxORTG_INF )
						p[i] = ( i >= start && i < end ) * V[i];
					else
						; // ??
				}


				// store A*p_r
				mult(A, p, Ap);

				// get the norm for A of this new direction vector
				normA_p_sq = scalar_product(n, p, Ap);

				//alpha = err_sq / normA_p_sq;
				alpha = (scalar_product(n, b, p) - scalar_product(n, iterate, Ap)) / normA_p_sq;
				
				// update iterate with contribution along new direction
				daxpy(n, alpha, p, iterate, iterate);
			}

			// recompute g on next iteration
			update_gradient = 0;
			if( recovery_iteration == SEARCH_V_INF || recovery_iteration == SEARCH_ORTG_INF || recovery_iteration == SEARCH_VxORTG_INF || recovery_iteration == SEARCH_ORT_GV_INF ) 
				old_err_sq = INFINITY;
			else if( recovery_iteration == SEARCH_V_0 || recovery_iteration == SEARCH_ORTG_0 || recovery_iteration == SEARCH_VxORTG_0 || recovery_iteration == SEARCH_ORT_GV_0 )
				old_err_sq = 0.0; // will compute beta old-fashioned way using Ap and normA_p_sq, needs p to update new p
			else // RECUP_X
				old_err_sq = err_sq; // will compute beta using err_sq and old_err_sq, needs p to update new p
			recovery_iteration = 0;
		}
		else
		{
			// update gradient to solution (a.k.a. error) : b - A * it
			// every now and then, recompute properly to remove rounding errors
			// NB. do this on first iteration where alpha Ap and gradient aren't defined 
			if( update_gradient-- || r == MAGIC_ITERATION )
				daxpy(n, -alpha, Ap, gradient, gradient);
			else
			{
				mult(A, iterate, gradient);
				daxpy(n, -1.0, gradient, b, gradient);

				// do this computation, say, every 50 iterations
				update_gradient = 50;
			}

			err_sq = scalar_product(n, gradient, gradient);
			// we've got the gradient to get next direction (= error vector)
			// make it orthogonal to the last direction (it already is to all the previous ones)

			// Initially beta is 0 and p unimportant : p <- gradient (on (re)starts, old_err_sq = +infinity)
			if( old_err_sq == 0.0 )
				beta = scalar_product(n, gradient, Ap) / normA_p_sq;
			else
				beta = err_sq / old_err_sq;

			daxpy(n, beta, p, gradient, p);

			// store A*p_r
			mult(A, p, Ap);

			// get the norm for A of this new direction vector
			normA_p_sq = scalar_product(n, p, Ap);

			alpha = err_sq / normA_p_sq ;

			log_err(FULL_VERBOSE, "||p_%d||_A = %e ; alpha = %e \n", r, normA_p_sq, alpha );

			// update iterate with contribution along new direction
			daxpy(n, alpha, p, iterate, iterate);

			old_err_sq = err_sq;

			// this to break when we have errors
			stop_iteration();
			//if( r == MAGIC_ITERATION )
			//	report_failure( MAGIC_BLOCKTORECOVER );

			failures = get_nb_failed_blocks();

			log_out("%4d, % e %d\n", r, err_sq, failures );

			if ( failures > 0 )
			{
				total_failures += failures;

				// NOTA BENE--> while debugging, go at start of recovery iteration
				recovery_iteration = RECUP_X_NO_RECOMPUTE_G_NO_P;

				// do the recovery here 
				// recover by interpolation since our submatrix is always spd
				//recover_interpolation( A, b, iterate, fault_strat );

				//old_err_sq = INFINITY;
				//update_gradient = 0;

				#if VERBOSE > SHOW_FAILINFO
				mult(A, iterate, gradient);
				daxpy(n, -1.0, gradient, b, gradient);
				
				log_err(SHOW_FAILINFO, "Recovered : moved from % 1.2e to % 1.2e\n", err_sq, scalar_product(n, gradient, gradient));
				#endif
			}
			if (err_sq <= thres_sq)
				break;
		}
	}

	// end of the math, showing infos
	stop_measure();

	log_out("\n\n------\nConverged at rank %d\n------\n\n", r);
	printf("CG method finished iterations:%d with error:%e (failures:%d)\n", r, sqrt(err_sq/norm_b), total_failures);

	free(p);
	free(Ap);
	free(gradient);
}

