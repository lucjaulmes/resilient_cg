#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"

#include "cg.h"

void solve_cg( const int n, const void *A, const double *b, double *iterate, double thres )
{
	double error, norm_b, time, comp_thres;
	int it, total_it = 0, failures = 0;

	norm_b = scalar_product(n, b, b);
	comp_thres = thres * thres * norm_b;
	log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);
	norm_b = sqrt(norm_b);

	start_measure();

	do{
		restart_cg(n, A, b, iterate, comp_thres, &error, &it);

		if( get_nb_failed_blocks() > 0 )
		{
			failures += get_nb_failed_blocks();

			// recover by interpolation since our submatrix is always spd
			recover_interpolation( A, b, iterate, &solve_cholesky, fault_strat );
		}
		else if( error > comp_thres )
			log_out("Restart.\n");

		total_it += it;
	}
	while( error > comp_thres );

	time = stop_measure();
	printf("\nCG method finished in wall clock time %e usecs with %d failures (%d iterations, error %e)\n", time, failures, total_it, sqrt(error)/norm_b);
}

void restart_cg( const int n, const void *A, const double *b, double *iterate, double thres_sq, double *error, int *rank_converged )
{
	int r, failures;
	*rank_converged = -1;
	double *p, *Ap, normA_p_sq, *gradient, err_sq, old_err_sq, beta = 0.0;

	p = (double*)calloc( n, sizeof(double) );
	Ap = (double*)malloc( n * sizeof(double) );
	gradient = (double*)malloc( n * sizeof(double) );

	double *p2 = (double*)calloc( n, sizeof(double) );

	// initialize first direction and iterate
	// direction vector is b when iterate is 0, since gradient is b - A * it
	mult(A, iterate, gradient);

	// gradient <- -1 * gradient + b 
	daxpy(n, -1.0, gradient, b, gradient);

	err_sq = scalar_product(n, gradient, gradient);

	for(r=0; r < 500*n ; r++)
	{
		start_iteration();

		// we've got the gradient to get next direction (= error vector)
		// make it orthogonal to the last direction (it already is to all the previous ones)

		// Initially beta is 0 and p unimportant : p <- gradient
		if( r > 0 )
			beta = err_sq / old_err_sq;

		daxpy(n, beta, p, gradient, p2);

		int i;
		double *swap = p2, norm_p_p2 = sqrt( scalar_product(n, p, p) * scalar_product(n, p2, p2) );
		if( r > 0 )
		{
			log_err(FULL_VERBOSE, "\nbeta = %e => cos(p_%d,p_%d) = %e\n", beta, r, r-1, scalar_product(n, p, p2) / norm_p_p2);

			log_err(FULL_VERBOSE, "\np_%d\t\tp_%d\n", r-1, r);
			for(i=0; i<n; i++)
				log_err(FULL_VERBOSE, "%1.2e\t%1.2e\n", p[i], p2[i]);
		}
		p2 = p;
		p = swap;


		double alpha;

		// store A*p_r
		mult(A, p, Ap);

		// get the norm for A of this new direction vector
		normA_p_sq = scalar_product(n, p, Ap);

		alpha = err_sq / normA_p_sq ;

		log_err(FULL_VERBOSE, "||p_%d||_A = %e ; alpha = %e \n", r, normA_p_sq, alpha );

		// update iterate with contribution along new direction
		daxpy(n, alpha, p, iterate, iterate);

		// update gradient to solution (a.k.a. error) : b - A * it
		// every now and then, recompute properly to remove rounding errors
		if( (r+1) % 50)
			daxpy(n, -alpha, Ap, gradient, gradient);
		else
		{
			mult(A, iterate, gradient);
			daxpy(n, -1.0, gradient, b, gradient);
		}

		old_err_sq = err_sq;
		err_sq = scalar_product(n, gradient, gradient);

		// this to break when we have errors
		{
			stop_iteration();
			failures = get_nb_failed_blocks();

			log_out("% e %d\n", err_sq, failures);

			if ((err_sq <= thres_sq) || (failures > 0))
				break;
		}
	}

	log_out("\n\n------\nConverged at rank %d\n------\n\n", r);

	*error = err_sq;
	*rank_converged = r;

	free(p);
	free(p2);
	free(Ap);
	free(gradient);
}

