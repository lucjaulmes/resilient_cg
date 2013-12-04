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
	// do some memory allocations
	double norm_b, time, thres_sq;
	int r, failures, total_failures = 0, update_gradient = 0;
	double *p, *Ap, normA_p_sq, *gradient, err_sq, old_err_sq = INFINITY, alpha, beta = 0.0;

	p = (double*)calloc( n, sizeof(double) );
	Ap = (double*)malloc( n * sizeof(double) );
	gradient = (double*)malloc( n * sizeof(double) );

	// some parameters pre-computed, and show some informations
	norm_b = scalar_product(n, b, b);
	thres_sq = thres * thres * norm_b;
	log_out("Error shown is ||Ax-b||^2, you should plot ||Ax-b||/||b||. (||b||^2 = %e)\n", norm_b);
	norm_b = sqrt(norm_b);


	// real work starts here
	start_measure();

	for(r=0; r < 500*n ; r++)
	{
		start_iteration();

		// update gradient to solution (a.k.a. error) : b - A * it
		// every now and then, recompute properly to remove rounding errors
		// NB. do this on first iteration where alpha Ap and gradient aren't defined 
		if( update_gradient-- )
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
		failures = get_nb_failed_blocks();

		log_out("% e %d\n", err_sq, failures);

		if ( failures > 0 )
		{
			total_failures += failures;
			// do the recovery here
			// recover by interpolation since our submatrix is always spd
			recover_interpolation( A, b, iterate, fault_strat );

			old_err_sq = INFINITY;
			update_gradient = 0;

			#if VERBOSE > SHOW_FAILINFO
			mult(A, iterate, gradient);
			daxpy(n, -1, gradient, b, gradient);
			
			log_err(SHOW_FAILINFO, "Recovered : moved from % 1.2e to % 1.2e\n", err_sq, scalar_product(n, gradient, gradient));
			#endif
		}
		if (err_sq <= thres_sq)
			break;
	}

	// end of the math, showing infos
	time = stop_measure();

	log_out("\n\n------\nConverged at rank %d\n------\n\n", r);
	printf("\nCG method finished in wall clock time %e usecs with %d failures (%d iterations, error %e)\n", time, total_failures, r, sqrt(err_sq)/norm_b);

	free(p);
	free(Ap);
	free(gradient);
}

