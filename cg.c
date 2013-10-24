#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"

#include "cg.h"

// debug methods, quantifying conjugacy or orthogonality between vectors
double ort(const int n, const double *v, const double *w)
{
	double sp = scalar_product(n, v, w),
		norm2_v = scalar_product(n, v, v),
		norm2_w = scalar_product(n, w, w);

	return sp / sqrt( norm2_v * norm2_w );
}

double conjug(const int n, const void *A, const double *v, const double *w)
{
	double Aw[n], Av[n];
	mult(A, v, Av);
	mult(A, w, Aw);
	double sp = scalar_product(n, v, Aw),
		norm2_v = scalar_product(n, v, Av),
		norm2_w = scalar_product(n, w, Aw);

	return sp / sqrt( norm2_v * norm2_w );
}

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
	int r;
	*rank_converged = -1;
	double p[n], Ap[n], normA_p_sq, gradient[n], err_sq, old_err_sq, coeff = 0.0;

	// initialize first direction and iterate
	// direction vector is b when iterate is 0, since gradient is b - A * it
	{
		mult(A, iterate, gradient);

		// gradient <- -1 * gradient + b 
		daxpy(n, -1.0, gradient, b, gradient);
	}

	err_sq = scalar_product(n, gradient, gradient);

	for(r=0; *rank_converged < 0 && r < 500*n ; r++)
	{
		start_iteration();

		// we've got the gradient to get next direction (= error vector)
		// make it orthogonal to the last direction (it already is to all the previous ones)

		// Initially coeff is 0 and p unimportant : p <- gradient
		if( r > 0 )
			coeff = err_sq / old_err_sq;

		daxpy(n, coeff, p, gradient, p);


		double alpha;

	    // store A*p_r
	    mult((void*)A, p, Ap);

	    // get the norm for A of this new direction vector
	    normA_p_sq = scalar_product(n, p, Ap);

		{
			alpha = err_sq / normA_p_sq ;
			old_err_sq = err_sq;
		}

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

		err_sq = scalar_product(n, gradient, gradient);

		// finally, compute (squared) error
		{
			int failures = 0;

			stop_iteration();
			failures = get_nb_failed_blocks();

			char stop_here = (err_sq <= thres_sq) || (failures > 0);

			#pragma omp critical
			{
				if( stop_here && *rank_converged < 0 )
					*rank_converged = r;
			}

			log_out("% e %d\n", err_sq, failures);
			if( *rank_converged == r )
				log_out("\n\n------\nConverged at rank %d\n------\n\n", r);
		}
	}

	*error = err_sq;
}

