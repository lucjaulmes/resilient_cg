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
    int r, i;
    double p[n], Ap[n], normA_p_sq, gradient[n], err_sq, old_err_sq = DBL_MAX;

	// initialize first direction and iterate
	// direction vector is b when iterate is 0, since gradient is b - A * it
	mult(A, iterate, gradient);

	for(i=0; i<n; i++)
		gradient[i] = b[i] - gradient[i];

	err_sq = scalar_product(n, gradient, gradient);

    for(r=0; err_sq > thres_sq && r < 500*n ; r++)
	{
		start_iteration();

		// we've got the gradient to get next direction (= error vector)
		// make it orthogonal to the last direction (it already is to all the previous ones)

		if(r == 0)
			for(i=0; i<n; i++)
				p[i] = gradient[i];

		else
		{
			double coeff = err_sq / old_err_sq;

			for(i=0; i<n; i++)
				p[i] = gradient[i] + coeff * p[i];
		}


		double alpha;

        // store A*p_r
        mult((void*)A, p, Ap);

        // get the norm for A of this new direction vector
        normA_p_sq = scalar_product(n, p, Ap);

        alpha = err_sq / normA_p_sq ;

        // update iterate with contribution along new direction
        // update gradient to solution (a.k.a. error) : b - A * it
        for(i=0; i<n; i++)
            iterate[i] += alpha * p[i];

		// every now and then, recompute properly to remove rounding errors
		if( (r+1) % 50)
			for(i=0; i<n; i++)
				gradient[i] -= alpha * Ap[i];
		else
		{
			mult(A, iterate, gradient);
			for(i=0; i<n; i++)
				gradient[i] = b[i] - gradient[i];
		}

        // finally, compute (squared) error
		old_err_sq = err_sq;
        err_sq = scalar_product(n, gradient, gradient);

		stop_iteration();

		int failures = get_nb_failed_blocks();

		log_out("%      d %e %d\n", r, err_sq, failures);

		if( failures )
			break;
    }

	*error = err_sq;
	*rank_converged = r;
}

