#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "failinfo.h"
#include "recover.h"

#include "cg.h"

static double norm_b;

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
	double err = 0;
	int i, failures = 0;

	for(i=0; i<n; i++)
		iterate[i] = 0;

	norm_b = sqrt(scalar_product(n, b, b));

	do{
		restart_cg(n, A, b, iterate, thres, &err);

		if( get_nb_failed_blocks() > 0 )
		{
			{
				// do some checking on the recovery
				double y[n], new_err;
				mult(A, iterate, y);
				new_err = 0;
				for(i=0; i<n; i++)
					new_err += (b[i] - y[i]) * (b[i] - y[i]);

				fprintf(stderr, "Before recovery error is %e\n", sqrt( new_err ) / norm_b );
			}

			failures += get_nb_failed_blocks();

			// recover by interpolation since our submatrix is always spd
			recover_interpolation( A, b, iterate, &solve_cholesky, fault_strat );

			{
				// do some checking on the recovery
				double y[n], new_err;
				mult(A, iterate, y);
				new_err = 0;
				for(i=0; i<n; i++)
					new_err += (b[i] - y[i]) * (b[i] - y[i]);

				fprintf(stderr, "Recovery changed error to %e\n", sqrt( new_err ) / norm_b );
			}
		}
		else if( err >= -thres && err <= thres )
			printf("Converged\n%d failures in this run.\n\n", failures);
		else
			printf("Restart.\n");

	} while( err > thres );
}

void restart_cg( const int n, const void *A, const double *b, double *iterate, double thres, double *err )
{
    int r, i;
    double p[n], Ap[n], normA_p_sq, gradient[n], thres_sq = thres * thres, err_sq, old_err_sq = DBL_MAX;

	// initialize first direction and iterate
	// direction vector is b when iterate is 0, since gradient is b - A * it
	mult(A, iterate, gradient);

	for(i=0; i<n; i++)
		gradient[i] = b[i] - gradient[i];

	err_sq = scalar_product(n, gradient, gradient);

    for(r=0; err_sq > thres_sq && r < 10*n ; r++)
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
		*err = sqrt(err_sq);

		//printf("\nAt step %d, Euclidian norm of the error : %e\n", r, *err);
		
		stop_iteration();
		int failures = get_nb_failed_blocks();

		printf("%e %d\n", (*err)/norm_b, failures);

		if( failures )
			break;
    }
}



double estimate_cg_condition_number( const DenseMatrix *A )
{
	// this gets us a norm on A
	double max_sum_row = 0, sum_row, max_in_row, smallest_max_in_row = DBL_MAX;
	int i, j;

	for (i = 0; i < A->n; i++)
	{
		sum_row = 0;
		max_in_row = 0;
		for (j = 0; j < A->m; j++)
		{
			double A_ij = A->v[i][j];
			if( A_ij < 0 )
				A_ij = -A_ij;

			if( A_ij == 0 )
				continue;

			sum_row += A_ij;

			if( A_ij > max_in_row )
				max_in_row = A_ij;
		}

		if( sum_row > max_sum_row )
			max_sum_row = sum_row;

		// or we could decide to keep the number k of this row
		// and then solve the problem for A x = e_k, then we would have 
		// A^-1 's colum, number k in x
		if( smallest_max_in_row > max_in_row && max_in_row > 0 )
			smallest_max_in_row = max_in_row ;
	}
	
	double cond_num = max_sum_row / smallest_max_in_row;

	printf("||A|| = %e and ||A^-1|| >= %e\n", max_sum_row, 1/smallest_max_in_row);

	if(cond_num > 100)
		printf("Matrix is ILL-CONDITIONED\n");
	if(cond_num < 10)
		printf("Matrix is well conditioned\n");
	

	printf("The condition number is >= %e\n\n", cond_num);

	return cond_num;
}

