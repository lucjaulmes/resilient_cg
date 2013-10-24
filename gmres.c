#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "global.h"
#include "failinfo.h"
#include "recover.h"
#include "solvers.h"
#include "debug.h"

#include "gmres.h"

void solve_gmres( const int n, const void *A, const double *b, double *x, double thres, const int restart )
{
	double error, norm_b, time, comp_thres;
	int it, total_it = 0, steps, failures = 0;

	// not-restarted strategy : maximum n steps (NB. the algorithm can still restart before that sometimes, e.g. degenerate cases)
	if( restart < 1 || restart > n )
		steps = n;
	else
		steps = restart;

	norm_b = sqrt(scalar_product(n, b, b));
	comp_thres = thres * norm_b;
	log_out("Error shown is ||Ax-b||, you should plot ||Ax-b||/||b||. (||b|| = %e)\n", norm_b);

	start_measure();

	do{
		restart_gmres(n, A, b, x, comp_thres, steps, &error, &it);

		if( get_nb_failed_blocks() > 0 )
		{
			failures += get_nb_failed_blocks();

			// recover with least squares.
			// NB : Actually we could interpolate when the submatrix is full rank, but how to know that efficiently ?
			recover_leastsquares( A, b, x , fault_strat );
		}
		else if( error >= comp_thres || error <= -comp_thres )
			log_out("Restart.\n");

		total_it += it;
	}
	while( error < -comp_thres || error > comp_thres );

	time = stop_measure();
	printf("\nGMRES method finished in wall clock time %e usecs with %d failures (%d iterations, error %e)\n", time, failures, total_it, error / norm_b);
}

void restart_gmres( const int n, const void *A, const double *b, double *x, double thres, const int max_steps, double *error, int *rank_converged )
{
	int i, j, r;
	*rank_converged = -1;

	double
	    // for Arnoldi iterations.
	    // !! since we use the column-vectors of each matrix, they are stocked in column-major
	    **q, **h,
	    // for the QR decomposition of h
	    **o, **p,
	    // contains the rhs of the "innermost" problem (thus also the error)
	    g[max_steps];

	*error = DBL_MAX;

	DenseMatrix mat_o, mat_q, mat_p, mat_h;

	allocate_dense_matrix(max_steps, n, &mat_q);
	allocate_dense_matrix(max_steps-1, max_steps, &mat_h);

	allocate_dense_matrix(max_steps, max_steps, &mat_o);
	allocate_dense_matrix(max_steps-1, max_steps, &mat_p);

	q = mat_q.v;
	h = mat_h.v;
	o = mat_o.v;
	p = mat_p.v;

	// O initially identity, vector g zero
	for(i=0; i<max_steps; i++)
	{
		o[i][i] = 1;
	    g[i] = 0;
	}

	// instead of setting b as the initial step of the algorithm
	// set b - A * x where x is the iterate
	// this is the same with initial guess x = 0 but changes when 
	// x contains a better guess

	double norm0;

	{
		mult((void*)A, x, q[0]);

		// q_0 <- b - q_0 = b - A * it
		daxpy(n, -1.0, q[0], b, q[0]);

		norm0 = sqrt( scalar_product(n+1, q[0], q[0]) );

		for(i=0; i<n; i++)
			q[0][i] /= norm0;

		// q_0 decomposition on dimension 1
		g[0] = norm0;
	}

	for(r=1; *rank_converged < 0 && r < max_steps; r++)
	{
		start_iteration();
		
	    // build next vector of q as Arnoldi iteration
	    // expand Krylov subspace through multiplying by A
	    mult((void*)A, q[r-1], q[r]);

	    // get values of h, which will be used to orthonormalize q_r
		mgs(n, r, q[r], h[r-1], (const double**)q);

	    // degenerate case ! better to use a threshold, e.g. h[r-1][r] / norm < e-14 ?
		// the right thing to do here is to compute the iterate x
		// and restart the method with this iterate as initial guess
		{
			if( h[r-1][r] == 0 )
			{
				log_out("\n\n------\nDegenerate case at rank %d (with error %e) : solve-restart needed\n------\n\n", r, *error);
				#pragma omp critical
					*rank_converged = r+1;
			}
			else
				// normalize
				for(j=0; j<n; j++)
					q[r][j] /= h[r-1][r];
		}

	    // update the QR decomposition of H with a Givens rotation
	    double cos, sin, norm;

		// p's new last column is h's last column (left-)multiplied by o, with a 1 on (r,r)
		mult_dense(&mat_o, h[r-1], p[r-1]); // rank-limited multiplication ? e.g. set mat_o.n = mat_o.m = r ?

	    // define the givens rotation to get the neat triangular form on p
		{
			cos = p[r-1][r-1];
			sin = p[r-1][r];
			norm = sqrt( cos * cos + sin * sin );

			cos /= norm;
			sin /= norm;

			// update the new last column of p, after givens rotation
			p[r-1][r-1] = norm;
			p[r-1][r] = 0;
		}

	    // apply givens rotation to o
	    // (or alternatively save the values {r,cos,sin} for later)
		givens_rotate(r+1, o[r-1], o[r], cos, sin);

		{
			int failures = 0;

			// rotate the vector g also
			givens_rotate(1, &g[r-1], &g[r], cos, sin);
			*error = g[r];

			stop_iteration();
			failures = get_nb_failed_blocks();

			char stop_here = (*error <= thres && *error >= -thres) || (failures > 0);

			#pragma omp critical
			{
				if( stop_here && *rank_converged < 0 )
					*rank_converged = r;
			}

			log_out("% e %d\n", *error, failures);
			if(*rank_converged == r )
				log_out("\n\n------\nConverged at rank %d (with error %e)\n------\n\n", r, *error);
		}

		if( r == max_steps )
		{
			log_out("\n\n------\nMaximum number of steps %d reached in restarted gmres (with error %e) :"
					"solve-restart programmed\n------\n\n", r, *error);
			#pragma omp critical
				*rank_converged = r;
		}

	}


	double y[max_steps];
	int s = 0;
	{
		// check where we can start (should be r-1 except on degenerate cases r-2)
		s = *rank_converged;
		while( s >= 0 && (s >= max_steps-1 || p[s][s] == 0) )
			s--;

		// really this hould never happen
		if( s < 0 )
		{
			log_out("BAD STUFF : s = %d, r = %d, rank_converged = %d, max_steps-1 = %d\n", s, r, *rank_converged, max_steps-1);
			exit(1);
			return;
		}

		for(i=s+1; i<max_steps; i++)
			y[i] = 0;

		// x = q * y = q * (p^-1 * g)
		for(i=s; i>=0; i--)
		{
			y[i] = g[i];

			for(j=i+1; j<s; j++)
				y[i] -= p[j][i] * y[j];

			y[i] /= p[i][i];
		}
	}

	// get q in row-major and multiply into z, then add to x
	double z[n];
	mult_dense_transposed(&mat_q, y, z);

	daxpy(n, 1.0, z, x, x);
	
	#if defined VERBOSE && VERBOSE > FULL_VERBOSE
		// Everything we could possibly want to debug. Remember that we use row-major representation
		// so for ease of use let's transpose all the matrices.
		DenseMatrix test, H, P, O;
		allocate_dense_matrix(max_steps+1, max_steps+1, &test);
		allocate_dense_matrix(mat_o.m, mat_o.n, &O);
		allocate_dense_matrix(mat_p.m, mat_p.n, &P);
		allocate_dense_matrix(mat_h.m, mat_h.n, &H);
		transpose_dense_matrix(&mat_h, &H);
		transpose_dense_matrix(&mat_p, &P);
		transpose_dense_matrix(&mat_o, &O);

		log_err("Some Verifications. P is :\n");
		print_dense(&P);

		log_err("H is :\n");
		print_dense(&H);

		log_err("O * P is :\n");
		test.m = P.m;
		test.n = O.n;
		mult_dense_matrix(&O, &P, &test);
		print_dense(&test);

		log_err("t(O) * O is :\n");
		test.m = O.m;
		test.n = O.n;
		mult_dense_matrix(&mat_o, &O, &test);

		for(i=0; i<max_steps+1; i++)
			for(j=0; j<max_steps+1; j++)
				if( test.v[i][j] < 1e-15 )
					test.v[i][j] = 0;
		print_dense(&test);

		double verif[max_steps], verif_err = 0;
		if(s > *rank_converged)
			s = *rank_converged;
		mult_dense(&P, y, verif);
		for(i=0; i<s; i++)
			verif_err += (g[i] - verif[i]) * (g[i] - verif[i]);

		log_err("After end of the GMRES(m), real error is %e :\ng =", sqrt(verif_err));
		for(i=0; i<s; i++)
			log_err(" % 1.2e", g[i]);
		log_err("\nv =");
		for(i=0; i<s; i++)
			log_err(" % 1.2e", verif[i]);
		log_err("\n");

		double norm_b = sqrt(scalar_product(n, b, b));
		mult_dense(&H, y, verif);
		verif_err = (verif[0] - norm_b) * (verif[0] - norm_b);
		for(i=1; i<s+1; i++)
			verif_err += verif[i] * verif[i];

		log_err("After end of the GMRES(m), real error is %e :\ng =", sqrt(verif_err));
		for(i=0; i<s+1; i++)
			log_err(" % 1.2e", i==0 ? norm_b : 0.0);
		log_err("\nv =");
		for(i=0; i<s+1; i++)
			log_err(" % 1.2e", verif[i]);
		log_err("\n");

		deallocate_dense_matrix(&test);
		deallocate_dense_matrix(&O);
		deallocate_dense_matrix(&P);
		deallocate_dense_matrix(&H);
		
	#endif


	deallocate_dense_matrix(&mat_q);
	deallocate_dense_matrix(&mat_h);
	deallocate_dense_matrix(&mat_o);
	deallocate_dense_matrix(&mat_p);

	log_out("stop inner loop after %d steps, error = %e whereas threshold = %e\n", r, *error, thres);
}

// takes two rows of length n, and returns them with {r1,r2} = {cos*r1+sin*r2n, -sin*r1+cos*r2}
//#pragma omp task in(cos,sin) inout([n]r1, [n]r2)
void givens_rotate( const int n, double *r1, double *r2, const double cos, const double sin )
{
	int i;
	double r1_i;

	for(i=0; i<n; i++)
	{
	    r1_i = cos * r1[i] + sin * r2[i];
	    r2[i] = - sin * r1[i] + cos * r2[i];
	    r1[i] = r1_i;
	}
}

//#pragma omp task inout([n+1]q_r) out([n+1]h)
void mgs(const int n, const int r, double *q_r, double *h, const double **q)
{
	int i, j;

	for(i=0; i<r; i++)
	{
		h[i] = scalar_product(n+1, q[i], q_r);
		// update q_r as we go, which is mathematically identical
		// to removing all the h_i*q_i at the end since all the q_i
		// are orthogonal, but yields smaller rounding errors
		// (modified gram-schmidt)
		// Better lagorithm for block-parallelization of orthogonalizing a vector ? Householder ? Givens ?
		for(j=0; j<n; j++)
			q_r[j] -= h[i] * q[i][j];
	}

	// norm of the orthogonal vector in h_r,r-1
	h[r] = sqrt( scalar_product(n+1, q_r, q_r) );

}


