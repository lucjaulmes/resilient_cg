#include <math.h>
#include <stdlib.h>

#include "global.h"

#include "dense_solvers.h"

void solve_qr(const DenseMatrix *A, const double* rhs, double *x)
{
	DenseMatrix tq, r;
	// A not necessarily square here, solving least squares problem.
	// Assume though that A->m <= A->n
	allocate_dense_matrix(A->n, A->n, &tq); // actually transposed of tq
	allocate_dense_matrix(A->m, A->n, &r);

	int i, j, k;

	for(i=0; i<A->n; i++)
		for(j=0; j<i && j<A->m; j++)
			r.v[i][j] = 0;

	for(j=0; j<A->m; j++)
	{
		// this is going to be stored in row j of tq
		// (so we are actually building the transposed of tq in the qr factorization)

		// get column j
		for(i=0; i<A->n; i++)
			tq.v[j][i] = A->v[i][j];

		// MGS to orthogonalize
		for(k=0; k<j; k++)
		{
			r.v[k][j] = scalar_product(A->n, tq.v[j], tq.v[k]);

			for(i=0; i<A->m; i++)
				tq.v[j][i] -= r.v[k][j] * tq.v[k][i];
		}

		// get norm ... to normaljje
		r.v[j][j] = sqrt( scalar_product(A->n, tq.v[j], tq.v[j]) );

		for(i=0; i<A->m; i++)
			tq.v[j][i] /= r.v[j][j];
	}

	// now that we have factorized, compute the solution
	// with y = t(tq) * rhs, we have left to solve R * x = y
	double y[A->n];
	mult_dense((void*)&tq, rhs, y);

	for(i=A->n-1; i>=0; i--)
	{
		x[i] = y[i];

		for(j=i+1; j<A->m; j++)
			x[i] -= r.v[i][j] * x[j];

		x[i] /= r.v[i][i];
	}

	deallocate_dense_matrix(&tq);
	deallocate_dense_matrix(&r);
}

// utility function for the householder qr solver
void matrix_minor(const int d, const DenseMatrix *A, DenseMatrix *B)
{
	int i, j;

	for( i = 0; i < A->n; i++)
		for( j = 0; j < A->m; j++)
			B->v[i][j] = 0;

	for( i = 0; i < d; i++)
		B->v[i][i] = 1;

	for( i = d; i < A->n; i++)
		for( j = d; j < A->m; j++)
			B->v[i][j] = A->v[i][j];
}

/* m = I - v v^T */
void householder(DenseMatrix *h, double *v)
{
	int i, j;

	for( i = 0; i < h->n; i++)
		for( j = 0; j < h->n; j++)
			h->v[i][j] = -2 *  v[i] * v[j];

	for( i = 0; i < h->n; i++)
		h->v[i][i] += 1;
}

void get_tq_householder(const DenseMatrix *A, DenseMatrix *transQ)
{
	// we are going to build the transposed of the Q matrix in the Q-R factorization of A
	// alternatively in tq and tq2 (because we can't multiply matrices in-place).
	int k, i;

	DenseMatrix z, z1, h, *tq, *tq2, *swap;
	allocate_dense_matrix(A->n, A->m, &z);
	allocate_dense_matrix(A->n, A->m, &z1);
	allocate_dense_matrix(A->n, A->n, &h);
	swap = malloc(sizeof(DenseMatrix));
	allocate_dense_matrix(A->n, A->n, swap);

	// Here we set the pointers so that the last iteration points to transQ.
	if( (A->n-1 <= A->m && A->n-1 % 2 == 1) || (A->m < A->n-1 && A->m % 2 == 1) )
	{
		tq = swap;
		tq2 = transQ;
	}
	else
	{
		tq2 = swap;
		tq = transQ;
	}

	// start with tq as identity
	for( i = 0; i < A->n; i++)
		tq->v[i][i] = 1;

	for( k = 0; k < A->n-1 && k < A->m ; k++)
	{
		double x[A->n], a;

		// in-place matrix minor (if not A)
		if( k == 0 )
			matrix_minor(k, A, &z);
		else
			matrix_minor(k, &z1, &z);

		// x <- z_,k + a * e_k
		for(i=0; i<A->n; i++)
			x[i] = z.v[i][k]; // TODO bad : get column should be black-boxed so we don't know the structure of matrix

		// a is +/- || z_,k ||
		a = sqrt( scalar_product(A->n, x, x) );

		if (A->v[k][k] > 0) // TODO bad : get column should be black-boxed so we don't know the structure of matrix
			a = -a;

		x[k] += a;

		// normalize x
		a = sqrt(scalar_product(A->n, x, x));
		for( i = 0; i < A->n; i++)
			x[i] /= a;

		// get h the householder transformation
		// m = I - e e^T
		householder(&h, x);

		mult_dense_matrix(&h, tq, tq2);
		swap = tq;
		tq = tq2;
		tq2 = swap;

		// in-place matrix multiplication (if not A)
		mult_dense_matrix(&h, &z, &z1);
	}

	deallocate_dense_matrix(&z);
	deallocate_dense_matrix(&z1);
	deallocate_dense_matrix(&h);

	if( tq == transQ )
	{
		deallocate_dense_matrix(tq2);
		free(tq2);
	}
	else
	{
		deallocate_dense_matrix(tq);
		free(tq);
	}
}


void solve_qr_house(const DenseMatrix *A, const double* rhs, double *x)
{
	// A->n should be size of rhs > A->m, also size of x
	DenseMatrix tQ, R;
	allocate_dense_matrix(A->n, A->n, &tQ);
	allocate_dense_matrix(A->n, A->m, &R);

	get_tq_householder(A, &tQ);

	double y[A->n];
	mult_dense(&tQ, rhs, y);

	mult_dense_matrix(&tQ, A, &R);

	int i, j;
	for(i=A->m-1; i>=0; i--)
	{
		x[i] = y[i];

		for(j=i+1; j<A->m; j++)
			x[i] -= R.v[i][j] * x[j];

		x[i] /= R.v[i][i];
	}

	deallocate_dense_matrix(&R);
	deallocate_dense_matrix(&tQ);
}



// using Crout's algorithm
void solve_lu(const DenseMatrix *A, const double* rhs, double *x)
{
	DenseMatrix lu;
	// Assume A->n == A->m
	allocate_dense_matrix(A->n, A->n, &lu);

	int i, j, k, p;

	for(k=0; k<A->n; ++k)
	{
		for(i=k; i<A->n; ++i)
		{
			double sum=0;

			for(p=0; p<k; ++p)
				sum += lu.v[i][p] * lu.v[p][k];

			lu.v[i][k] = A->v[i][k] - sum; // not dividing by diagonals
		}

		for(j=k+1; j<A->n; ++j)
		{
			double sum=0;

			for(p=0; p<k; ++p)
				sum += lu.v[k][p] * lu.v[p][j];

			lu.v[k][j] = (A->v[k][j] - sum) / lu.v[k][k];
		}
	}

	double y[A->n];
	for(i=0; i<A->n; ++i)
	{
		double sum = 0;

		for(k=0; k<i; ++k)
			sum += lu.v[i][k] * y[k];

		y[i] = (rhs[i] - sum) / lu.v[i][i];
	}

	for(i=A->n-1; i>=0; --i)
	{
		double sum=0;

		for(k=i+1; k<A->n; ++k)
			sum += lu.v[i][k] * x[k];

		x[i] = (y[i] - sum); // not dividing by diagonals
	}

	deallocate_dense_matrix(&lu);
}


void solve_cholesky(const DenseMatrix *A, const double* rhs, double *x)
{
	DenseMatrix lu;
	// Assume A spd (so also square A->n == A->m)
	allocate_dense_matrix(A->n, A->n, &lu);

	int i, k, p;


	// Cholesky requires the matrix to be symmetric positive-definite
	for(k=0; k<A->n; ++k)
	{
		double sum = 0.;

		for(p=0; p<k; ++p)
			sum += lu.v[k][p] * lu.v[k][p];

		lu.v[k][k] = sqrt(A->v[k][k] - sum);

		for(i=k+1; i<A->n; ++i)
		{
			double sum = 0.;

			for(p=0; p<k; ++p)
				sum += lu.v[i][p] * lu.v[k][p];

			lu.v[i][k] = (A->v[i][k] - sum) / lu.v[k][k];
		}
	}

	double y[A->n];

	for(i=0; i<A->n; ++i)
	{
		double sum = 0.;

		for(k=0; k<i; ++k)
			sum += lu.v[i][k] * y[k];

		y[i] = (rhs[i]-sum) / lu.v[i][i];
	}

	for(i=A->n-1;i>=0;--i)
	{
		double sum = 0.;

		for(k=i+1; k<A->n; ++k)
			sum += lu.v[k][i] * x[k];

		x[i] = (y[i] - sum) / lu.v[i][i];
	}

	deallocate_dense_matrix(&lu);
}

