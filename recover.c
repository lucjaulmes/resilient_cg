#include <math.h>
#include <stdlib.h>
#include <float.h>

#include "global.h"
#include "matrix.h"
#include "failinfo.h"
#include "debug.h"

#ifdef MATRIX_DENSE
#include "dense_solvers.h"
#else
#include "csparse.h"
#endif

#include "pcg.h"
#include "recover.h"

void prepare_x_decorrelated( double *x , const int n )
{
	int b, i, id, fbs = get_failblock_size();

	for(id=0; id < get_nb_failed_blocks(); id++)
	{
		b = get_failed_block(id);

		for(i=b*fbs; i<(b+1)*fbs && i < n; i++)
			x[ i ] = 0;
	}
}

void prepare_x_uncorrelated( double *x, const double *initial_x , const int n)
{
	int b, i, id, fbs = get_failblock_size();

	for(id=0; id < get_nb_failed_blocks(); id++)
	{
		b = get_failed_block(id);

		for(i=b*fbs; i<(b+1)*fbs && i < n; i++)
			x[ i ] = initial_x[ i ];
	}
}


void recover_xk( const Matrix *A, const double *b, const double *g, double *x, Precond *M, const int strategy )
{
	int block, id;

	if( strategy == NOFAULT )
		fprintf(stderr, "Warning : %d fault(s) with NOFAULT strategy, falling back to global if more than 1\n", get_nb_failed_blocks());

	if( get_nb_failed_blocks() == 1 )
	{
		block = get_failed_block(0);

		do_single_interpolation(A, b, g, x, block, M->S[block], M->N[block]);

		set_fixed(1);
	}


	switch( strategy )
	{
		case SINGLEFAULT :

			fprintf(stderr, "Warning : multiple faults with SINGLEFAULT strategy, falling back to uncorrelated\n");

		case MULTFAULTS_UNCORRELATED:
			prepare_x_uncorrelated( x, b, A->n );

			for(id=0; id < get_nb_failed_blocks(); id++)
			{
				block = get_failed_block(id);

				do_single_interpolation(A, b, g, x, block, M->S[block], M->N[block]);
				set_fixed(1);
			}
			break;

		case MULTFAULTS_DECORRELATED:
			prepare_x_decorrelated( x, A->n );

			for(id=0; id < get_nb_failed_blocks(); id++)
			{
				block = get_failed_block(id);

				do_single_interpolation(A, b, g, x, block, M->S[block], M->N[block]);
				set_fixed(1);
			}
			break;

		case NOFAULT :
		case MULTFAULTS_GLOBAL:
			// get list of failed blocks, group by neighbour clusters, and interpolate
			{
				int flb = get_nb_failed_blocks(), id = 0, i, j, lost[flb], m, set[flb];

				// fill lost with all the failed blocks
				for(id=0; id < flb; id++)
				{
					lost[id] = get_failed_block(id);

					// just to be safe, remove out of bounds items
					if(lost[id] >= get_nb_failblocks())
					{
						lost[id] = -1;
						set_fixed(1);
					}
				}

				for(id=0; id < flb; id++)
				{
					if (lost[id] < 0)
						continue;

					// get in set the block lost[id] and all its neighbours
					get_failed_neighbourset(lost[id], set, &m);
					
					printf("global strategy found that neighbourhood of failed blocks for block %d is of size %d\n", lost[id], m);
					
					// how are going to do this for several things at once ?
					if( m > 1 )
						do_multiple_interpolation(A, b, g, x, m, set, M->S, M->N);
					else
						do_single_interpolation(A, b, g, x, lost[id], M->S[lost[id]], M->N[lost[id]]);

					set_fixed(m);

					// remove from lost all blocks that are in the set that we recover now
					// no need to recover them twice
					lost[id] = -1;
					for(i=id+1; i < flb; i++)
						for(j=0; j<m; j++)
							if( set[j] == lost[i] )
								lost[i] = -1;
				}
			}
			break;
	}
}

void do_interpolation( const double *rhs, double *x, const int total_lost, css *S, csn *N )
{
	double *y = malloc( total_lost * sizeof(double) );

	cs_ipvec (total_lost, S->Pinv, rhs, y) ;	/* y = P*rhs */
	cs_lsolve (N->L, y) ;		/* y = L\y */
	cs_ltsolve (N->L, y) ;		/* y = L'\y */
	cs_pvec (total_lost, S->Pinv, y, x) ;	/* x = P'*y */

	free(y);
}

void do_single_interpolation( const Matrix *A, const double *b, const double *g, double *x, const int lost_block, css *S, csn *N )
{
	int fbs = get_failblock_size(), total_lost = fbs, lost = lost_block * fbs;
	
	// change from block numver to first row in block number
	if( lost + fbs > A->n )
		total_lost -= (lost + fbs - A->n);

	double *rhs = (double*)calloc( total_lost, sizeof(double) );

	// fill in the rhs with the part we need 
	if( g == NULL )
		get_rhs(1, &lost, 1, &lost, total_lost, A, b, x, rhs);
	else
		get_rhs_sparse_with_grad(1, &lost, 1, &lost, total_lost, A, b, g, x, rhs);

	// from csparse
	do_interpolation(rhs, &x[lost], total_lost, S, N);
	free(rhs);
}

void do_multiple_interpolation( const Matrix *A, const double *b, const double *g, double *x, const int nb_lost, const int *lost_blocks, css **S, csn **N )
{
	Matrix submat;
	int i, j, lost[nb_lost], fbs = get_failblock_size(), total_lost = nb_lost * fbs, nnz = 0, last_fbs = fbs;
	// let us remap our factorized diagonal blocks
	css *subS[nb_lost];
	csn *subN[nb_lost];

	// lost contains starting row of each block
	for(i=0; i<nb_lost; i++)
	{
		lost[i] = lost_blocks[i] * fbs;

		subS[i] = S[ lost_blocks[i] ];
		subN[i] = N[ lost_blocks[i] ];

		int end_block = lost[i] + fbs;
		if( end_block > A->n )
		{
			last_fbs = A->n - lost[i];
			total_lost -= end_block - A->n;
			end_block = A->n;
		}
		nnz += A->r[end_block] - A->r[ lost[i] ];
	}

	// next call taskes care of allocating as well
	submatrix_sparse_nodiagblocks(A, lost, nb_lost, lost, nb_lost, fbs, &submat);

	double err = INFINITY,
		*rhs = (double*)calloc( total_lost, sizeof(double) ),
		*y1 = (double*)calloc( total_lost, sizeof(double) ),
		*y2 = (double*)calloc( total_lost, sizeof(double) ),
		*z = (double*)calloc( total_lost, sizeof(double) );

	// fill in the rhs with the part we need for our whole multiple problem
	if( g == NULL )
		get_rhs(nb_lost, lost, nb_lost, lost, fbs, A, b, x, rhs);
	else
		get_rhs_sparse_with_grad(nb_lost, lost, nb_lost, lost, fbs, A, b, g, x, rhs);
	
	double thres = DBL_EPSILON * total_lost;

	// we'll be using a blocked jacobi
	for(j=0; sqrt(err) > thres ; j++)
	{
		mult(&submat, y1, z);
		daxpy(total_lost, -1.0, z, rhs, z);
		// z ~= b - A * y1
		// without diagonal-block participations

		// now on each block i, we invert : y2_i = A_i^-1 * z_i = A_i^-1 * ( b - sum_{j != i} A_ij * y1_j )
		for(i=0; i<nb_lost; i++)
			do_interpolation( &z[i * fbs], &y2[i * fbs], i+1 == nb_lost ? last_fbs : fbs, subS[i], subN[i] );

		// check diff y1 - y2 to see if near fixed point
		err = 0.0;
		for(i=0; i<total_lost; i++)
			err += (y1[i] - y2[i])*(y1[i] - y2[i]);

		// finally swap y1 and y2 and continue
		double *swap = y2;
		y2 = y1;
		y1 = swap;
	}

	printf("blocked jacobi made %d iterations, error is %e < %e\n", j, sqrt(err), thres);

	for(i=0; i<nb_lost; i++)
		for(j=0; j < fbs && lost[i] + j < A->n; j++)
			x[ lost[i] + j ] = y1[ i * fbs + j];

	deallocate_matrix(&submat);
	free(rhs);
	free(z);
	free(y1);
	free(y2);
}

// give rows to but in rhs, and cols to avoid
void get_rhs_dense(const int n, const int *rows, const int m, const int *except_cols, const int bs, const DenseMatrix *A, const double *b, const double *x, double *rhs)
{
	int i, ii, j, jj, k;

	for(i=0, k=0; i<n; i++)
		for(ii=rows[i]; ii < rows[i] + bs && ii<A->n; ii++, k++)
		{
			// for each lost line i, start with b_i
			// and remove contributions A_ij * x_j 
			// from all rows j that are not lost
			rhs[k] = b[ ii ];

			log_err(SHOW_FAILINFO, "rhs_%d = b_%d - sum_j( dA_%d,j * x_j ), j =", k, ii, ii);

			for(j=0, jj=0; j<A->m; j++)
			{
				// update jj so that except_cols[jj] + bs > A->c[j]
				if( jj < m && except_cols[jj] + bs <= j)
					jj++;

				// if j is not in a columns-to-avoid set
				if( jj >= m || j < except_cols[jj] )
				{
					rhs[k] -= A->v[ ii ][j] * x[j];
					log_err(SHOW_FAILINFO, " %d", j);
				}
			}
			log_err(SHOW_FAILINFO, "\n");
		}
}


void get_rhs_sparse(const int n, const int *rows, const int m, const int *except_cols, const int bs, const SparseMatrix *A, const double *b, const double *x, double *rhs)
{
	int i, ii, j, jj, k;

	for(i=0, k=0; i<n; i++)
		for(ii=rows[i]; ii < rows[i] + bs && ii<A->n; ii++, k++)
		{
			// for each lost line ii, start with b_ii
			// and remove contributions A_ii,j * x_j 
			// from all rows j that are not lost
			rhs[k] = b[ ii ];

			log_err(SHOW_FAILINFO, "rhs_%d = b_%d - sum_j( sA_%d,j * x_j ), j =", k, ii, ii);

			for(j=A->r[ ii ], jj=0; j<A->r[ ii+1 ]; j++)
			{
				// update jj so that except_cols[jj] + bs > A->c[j]
				while( jj < m && except_cols[jj] + bs <= A->c[j] )
					jj++;


				// if the column of item j is not in the [except_cols[jj],except_cols[jj]+bs-1] set
				if( jj >= m || A->c[j] < except_cols[jj] )
				{
					rhs[k] -= A->v[j] * x[ A->c[j] ];
					log_err(SHOW_FAILINFO, " %d", A->c[j]);
				}
			}
			log_err(SHOW_FAILINFO, "\n");
		}
}


void get_rhs_sparse_with_grad(const int n, const int *rows, const int m, const int *except_cols, const int bs, const SparseMatrix *A, const double *b, const double *g, const double *x, double *rhs)
{
	int i, ii, j, jj, k;

	for(i=0, k=0; i<n; i++)
		for(ii=rows[i]; ii < rows[i] + bs && ii<A->n; ii++, k++)
		{
			// for each lost line ii, start with b_ii
			// and remove contributions A_ii,j * x_j 
			// from all rows j that are not lost
			rhs[k] = b[ ii ] - g[ ii ];

			log_err(SHOW_FAILINFO, "rhs_%d = b_%d - sum_j( sA_%d,j * x_j ), j =", k, ii, ii);

			for(j=A->r[ ii ], jj=0; j<A->r[ ii+1 ]; j++)
			{
				// update jj so that except_cols[jj] + bs > A->c[j]
				while( jj < m && except_cols[jj] + bs <= A->c[j] )
					jj++;


				// if the column of item j is not in the [except_cols[jj],except_cols[jj]+bs-1] set
				if( jj >= m || A->c[j] < except_cols[jj] )
				{
					rhs[k] -= A->v[j] * x[ A->c[j] ];
					log_err(SHOW_FAILINFO, " %d", A->c[j]);
				}
			}
			log_err(SHOW_FAILINFO, "\n");
		}
}


