#include <math.h>
#include <stdlib.h>
#include <float.h>

#include "global.h"
#include "matrix.h"
#include "failinfo.h"
#include "debug.h"

#include "csparse.h"

#include "pcg.h"
#include "recover.h"

void prepare_x_decorrelated( double *x , const int n, const int *lost_blocks, const int nb_lost )
{
	int b, i, fbs = get_failblock_size();

	for(b=0; b < nb_lost; b++)
	{
		for(i=lost_blocks[b]*fbs; i<lost_blocks[b+1]*fbs && i < n; i++)
			x[ i ] = 0;
	}
}

void prepare_x_uncorrelated( double *x, const double *initial_x , const int n, const int *lost_blocks, const int nb_lost )
{
	int b, i, fbs = get_failblock_size();

	for(b=0; b < nb_lost; b++)
	{
		for(i=lost_blocks[b]*fbs; i<lost_blocks[b+1]*fbs && i < n; i++)
			x[ i ] = initial_x[ i ];
	}
}


void recover_xk( const Matrix *A, const double *b, const double *g, double *x, const Precond *M, const int strategy, int *lost_blocks, const int nb_lost )
{
	int block, id;

	if( strategy == NOFAULT )
		fprintf(stderr, "Warning : %d fault(s) with NOFAULT strategy, falling back to global if more than 1\n", nb_lost);

	if( nb_lost == 1 )
	{
		block = lost_blocks[0];

		do_single_interpolation(A, b, g, x, block, M->S[block], M->N[block]);

		return;
	}


	switch( strategy )
	{
		case MULTFAULTS_UNCORRELATED:
			prepare_x_uncorrelated( x, b, A->n, lost_blocks, nb_lost );

			for(id=0; id < nb_lost; id++)
			{
				block = lost_blocks[id];

				do_single_interpolation(A, b, g, x, block, M->S[block], M->N[block]);
			}
			break;

		case SINGLEFAULT :

			fprintf(stderr, "Warning : multiple faults with SINGLEFAULT strategy, falling back to decorrelated\n");

		case MULTFAULTS_DECORRELATED:
			prepare_x_decorrelated( x, A->n, lost_blocks, nb_lost );

			for(id=0; id < nb_lost; id++)
			{
				block = lost_blocks[id];

				do_single_interpolation(A, b, g, x, block, M->S[block], M->N[block]);
			}
			break;

		case NOFAULT :
		case MULTFAULTS_GLOBAL:
			// get list of failed blocks, group by neighbour clusters, and interpolate
			{
				int flb = nb_lost, id = 0, i, j, lost[flb], m, set[flb];

				// fill lost with all the failed blocks
				for(id=0; id < flb; id++)
				{
					lost[id] = lost_blocks[id];

					// just to be safe, remove out of bounds items
					if(lost[id] >= get_nb_failblocks())
						lost[id] = -1;
				}

				for(id=0; id < flb; id++)
				{
					if (lost[id] < 0)
						continue;

					// get in set[] the block lost[id] and all its neighbours
					get_failed_neighbourset(lost_blocks, nb_lost, lost_blocks[id], set, &m);
					
					log_err(LIGHT_VERBOSE, "global strategy found that neighbourhood of failed blocks for block %d is of size %d\n", lost[id], m);
					
					if( m > 1 )
						do_multiple_interpolation(A, b, g, x, m, set, M->S, M->N);
					else
						do_single_interpolation(A, b, g, x, lost[id], M->S[lost[id]], M->N[lost[id]]);

					// remove from lost[] all blocks that are in the set that we recover now
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
		get_rhs_with_grad(1, &lost, 1, &lost, total_lost, A, b, g, x, rhs);

	// from csparse
	do_interpolation(rhs, &x[lost], total_lost, S, N);
	free(rhs);
}

void do_multiple_interpolation(const Matrix *A, const double *b, const double *g, double *x, const int nb_lost, const int *lost_blocks, css **S UNUSED, csn **N UNUSED)
{
	int i, j, lost[nb_lost], fbs = get_failblock_size(), total_lost = nb_lost * fbs, nnz = 0;
	Matrix sm;

	// lost contains starting row of each block
	for(i=0; i<nb_lost; i++)
	{
		lost[i] = lost_blocks[i] * fbs;

		int end_block = lost[i] + fbs;
		if( end_block > A->n )
		{
			total_lost -= end_block - A->n;
			end_block = A->n;
		}
		nnz += A->r[end_block] - A->r[ lost[i] ];
	}

	allocate_matrix(total_lost, total_lost, nnz, &sm);
	get_submatrix(A, lost, nb_lost, lost, nb_lost, fbs, &sm);

	double	*rhs = (double*)calloc( total_lost, sizeof(double) );

	// fill in the rhs with the part we need for our whole multiple problem
	if( g == NULL )
		get_rhs(nb_lost, lost, nb_lost, lost, fbs, A, b, x, rhs);
	else
		get_rhs_with_grad(nb_lost, lost, nb_lost, lost, fbs, A, b, g, x, rhs);

	// from csparse
	cs submatrix;
	submatrix.m = sm.m ;
	submatrix.n = sm.n ;
	submatrix.nzmax = sm.nnz ;
	submatrix.nz = -1 ;
	// don't work with triplets, so has to be compressed column even though we work with compressed row
	// but since here the matrix is symmetric they are interchangeable
	submatrix.p = sm.r;
	submatrix.i = sm.c;
	submatrix.x = sm.v;

	cs_cholsol(&submatrix, rhs, 0);

	for(i=0; i<nb_lost; i++)
		for(j=0; j < fbs && lost[i] + j < A->n; j++)
			x[ lost[i] + j ] = rhs[ i * fbs + j];

	deallocate_matrix(&sm);
	free(rhs);
}

void get_rhs(const int n, const int *rows, const int m, const int *except_cols, const int bs, const Matrix *A, const double *b, const double *x, double *rhs)
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


void get_rhs_with_grad(const int n, const int *rows, const int m, const int *except_cols, const int bs, const Matrix *A, const double *b, const double *g, const double *x, double *rhs)
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


