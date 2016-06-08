#include <math.h>
#include <stdlib.h>

#include "global.h"
#include "matrix.h"
#include "failinfo.h"
#include "debug.h"

#ifdef MATRIX_DENSE
#include "dense_solvers.h"
#else
#include "csparse.h"
#endif

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


void recover( const Matrix *A, const double *b, double *x, const char A_full_rank, const int strategy )
{
	int block, id;

	switch( strategy )
	{
		case SINGLEFAULT :
			block = get_failed_block(0);

			if( A_full_rank )
				do_interpolation(A, b, x, 1, &block);
			else
				do_leastsquares(A, b, x, 1, &block);

			break;

		case MULTFAULTS_UNCORRELATED:
			prepare_x_uncorrelated( x, b, A->n );

			for(id=0; id < get_nb_failed_blocks(); id++)
			{
				block = get_failed_block(id);

				if( A_full_rank )
					do_interpolation(A, b, x, 1, &block);
				else
					do_leastsquares(A, b, x, 1, &block);
			}
			break;

		case MULTFAULTS_DECORRELATED:
			prepare_x_decorrelated( x, A->n );

			for(id=0; id < get_nb_failed_blocks(); id++)
			{
				block = get_failed_block(id);

				if( A_full_rank )
					do_interpolation(A, b, x, 1, &block);
				else
					do_leastsquares(A, b, x, 1, &block);
			}
			break;

		case MULTFAULTS_GLOBAL:
			// get list of failed blocks, group by neighbour clusters, and interpolate
			{
				int flb = get_nb_failed_blocks(), id = 0, i, j, lost[flb], m, set[flb];

				// fill lost with all the failed blocks
				for(id=0; id < flb; id++)
					lost[id] = get_failed_block(id);

				for(id=0; id < flb; id++)
				{
					if (lost[id] < 0)
						continue;

					// get in set the block lost[id] and all its neighbours
					get_failed_neighbourset(lost[id], set, &m);

					// remove from lost all blocks that are in the set that we recover now
					// no need to recover them twice
					lost[id] = -1;
					for(i=id+1; i < flb; i++)
						for(j=0; j<m; j++)
							if( set[j] == lost[i] )
								lost[i] = -1;

					if( A_full_rank )
						do_interpolation(A, b, x, 1, set);
					else
						do_leastsquares(A, b, x, 1, set);
				}
			}
			break;
	}
}


void do_leastsquares( const Matrix *A, const double *b, double *x, const int nb_lost, const int *lost_blocks )
{
	int i, fbs = get_failblock_size(), total_lost = 0, total_blocks = get_nb_failblocks(), nb_neighbours = 0;
	int lost[nb_lost], neighbours[total_blocks];

	// get first row in each block instead of block number
	for(i=0; i<nb_lost; i++)
		lost[i] = fbs * lost_blocks[i];

	// do the same (sic) with the neighbours (NB all "lost" are in "neighbours", this should be done once only)
	char neighbourhood[ total_blocks ];

	for(i=0; i<total_blocks; i++)
		neighbourhood[i] = 0;

	for(i=0; i<nb_lost; i++)
		get_complete_neighbourset( lost_blocks[i], neighbourhood );

	for(i=0; i<total_blocks; i++)
		if( neighbourhood[i] )
		{
			neighbours[ nb_neighbours ] = i * fbs;
			nb_neighbours++;
		}

	Matrix recup;
	double *rhs = (double*)calloc( total_lost, sizeof(double) );

	#ifndef MATRIX_DENSE
	int nnz = 0;
	for(i=0; i<nb_lost; i++)
	{
		int max = lost[i] +fbs;
		if( max > A->n )
			max = A->n ;
		nnz += A->r[ max ] - A->r[ lost[i] ];
	}
	#endif

	allocate_matrix( nb_neighbours, total_lost, nnz, &recup );

	// get the submatrix for those lines
	get_submatrix(A, neighbours, nb_neighbours, lost, nb_lost, fbs, &recup);

	// fill in the rhs with the part we need
	get_rhs(nb_neighbours, neighbours, total_lost, lost, fbs, A, b, x, rhs);

	#ifdef MATRIX_DENSE
	double *interpolated;

	if( nb_lost > 1 )
		interpolated = (double*)calloc( total_lost, sizeof(double) );
	else
		interpolated = &x[ lost[0] ];

	solve_qr_house(&recup, rhs, interpolated);

	if(nb_lost > 1)
	{
		int j, k;
		for(i=0, k=0; i<nb_lost; i++)
			for(j=lost[i]; j<lost[i]+fbs && j<A->n; j++, k++)
				x[ j ] = interpolated[ k ];

		free(interpolated);
	}
	#else
	// from csparse
	cs *submatrix_tr = cs_calloc (1, sizeof (cs)) ;
	submatrix_tr->m = recup.n ;
	submatrix_tr->n = recup.m ;
	submatrix_tr->nzmax = recup.nnz ;
	submatrix_tr->nz = -1 ;
	// don't work with triplets, so has to be compressed column even though we work with compressed row
	submatrix_tr->p = recup.r;
	submatrix_tr->i = recup.c;
	submatrix_tr->x = recup.v;

	cs *submatrix = cs_transpose (submatrix_tr, 1);
	cs_qrsol(submatrix, rhs, 0);
	cs_spfree(submatrix);
	cs_free(submatrix_tr);

	// and update the x values we interpolated, that are returned in rhs
	int j, k;
	for(i=0, k=0; i<nb_lost; i++)
		for(j=lost[i]; j<lost[i]+fbs && j<A->n; j++, k++)
			x[ j ] = rhs[ k ];
	#endif

	deallocate_matrix(&recup);
	free(rhs);
}


void do_interpolation_with_grad( const Matrix *A, const double *b, const double *g, double *x, const int nb_lost, const int *lost_blocks )
{
	int i, fbs = get_failblock_size(), total_lost = nb_lost * fbs, lost[nb_lost];

	// change from block numver to first row in block number
	for(i=0; i<nb_lost; i++)
		lost[i] = lost_blocks[i] * fbs;

	if( lost[nb_lost -1] + fbs > A->n )
		total_lost -= (lost[nb_lost -1] + fbs - A->n);

	Matrix recup;
	double *rhs = (double*)calloc( total_lost, sizeof(double) );

	#ifndef MATRIX_DENSE
	int nnz = 0;
	for(i=0; i<nb_lost; i++)
	{
		int max = lost[i] +fbs;
		if( max > A->n )
			max = A->n ;
		nnz += A->r[ max ] - A->r[ lost[i] ];
	}
	#endif

	allocate_matrix(total_lost, total_lost, nnz, &recup);

	// get the submatrix for those lines
	get_submatrix(A, lost, nb_lost, lost, nb_lost, fbs, &recup);

	// fill in the rhs with the part we need
	if( g == NULL )
		get_rhs(nb_lost, lost, nb_lost, lost, fbs, A, b, x, rhs);
	else
		get_rhs_sparse_with_grad(nb_lost, lost, nb_lost, lost, fbs, A, b, g, x, rhs);


	#ifdef MATRIX_DENSE
	double *interpolated;

	if( nb_lost > 1 )
		interpolated = (double*)calloc( total_lost, sizeof(double) );
	else
		interpolated = &x[ lost[0] ];

	solve_cholesky(&recup, rhs, interpolated);

	#if VERBOSE > SHOW_FAILINFO
	double *verif = (double*) calloc ( total_lost, sizeof(double) ), err = 0.0;

	mult(&recup, interpolated, verif);

	for(i=0; i<total_lost; i++)
		err += (verif[i] - rhs[i]) * (verif[i] - rhs[i]);

	log_err(SHOW_FAILINFO, "error of recovery Cholesky solver is % 1.4e\n", sqrt(err));

	free(verif);
	#endif

	if(nb_lost > 1)
	{
		int j, k;
		for(i=0, k=0; i<nb_lost; i++)
			for(j=lost[i]; j<lost[i]+fbs && j<A->n; j++, k++)
				x[ j ] = interpolated[ k ];

		free(interpolated);
	}
	#else
	// from csparse
	cs *submatrix = cs_calloc (1, sizeof (cs)) ;
	submatrix->m = recup.m ;
	submatrix->n = recup.n ;
	submatrix->nzmax = recup.nnz ;
	submatrix->nz = -1 ;
	// don't work with triplets, so has to be compressed column even though we work with compressed row
	// but since here the matrix is symmetric they are interchangeable
	submatrix->p = recup.r;
	submatrix->i = recup.c;
	submatrix->x = recup.v;

	cs_cholsol(submatrix, rhs, 0);

	// and update the x values we interpolated, that are returned in rhs
	int j, k;
	for(i=0, k=0; i<nb_lost; i++)
		for(j=lost[i]; j<lost[i]+fbs && j<A->n; j++, k++)
			x[ j ] = rhs[ k ];

	cs_free(submatrix);
	#endif


	deallocate_matrix(&recup);
	free(rhs);
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


