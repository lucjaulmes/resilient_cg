#include <math.h>
#include <stdlib.h>

#include "global.h"
#include "matrix.h"
#include "failinfo.h"
#include "solvers.h"
#include "debug.h"

#include "recover.h"

void prepare_x_decorrelated( double *x , const int n )
{
	int b, i, id;

	for(id=0; id < get_nb_failed_blocks(); id++)
	{
		b = get_failed_block(id);

		for(i=b*BS; i<(b+1)*BS && i < n; i++)
			x[ i ] = 0;
	}
}

void prepare_x_uncorrelated( double *x, const double *initial_x , const int n)
{
	int b, i, id;

	for(id=0; id < get_nb_failed_blocks(); id++)
	{
		b = get_failed_block(id);

		for(i=b*BS; i<(b+1)*BS && i < n; i++)
			x[ i ] = initial_x[ i ];
	}
}


void recover_interpolation( const Matrix *A, const double *b, double *x, SolveFunction solver, const int strategy )
{
	recover( A, b, x, solver, 1, strategy );
}


void recover_leastsquares( const Matrix *A, const double *b, double *x, const int strategy )
{
	recover( A, b, x, NULL, 0, strategy );
}


void recover( const Matrix *A, const double *b, double *x, SolveFunction solver, const char A_full_rank, const int strategy )
{
	int block, id;

	switch( strategy )
	{
		case SINGLEFAULT :
			block = get_failed_block(0);

			if( A_full_rank )
				do_interpolation(A, b, x, 1, &block, solver);
			else
				do_leastsquares(A, b, x, 1, &block);

			break;

		case MULTFAULTS_UNCORRELATED:
			prepare_x_uncorrelated( x, b, A->n );

			for(id=0; id < get_nb_failed_blocks(); id++)
			{
				block = get_failed_block(id);

				if( A_full_rank )
					do_interpolation(A, b, x, 1, &block, solver);
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
					do_interpolation(A, b, x, 1, &block, solver);
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
					do_interpolation(A, b, x, 1, set, solver);
				else
					do_leastsquares(A, b, x, 1, set);
			}
			}
			break;
	}
}


void do_leastsquares( const Matrix *A, const double *b, double *x, const int nb_lost, const int *lost_blocks )
{
	int i, total_lost = 0, total_blocks = get_nb_blocks(), nb_neighbours = 0;
	int lost[nb_lost], neighbours[total_blocks];

	// get first row in each block instead of block number
	for(i=0; i<nb_lost; i++)
		lost[i] = BS * lost_blocks[i];

	// do the same (sic) with the neighbours (NB all "lost" are in "neighbours", this should be done once only)
	char neighbourhood[ total_blocks ];

	for(i=0; i<total_blocks; i++)
		neighbourhood[i] = 0;

	for(i=0; i<nb_lost; i++)
		get_complete_neighbourset( lost_blocks[i], neighbourhood );

	for(i=0; i<total_blocks; i++)
		if( neighbourhood[i] )
		{
			neighbours[ nb_neighbours ] = i * BS;
			nb_neighbours++;
		}
	
	// now we can start doing the actual recovery
	#ifndef MATRIX_DENSE
	int nnz = 0;
	for(i=0; i<nb_lost; i++)
		nnz += A->r[ lost[i] +BS ] - A->r[ lost[i] ];
	#endif

	//Matrix recup;
	//allocate_matrix( nb_neighbours, total_lost, nnz, &recup );
	DenseMatrix recup;
	allocate_dense_matrix( nb_neighbours, total_lost, &recup );
	double *rhs, *interpolated;
	rhs = (double*)calloc( total_lost, sizeof(double) );

	if( nb_lost > 1 )
		interpolated = (double*)calloc( total_lost, sizeof(double) );
	else
		interpolated = &x[ lost[0] ];

	//get_submatrix(A, neighbours, nb_neighbours, lost, nb_lost, BS, &recup);
	get_dense_submatrix(A, neighbours, nb_neighbours, lost, nb_lost, BS, &recup);

	get_rhs(nb_neighbours, neighbours, total_lost, lost, BS, A, b, x, rhs);

	solve_qr_house(&recup, rhs, interpolated);


	//deallocate_matrix(&recup);
	deallocate_dense_matrix(&recup);
	free(rhs);

	// and update the x values we interpolated
	if(nb_lost > 1)
	{
		int j, k;
		for(i=0, k=0; i<nb_lost; i++)
			for(j=lost[i]; j<lost[i]+BS && j<A->n; j++, k++)
				x[ j ] = interpolated[ k ];

		free(interpolated);
	}
}


void do_interpolation( const Matrix *A, const double *b, double *x, const int nb_lost, const int *lost_blocks, SolveFunction solver )
{
	int i, total_lost = nb_lost * BS, lost[nb_lost];
	
	// change from block numver to first row in block number
	for(i=0; i<nb_lost; i++)
		lost[i] = lost_blocks[i] * BS;

	if( lost[nb_lost -1] + BS > A->n )
		total_lost -= (lost[nb_lost -1] + BS - A->n);

	#ifndef MATRIX_DENSE
	int nnz = 0;
	for(i=0; i<nb_lost; i++)
		nnz += A->r[ lost[i] +BS ] - A->r[ lost[i] ];
	#endif

	//Matrix recup;
	//allocate_matrix(total_lost, total_lost, nnz, &recup);
	DenseMatrix recup;
	allocate_dense_matrix(total_lost, total_lost, &recup);

	double *rhs, *interpolated;
	rhs = (double*)calloc( total_lost, sizeof(double) );

	if( nb_lost > 1 )
		interpolated = (double*)calloc( total_lost, sizeof(double) );
	else
		interpolated = &x[ lost[0] ];

	// get the submatrix for those lines
	//get_submatrix(A, lost, nb_lost, lost, nb_lost, BS, &recup);
	get_dense_submatrix(A, lost, nb_lost, lost, nb_lost, BS, &recup);

	// fill in the rhs with the part we need 
	get_rhs(nb_lost, lost, nb_lost, lost, BS, A, b, x, rhs);

	// now solve with favourite method  : 
	// recup * interpolated = rhs
	solver(&recup, rhs, interpolated);
	
	
	#if VERBOSE > SHOW_FAILINFO
	double *verif = (double*)malloc(total_lost * sizeof(double)), diff = 0.0;
	//mult(&recup, interpolated, verif);
	mult_dense(&recup, interpolated, verif);

	for(i=0; i<total_lost; i++)
		diff += ( rhs[i] - verif[i] ) * ( rhs[i] - verif[i] );
	
	log_out("Total error of solving is %e \n", sqrt(diff));
	free(verif);
	#endif

	// and update the x values we interpolated

	//deallocate_matrix(&recup);
	deallocate_dense_matrix(&recup);
	free(rhs);

	if(nb_lost > 1)
	{
		int j, k;
		for(i=0, k=0; i<nb_lost; i++)
			for(j=lost[i]; j<lost[i]+BS && j<A->n; j++, k++)
				x[ j ] = interpolated[ k ];

		free(interpolated);
	}
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

			for(j=0; j<A->m; j++)
			{
				// update l so that except_cols_jj >= j
				if( jj < m && except_cols[jj] + bs <= j)
					jj++;

				// if j is not in a columns-to-avoid set
				if( j < except_cols[jj] )
					rhs[k] -= A->v[ ii ][j] * x[j];
			}
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

			for(j=A->r[ ii ], jj=0; j<A->r[ ii +1 ]; j++)
			{
				// update jj so that except_cols[jj] + bs > A->c[j]
				while( jj < m && except_cols[jj] + bs <= A->c[j] )
					jj++;

				if( jj >= m )
					break;

				// if the column of item j is not in the [except_cols[jj],except_cols[jj]+bs-1] set
				if( A->c[j] < except_cols[jj] )
					rhs[k] -= A->v[j] * x[ A->c[j] ];
			}
		}
}

