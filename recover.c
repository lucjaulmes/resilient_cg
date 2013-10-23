#include <math.h>
#include <stdlib.h>

#include "global.h"
#include "matrix.h"
#include "failinfo.h"
#include "solvers.h"
#include "debug.h"

#include "recover.h"

void prepare_x_decorrelated( double *x )
{
	int b, i, bs, start, id;

	for(id=0; id < get_nb_failed_blocks(); id++)
	{
		b = get_failed_block(id);
		get_line_from_block(b, &start, &bs);

		for(i=start; i<start+bs; i++)
			x[ i ] = 0;
	}
}

void prepare_x_uncorrelated( double *x, const double *initial_x )
{
	int b, i, bs, start, id;

	for(id=0; id < get_nb_failed_blocks(); id++)
	{
		b = get_failed_block(id);
		get_line_from_block(b, &start, &bs);

		for(i=start; i<start+bs; i++)
			x[ i ] = initial_x[ i ];
	}
}


void recover_interpolation( const void *A, const double *b, double *x, SolveFunction solver, const int strategy )
{
	recover( A, b, x, solver, 1, strategy );
}


void recover_leastsquares( const void *A, const double *b, double *x, const int strategy )
{
	recover( A, b, x, NULL, 0, strategy );
}


void recover( const void *A, const double *b, double *x, SolveFunction solver, const char A_full_rank, const int strategy )
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
			prepare_x_uncorrelated( x, b );

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
			prepare_x_decorrelated( x );

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


void do_leastsquares( const void *A, const double *b, double *x, const int nb_lost, const int *lost_blocks )
{
	int i, j, k, total_lost = 0, nb = get_nb_blocks(), total_neighbours = 0;
	int max_items = (nb>nb_lost ? nb : nb_lost), startpoints[max_items], bs[max_items];

	// make explicit lists of all rows that were lost
	for(i=0; i<nb_lost; i++)
	{
		get_line_from_block(lost_blocks[i], &startpoints[i], &bs[i]);
		total_lost += bs[i];
	}

	int lost[total_lost];
	k = 0;

	for(i=0; i<nb_lost; i++)
		for(j=0; j<bs[i]; j++)
			lost[k++] = startpoints[i] + j;

	// do the same (sic) with the neighbours (TODO all lost are in the neighbourhood, this should be done once only)
	char neighbourhood[ nb ];

	for(i=0; i<nb; i++)
		neighbourhood[i] = 0;

	for(i=0; i<nb_lost; i++)
		get_complete_neighbourset( lost_blocks[i], neighbourhood );

	j = 0;
	for(i=0; i<nb; i++)
		if( neighbourhood[i] )
		{
			get_line_from_block(i, &startpoints[j], &bs[j]);
			total_neighbours += bs[j];
			j++;
		}
	
	int neighbours[total_neighbours], nb_neighbour_blocks = j;
	k = 0;
	
	for(i=0; i<nb_neighbour_blocks; i++)
		for(j=0; j<bs[i]; j++)
			neighbours[k++] = startpoints[i] + j;
	
	// now we can start doing the actual recovery
	DenseMatrix recup;
	allocate_dense_matrix( total_neighbours, total_lost, &recup );
	double rhs[total_neighbours], interpolated[total_lost];

	get_submatrix(A, neighbours, lost, &recup);

	get_rhs(total_neighbours, neighbours, total_lost, lost, A, b, x, rhs);

	solve_qr_house(&recup, rhs, interpolated);

	// and update the x values we interpolated
	for(i=0; i<total_lost; i++)
		x[ lost[i] ] = interpolated[i];

	deallocate_dense_matrix(&recup);
}


void do_interpolation( const void *A, const double *b, double *x, const int nb_lost, const int *lost_blocks, SolveFunction solver )
{
	int i, j, total_lost = 0, startpoints[nb_lost], bs[nb_lost];

	// make explicit lists of all rows that were lost
	for(i=0; i<nb_lost; i++)
	{
		get_line_from_block(lost_blocks[i], &startpoints[i], &bs[i]);
		total_lost += bs[i];
	}

	int lost[total_lost], k = 0;

	for(i=0; i<nb_lost; i++)
		for(j=0; j<bs[i]; j++)
			lost[k++] = startpoints[i] + j;

	DenseMatrix recup;
	allocate_dense_matrix(total_lost, total_lost, &recup);
	double rhs[total_lost], interpolated[total_lost];

	// get the submatrix for those lines
	get_submatrix(A, lost, lost, &recup);

	// fill in the rhs with the part we need 
	get_rhs(total_lost, lost, total_lost, lost, A, b, x, rhs);

	// now solve with favourite method  : 
	// recup * interpolated = rhs
	solver(&recup, rhs, interpolated);
	
	
	//DEBUG
	{
		double a[total_lost], error_ls = 0;

		// check the what we got from solver
		mult_dense(&recup, interpolated, a);

		log_err("Error of the inner solver is %e, relative error %e\n", sqrt( error_ls ),
			sqrt(error_ls/scalar_product(total_lost, rhs, rhs)));
	}

	// and update the x values we interpolated
	for(i=0; i<total_lost; i++)
		x[ lost[i] ] = interpolated[i];

	deallocate_dense_matrix(&recup);
}

// give rows to but in rhs, and cols to avoid
void rhs_dense(const int n, const int *rows, const int m, const int *except_cols, const void *mat, const double *b, const double *x, double *rhs)
{
	int i, j, k;
	DenseMatrix *A = (DenseMatrix*) mat;

	for(i=0; i<n; i++)
	{
		// for each lost line i, start with b_i
		// and remove contributions A_ij * x_j 
		// from all rows j that are not lost
		rhs[i] = b[ rows[i] ];
		k=0;

		for(j=0; j<A->m; j++)
		{
			// update k so that cols_k >= j
			if( k < m && except_cols[k] < j)
				k++;

			// if j is not a column to avoid
			if( except_cols[k] != j )
				rhs[i] -= A->v[ rows[i] ][j] * x[j];
		}
	}
}

void rhs_sparse(const int n, const int *rows, const int m, const int *except_cols, const void *mat, const double *b, const double *x, double *rhs)
{
	int i, j, k;
	SparseMatrix *A = (SparseMatrix*) mat;

	for(i=0; i<n; i++)
	{
		// for each lost line i, start with b_i
		// and remove contributions A_ij * x_j 
		// from all rows j that are not lost
		rhs[i] = b[ rows[i] ];
		k=0;

		for(j=A->r[ rows[i] ]; j<A->r[ rows[i] +1]; j++)
		{
			// update k so that except_cols_k >= col_j
			while( k < m && except_cols[k] < A->c[j])
				k++;

			// if j is not a except_cols row
			if( except_cols[k] != A->c[j] )
				rhs[i] -= A->v[j] * x[ A->c[j] ];
		}
	}
}

