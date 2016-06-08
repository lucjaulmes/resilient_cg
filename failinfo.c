#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "debug.h"

#include "failinfo.h"

#define MAX_BLOCKS 100

struct timeval start_it, end_it;
double *iterationDuration, *nextFault, lambda = 1000, k = 0.7;
int size, nb_failblocks, failblock_size;
int *failed_block, nb_failed_block = 0;
char **neighbours, *neighbour_data;


// from x a uniform distribution between 0 and 1, the weibull distribution
// is given by ( -lambda * ln( 1 - x ) ) ^(1/k)
double weibull(const double x)
{
	double inv_k = 1 / k, y;
	y = - log1p( - x ); // - log ( 1 - x )
	y *= lambda; // where 1/lambda ~ mean time between faults

	return pow(y, inv_k);
}

void set_fixed(const int nb)
{
	nb_failed_block -= nb;
	if( nb_failed_block < 0 )
		nb_failed_block = 0;
}

void setup(const Matrix *A, const int fbs, const double lambda_bis, const double k_bis)
{
	failblock_size = fbs / 8;
	size = A->n;
	nb_failblocks = (8 * size + fbs - 1) / fbs;

	lambda = lambda_bis;
	k = k_bis;

	int i;


	neighbours = calloc( nb_failblocks, sizeof(char*) );
	neighbour_data = calloc( nb_failblocks * nb_failblocks, sizeof(char) );

	for(i=0; i<nb_failblocks; i++)
		neighbours[i] = neighbour_data + i * nb_failblocks;

	compute_neighbourhoods(A, failblock_size);

	iterationDuration = malloc( nb_failblocks * sizeof(double) );
	nextFault = malloc( nb_failblocks * sizeof(double) );
	failed_block = calloc( nb_failblocks, sizeof(int) );

	if( fault_strat == SINGLEFAULT )
	{
		nextFault[0] = weibull( (double)rand()/RAND_MAX );
		iterationDuration[0] = 0;
		log_err(SHOW_FAILINFO, "Single fault initialized. Next fault in %e\n", nextFault[0]);
	}
	else if( fault_strat != NOFAULT )
	{
		for (i=0; i<nb_failblocks; i++)
		{
			nextFault[i] = weibull( (double)rand()/RAND_MAX );
			iterationDuration[i] = 0;
		}
		log_err(SHOW_FAILINFO, "Multiple faults initialized. e.g. next fault on block 0 in %e\n", nextFault[0]);
	}
}

void unset()
{
	free(neighbours);
	free(neighbour_data);
	free(iterationDuration);
	free(nextFault);
	free(failed_block);
}

void start_iteration()
{
	gettimeofday( &start_it, NULL );
}

void stop_iteration()
{
	gettimeofday( &end_it, NULL );

	if( fault_strat == NOFAULT )
		return;

	double incDuration = (1e6 * (end_it.tv_sec - start_it.tv_sec)) + end_it.tv_usec - start_it.tv_usec;

	nb_failed_block = 0;

	if( fault_strat == SINGLEFAULT )
	{
		iterationDuration[0] += incDuration ;
		// One fault happened, reset the timer and decide where it happened
		if( iterationDuration[0] > nextFault[0] )
		{
			iterationDuration[0] = 0;
			nextFault[0] = weibull( (double)rand()/RAND_MAX );

			// ok we've got a failed block. Which one will it be ?
			failed_block[0] = (int)((double)rand() / RAND_MAX * nb_failblocks);
			nb_failed_block = 1;

			log_err(SHOW_FAILINFO, "Fault (on block %d of %d), next fault in %e usecs\n", failed_block[0], nb_failblocks, nextFault[0]);
		}
	}

	else
	{
		// otherwise, go through all blocks and see which failed
		// if they did add to them to failed_block[] and reset timer
		int i;
		for(i=0; i<nb_failblocks; i++)
		{
			iterationDuration[i] += incDuration ;

			if( iterationDuration[i] > nextFault[i] )
			{
				iterationDuration[i] = 0;
				nextFault[i] = weibull( (double)rand()/RAND_MAX );

				failed_block[ nb_failed_block ] = i;

				nb_failed_block++;
				log_err(SHOW_FAILINFO, "Fault on block %d, next one in %e usecs\n", i, nextFault[i]);
			}
		}
	}
}

// function returning number of failed processors
int get_nb_failed_blocks()
{
	return nb_failed_block;
}

int get_nb_failblocks()
{
	return nb_failblocks;
}

// function returning the number of lines and setting their list in *lines for processor id
void report_failed_block(const int block)
{
	failed_block[nb_failed_block] = block;
	nb_failed_block++;
}

int get_failed_block(const int id)
{
	if( id >= nb_failed_block )
		return -1;

	return failed_block[id];
}

int get_failblock_size()
{
	return failblock_size;
}

void get_complete_neighbourset(const int id, char *sieve)
{
	int i;

	for(i=0; i<nb_failblocks; i++)
		if( i == id ||  neighbours[ id ][i] )
			sieve[i] = 1;
}

// function setting the number and set of lost blocks that are neighbours with block id
void get_failed_neighbourset(const int id, int *set, int *num)
{
	int i, j, k = 0, added[nb_failblocks];

	for(i=0; i<nb_failblocks; i++)
		added[i] = 0;

	*num = 1;
	added[id] = 1;
	set[k] = id;

	do {
		// search the neighbours of set_k
		// if set_k and i are neighbours and i is found in the failed blocks, add i to set
		for(i=0; i<nb_failblocks; i++)
			if( neighbours[ set[k] ][i] && added[i] == 0 )
			{
				for(j=0; j<nb_failed_block; j++)
					if( failed_block[j] == i )
					{
						added[i] = 1;
						set[*num] = i;
						(*num)++;
					}
			}
	}
	// if a failed block in set has failed neighbours, we should add them too
	while( ++k < *num );

	// okay now we should really sort the set...
	// should be mostly a small list, partly sorted already
	// so kiss and go for an insertion sort
	int insert;
	for (i = 1; i < *num; i++)
	{
		insert = set[i];

		for (j = i; j > 0 && set[j - 1] > insert ; j--)
			set[j] = set[j - 1];

		set[j] = insert;
	}
}


void compute_neighbourhoods_dense(const DenseMatrix *mat, const int bs)
{
	int i, ii, j, jj, bi, bj;

	// iterate all lines, i points to the start of the block, ii to the line and bi to the number of the block
	for(i=0, bi=0; i < mat->n; i += bs, bi++ )
		for( ii = i; ii < i+bs && ii < mat->n ; ii ++ )

			// iterate all columns, j points to the start of the block, jj to the column and bj to the number of the block
			for(j=0, bj = 0; j < mat->m; j += bs , bj++ )
				for( jj = j; jj < j+bs && jj < mat->m && neighbours[bj][bi] == 0; jj++ )

					if( mat->v[ii][jj] != 0 )
						neighbours[ bj ][ bi ] = 1;
}

void compute_neighbourhoods_sparse(const SparseMatrix *mat, const int bs)
{
	int i, ii, bi, k, bj;

	// iterate all lines, i points to the start of the block, ii to the line and bi to the number of the block
	for(i=0, bi=0; i < mat->n; i += bs, bi++ )
		for( ii = i; ii < i+bs && ii < mat->n ; ii ++ )

			// iterate all columns, k points to the position in mat, and bj to the number of the block
			for(k = mat->r[ii] ; k < mat->r[ii+1] ; k++ )
			{
				bj = mat->c[k] / bs;
				if( mat->v[k] != 0.0 )
					neighbours[ bi ][ bj ] = 1;
 			}
}


