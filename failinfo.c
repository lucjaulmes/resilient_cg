#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

#include "global.h"

#include "failinfo.h"

#define MAX_BLOCKS 100

struct timeval start, stop;
double *iterationDuration, *nextFault, lambda = 1000, k = 0.7;
int size, bs, nb_blocks;
int *failed_block, nb_failed_block;
char **neighbours, *neighbour_data, sf;


// from x a uniform distribution between 0 and 1, the weibull distribution 
// is given by ( -lambda * ln( 1 - x ) ) ^(1/k)
double weibull(const double x)
{
	double inv_k = 1 / k, y;
	y = - log1p( - x ); // - log ( 1 - x )
	y *= lambda; // where 1/lambda ~ mean time between faults

	// hop hop debug
	//return DBL_MAX;
	return pow(y, inv_k);
}

void setup(const int n, const int blocksize, const double lambda_bis, const double k_bis, const char fault_strat)
{
	size = n;
	bs = blocksize;
	nb_blocks = (n+bs-1)/bs;

	lambda = lambda_bis;
	k = k_bis;
	sf = (fault_strat == SINGLEFAULT);

	int i;
	

	neighbours = malloc( nb_blocks * sizeof(char*) );
	neighbour_data = calloc( nb_blocks * nb_blocks, sizeof(char) );

	for(i=0; i<nb_blocks; i++)
		neighbours[i] = neighbour_data + i * nb_blocks;


	if( sf )
	{
		iterationDuration = malloc( sizeof(double) );
		nextFault = malloc( sizeof(double) );
		failed_block = calloc( 1, sizeof(int) );

		nextFault[0] = weibull( (double)rand()/RAND_MAX );
		iterationDuration[0] = 0;
	}
	else
	{
		iterationDuration = malloc( nb_blocks * sizeof(double) );
		nextFault = malloc( nb_blocks * sizeof(double) );
		failed_block = calloc( nb_blocks, sizeof(int) );

		for (i=0; i<nb_blocks; i++)
		{
			nextFault[i] = weibull( (double)rand()/RAND_MAX );
			iterationDuration[i] = 0;
		}
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
	gettimeofday( &start, NULL );
}

void stop_iteration()
{
	gettimeofday( &stop, NULL );
	double incDuration = (1000000 * (stop.tv_sec - start.tv_sec)) + stop.tv_usec - start.tv_usec;

	if( sf )
		fprintf(stderr, "[%e usecs]  ", iterationDuration[0]);

	nb_failed_block = 0;

	if( sf )
	{
		iterationDuration[0] += incDuration ;
		// One fault happened, reset the timer and decide where it happened
		if( iterationDuration[0] > nextFault[0] )
		{
			iterationDuration[0] = 0;
			nextFault[0] = weibull( (double)rand()/RAND_MAX );

			// ok we've got a failed block. And it will be....
			failed_block[0] = (int)((double)rand() / RAND_MAX * nb_blocks);
			nb_failed_block = 1;

			fprintf(stderr, "\nFault (on block %d), next fault in %e usecs\n", failed_block[0], nextFault[0]);
		}
	}

	else
	{
		// otherwise, go through all blocks and see which failed
		// if they did add to them to failed_block[] and reset timer
		int i;
		for(i=0; i<nb_blocks; i++)
		{
			iterationDuration[i] += incDuration ;

			if( iterationDuration[i] > nextFault[i] )
			{
				iterationDuration[i] = 0;
				nextFault[i] = weibull( (double)rand()/RAND_MAX );

				failed_block[ nb_failed_block ] = i;

				nb_failed_block++;
				fprintf(stderr, "\nNext fault on block %d in %e usecs\n", failed_block[0], nextFault[0]);
			}
		}
	}
}

// function returning number of failed processors
int get_nb_failed_blocks()
{
	return nb_failed_block;
}

int get_nb_blocks()
{
	return nb_blocks;
}

// function returning the number of lines and setting their list in *lines for processor id
int get_failed_block(const int id)
{
	if( id >= nb_failed_block )
		return -1;

	return failed_block[id];
}

void get_complete_neighbourset(const int id, char *sieve)
{
	int i;

	for(i=0; i<nb_blocks; i++)
		if( i == id ||  neighbours[ id ][i] )
			sieve[i] = 1;
}

// function setting the number and set of lost blocks that are neighbours with block id
void get_failed_neighbourset(const int id, int *set, int *num)
{
	int i, j, k = 0, added[nb_blocks];

	for(i=0; i<nb_blocks; i++)
		added[i] = 0;

	*num = 1;
	added[id] = 1;
	set[k] = id;

	do {
		// search the neighbours of set_k
		// if set_k and i are neighbours and i is found in the failed blocks, add i to set
		for(i=0; i<nb_blocks; i++)
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

void get_line_from_block(const int b, int *start, int *blocksize)
{
	if( b == 0 )
	{
		*start = 0;
		*blocksize = (size % bs) ? (size % bs) : bs;
	}
	else if( size % bs )
	{
		*start = bs * (b - 1) + (size % bs);
		*blocksize = bs;
	}
	else
	{
		*start = bs * b;
		*blocksize = bs;
	}
}

void compute_neighbourhoods_dense(const DenseMatrix *mat, const int bs)
{
	int i, ii, j, k, block_j, off = mat->n % bs, n = mat->n;

	// suppose they are all zero, thus block-diagonal matrix for block size bs (was calloc'd)
	//for( i=0; i < (n+bs-1)/bs; i++ )
	//	for( j=0; j < (n+bs-1)/bs; j++ )
	//		neighbours[i][j] = 0;

	// for the first row-block that is not necessarily of the same size as the others
	for( k=0; k<off; k++ )
		for( j=0; j<n; j++ )
			if( mat->v[k][j] != 0 )
			{
				block_j = (j+bs-off)/bs;
				if( off == 0 )
					block_j--;

				neighbours[ block_j ][0] = 1;
			}

	// for each row-block i, and within it each row k, we go through the columns j
	for( i=off, ii = (off ? 1 : 0); i < n; i+=bs, ii++ )
		for( k=0; k<bs; k++ )
			for( j=0; j<mat->n; j++ )
				if( mat->v[i+k][j] != 0 )
				{
					block_j = (j+bs-off)/bs;
					if( off == 0 )
						block_j--;

					neighbours[ block_j ][ ii ] = 1;
				}
}

void compute_neighbourhoods_sparse(const SparseMatrix *mat, const int bs)
{
	int i, ii, k, block_col, off = mat->n % bs, n = mat->n;

	// suppose they are all zero, thus block-diagonal matrix for block size bs (was calloc'd)
	//for( i=0; i < (n+bs-1)/bs; i++ )
	//	for( j=0; j < (n+bs-1)/bs; j++ )
	//		neighbours[i][j] = 0;
	
	// for the first row-block that is not necessarily of the same size as the others
	for( k=0; k<mat->r[off+1]; k++ )
		if( mat->v[k] != 0 )
		{
			block_col = (mat->c[k]+bs-off)/bs;
			if( off == 0 )
				block_col--;

			neighbours[ block_col ][0] = 1;
		}

	// for each row-block i, and each element k within this row-block, we go through the columns j
	for( i=off, ii = (off ? 1 : 0); i < n; i+=bs, ii++ )
		for( k = mat->r[i] ; k < mat->r[i+bs] ; k++ )
			if( mat->v[k] != 0 )
			{
				block_col = (mat->c[k]+bs-off)/bs;
				if( off == 0 )
					block_col--;

				neighbours[ block_col ][ ii ] = 1;
			}
}


