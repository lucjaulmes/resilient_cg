#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "debug.h"

#include "failinfo.h"

#define MAX_BLOCKS 100

struct timeval last_it;
double *iterationDuration, *nextFault, lambda = 1000, k = 0.7;
int size, nb_failblocks, failblock_size;
int *failed_block, start_failed = 0, stop_failed = 0;
char **neighbours, *neighbour_data, fault_strat;


// from x a uniform distribution between 0 and 1, the weibull distribution 
// is given by ( -lambda * ln( 1 - x ) ) ^(1/k)
double weibull(const double x)
{
	double inv_k = 1 / k, y;
	y = - log1p( - x ); // - log ( 1 - x )
	y *= lambda; // where 1/lambda ~ mean time between faults

	return pow(y, inv_k);
}

char get_strategy()
{
	return fault_strat;
}

void swap_in(const Matrix *A, int **fb, double **id, double **nf, char ***n, char **nd)
{
	*fb = failed_block;
	*id = iterationDuration;
	*nf = nextFault;
	*n = neighbours;
	*nd = neighbour_data;

	failed_block = NULL;
	iterationDuration = NULL;
	nextFault = NULL;
	neighbours = NULL;
	neighbour_data = NULL;

	// assert (start_failed == stop_failed) 
	// since we remove all the failed blocks from this list before any recoveries

	start_failed = stop_failed = 0;

	setup(A, 8 * failblock_size, fault_strat, lambda, k);
}

void swap_out(const Matrix *A, int *fb, double *id, double *nf, char **n, char *nd)
{
	unset();

	failed_block = fb;
	iterationDuration = id;
	nextFault = nf;
	neighbours = n;
	neighbour_data = nd;

	size = A->n;
	nb_failblocks = (size + failblock_size - 1) / failblock_size;
}

void setup(const Matrix *A, const int fbs, const char strategy, const double lambda_bis, const double k_bis)
{
	size = A->n;
	failblock_size = fbs / sizeof(double);
	nb_failblocks = (size + failblock_size - 1) / failblock_size;

	fault_strat = strategy;
	lambda = lambda_bis;
	k = k_bis;

	int i;
	

	neighbours = calloc( nb_failblocks, sizeof(char*) );
	neighbour_data = calloc( nb_failblocks * nb_failblocks, sizeof(char) );

	failed_block = calloc( nb_failblocks, sizeof(int) );
	iterationDuration = malloc( nb_failblocks * sizeof(double) );
	nextFault = malloc( nb_failblocks * sizeof(double) );

	for(i=0; i<nb_failblocks; i++)
		neighbours[i] = neighbour_data + i * nb_failblocks;

	compute_neighbourhoods(A, failblock_size);

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

void start_iterations()
{
	gettimeofday( &last_it, NULL );
}

void report_failure(const int block)
{
	// ok we've got a failed block. Which one will it be ?
	failed_block[stop_failed] = block;

	stop_failed = (stop_failed + 1) % nb_failblocks;
}

void check_errors()
{
	struct timeval end_it;
	gettimeofday( &end_it, NULL );

	double incDuration = (1e6 * (end_it.tv_sec - last_it.tv_sec)) + end_it.tv_usec - last_it.tv_usec;
	last_it = end_it;

	if( fault_strat == NOFAULT )
		return;

	if( fault_strat == SINGLEFAULT )
	{
		iterationDuration[0] += incDuration ;
		// One fault happened, reset the timer and decide where it happened
		if( iterationDuration[0] > nextFault[0] )
		{
			iterationDuration[0] = 0;
			nextFault[0] = DBL_MAX; //weibull( (double)rand()/RAND_MAX );

			int b = (int)((double)rand() / RAND_MAX * nb_failblocks);

			log_err(SHOW_FAILINFO, "Fault (on block %d of %d), next fault in %e usecs\n", b, nb_failblocks, nextFault[0]);

			report_failure(b);
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

				report_failure( i );

				log_err(SHOW_FAILINFO, "Fault on block %d, next one in %e usecs\n", i, nextFault[i]);
			}
		}
	}
}

// function returning number of failed mem pages
int get_nb_failed_blocks()
{
	int d = stop_failed - start_failed ;
	return d < 0 ? d + nb_failblocks : d;
}

int get_nb_failblocks()
{
	return nb_failblocks;
}

int pull_failed_block()
{
	if( start_failed == stop_failed )
		return -1;
	
	int b = failed_block[start_failed];
	start_failed = ( start_failed + 1 ) % nb_failblocks;

	return b;
}

int get_failblock_size()
{
	return failblock_size;
}

void get_recovering_blocks_bounds(int *start, int *end, const int *lost, const int nb_lost)
{
	int min_block = nb_failblocks + 1, max_block = -1, i, b;
	
	for(i=0; i<nb_lost; i++)
	{
		if( lost[i] < min_block )
			min_block = lost[i];
		if( lost[i] > max_block )
			max_block = lost[i];
	}

	int min_lost_item = min_block * failblock_size, max_lost_item = (max_block + 1) * failblock_size -1;

	// link to blocks
	for(b=0; b<nb_blocks; b++)
	{
		if( min_lost_item >= get_block_start(b) && min_lost_item < get_block_end(b) )
			*start = get_block_start(b);
		if( max_lost_item >= get_block_start(b) && max_lost_item < get_block_end(b) )
			*end = get_block_end(b);
	}
}

void get_complete_neighbourset(const int id, char *sieve)
{
	int i;

	for(i=0; i<nb_failblocks; i++)
		if( i == id ||  neighbours[ id ][i] )
			sieve[i] = 1;
}

// function setting the number and set of lost blocks that are neighbours with block id
void get_failed_neighbourset(const int *all_lost, const int nb_lost, const int start_block, int *set, int *num)
{
	int i, j, k = 0, added[nb_failblocks];

	for(i=0; i<nb_failblocks; i++)
		added[i] = 0;

	*num = 1;
	added[start_block] = 1;
	set[k] = start_block;

	do {
		// search the neighbours of set_k
		// if set_k and i are neighbours and i is found in the failed blocks, add i to set
		for(i=0; i<nb_failblocks; i++)
			if( neighbours[ set[k] ][i] && added[i] == 0 )
			{
				for(j=0; j<nb_lost; j++)
					if( all_lost[j] == i )
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

void compute_neighbourhoods(const Matrix *mat, const int bs)
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



