#ifndef FAILINFO_H_INCLUDED
#define FAILINFO_H_INCLUDED

#include "matrix.h"

// information about block-topology that does not inform on the errors
int get_nb_failblocks();
void get_complete_neighbourset(const int id, char *sieve);

// get info about failures, for the recovery methods
int get_nb_failed_blocks();

void report_failed_block(const int block);
void set_fixed(const int nb);

int get_failed_block(const int id);
int get_failblock_size();
void get_failed_neighbourset(const int block, int *set, int *num);


// setup methods, called before anything happens
void setup(const Matrix *A, const int fbs, const double lambda, const double k);
void compute_neighbourhoods_dense(const DenseMatrix *mat, const int bs);
void compute_neighbourhoods_sparse(const SparseMatrix *mat, const int bs);
void unset();

#ifdef MATRIX_DENSE
	#define compute_neighbourhoods(mat, bs) compute_neighbourhoods_dense(mat, bs)
#else
	#define compute_neighbourhoods(mat, bs) compute_neighbourhoods_sparse(mat, bs)
#endif

// to be called to begin and end each iteration : do measures, allow to decide failed blocks
void start_iteration();
void stop_iteration();

#endif // FAILINFO_H_INCLUDED

