#ifndef FAILINFO_H_INCLUDED
#define FAILINFO_H_INCLUDED

#include "matrix.h"

// information about block-topology that does not inform on the errors
int get_nb_blocks();
void get_line_from_block(const int b, int *start, int *blocksize);
void get_complete_neighbourset(const int id, char *sieve);

// get info about failures, for the recovery methods
int get_nb_failed_blocks();

int get_failed_block(const int id);
void get_failed_neighbourset(const int block, int *set, int *num);


// setup methods, called before anything happens
void setup(const int n, const int blocksize, const double lambda, const double k, const char fault_strat);
void compute_neighbourhoods_dense(const DenseMatrix *mat, const int bs);
void compute_neighbourhoods_sparse(const SparseMatrix *mat, const int bs);
void unset();

// to be called to begin and end each iteration : do measures, allow to decide failed blocks
void start_iteration();
void stop_iteration();

#endif // FAILINFO_H_INCLUDED

