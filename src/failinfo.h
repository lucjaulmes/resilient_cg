#ifndef FAILINFO_H_INCLUDED
#define FAILINFO_H_INCLUDED

#include "matrix.h"

// information about block-topology that does not inform on the errors
int get_nb_failblocks();
void get_complete_neighbourset(const int id, char *sieve);

// get info about failures, for the recovery methods
char get_strategy();
int get_nb_failed_blocks();

int pull_failed_block();
int get_failblock_size();
void get_failed_neighbourset(const int *all_lost, const int nb_lost, const int start_block, int *set, int *num);
void get_recovering_blocks_bounds(int *start, int *end, const int *lost, const int nb_lost);

// report a failure
void report_failure(const int block);

// setup methods, called before anything happens
void setup(const Matrix *A, const int fbs, const char fault_strat, const double lambda, const double k);
void swap_in(const Matrix *A, int **fb, double **id, double **nf, char ***n, char **nd);
void swap_out(const Matrix *A, int *fb, double *id, double *nf, char **n, char *nd);
void unset();
void compute_neighbourhoods(const Matrix *mat, const int bs);

// to be called to begin and end each iteration : do measures, allow to decide failed blocks
void start_iterations();
void check_errors();

#endif // FAILINFO_H_INCLUDED

