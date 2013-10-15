#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

#include "matrix.h"
#include "solvers.h"
#include "recover.h"

#define SINGLEFAULT 0
#define MULTFAULTS_GLOBAL 1
#define MULTFAULTS_UNCORRELATED 2
#define MULTFAULTS_DECORRELATED 3

double scalar_product( const int n, const double *v, const double *w );

static inline int mod(const int n, const int m)
{
	int rem = n % m;
	return rem == 0 ? m : rem;
}

void start_measure();
double stop_measure();


extern MultFunction mult;
extern RhsFunction get_rhs;
extern SubmatrixFunction get_submatrix;

extern char fault_strat;

#endif // GLOBAL_H_INCLUDED

