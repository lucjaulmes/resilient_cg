#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

#define NOFAULT 0
#define SINGLEFAULT 1
#define MULTFAULTS_GLOBAL 2
#define MULTFAULTS_UNCORRELATED 3
#define MULTFAULTS_DECORRELATED 4

extern char fault_strat;
extern int BS;

double scalar_product( const int n, const double *v, const double *w );

void start_measure();
double stop_measure();


#endif // GLOBAL_H_INCLUDED

