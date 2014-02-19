#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

#define NOFAULT 0
#define SINGLEFAULT 1
#define MULTFAULTS_GLOBAL 2
#define MULTFAULTS_UNCORRELATED 3
#define MULTFAULTS_DECORRELATED 4

extern char fault_strat;
extern int MAGIC_BLOCKTORECOVER, MAGIC_ITERATION;

void start_measure();
void stop_measure();


#endif // GLOBAL_H_INCLUDED

