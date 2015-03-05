#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <signal.h>
#include <mpi.h>

#include "debug.h"

#include "counters.h"

// measures are chosen by defining EXTRAE_EVENTS or GETTIMEOFDAY (this latter one serving as fallback)

#ifdef EXTRAE_EVENTS
#include <extrae_user_events.h>
extrae_type_t extrae_measure_event = 9300000;

extrae_type_t extrae_iteration_event = 9300001;
extrae_type_t extrae_convergence_power_event = 9300002;
extrae_type_t extrae_convergence_significand_event = 9300003;
extrae_type_t extrae_failures_event = 9300004;

extrae_type_t extrae_sdc_power_event = 9300005;
extrae_type_t extrae_sdc_significand_event = 9300006;
extrae_type_t extrae_sdc_revert_event = 9300007;

extrae_value_t extrae_measure_start = 1, extrae_measure_stop = -1, extrae_measure_none = 0;

extrae_type_t log_conv_types [] = {9300001, 9300002, 9300003, 9300004};
extrae_value_t log_conv_vals [4] ;

extrae_type_t log_sdc_types [] = {9300005, 9300006, 9300007};
extrae_value_t log_sdc_vals [3] ;

#else

#ifndef GETTIMEOFDAY
#define GETTIMEOFDAY
#endif

#endif

#ifdef GETTIMEOFDAY
#include <sys/time.h>
struct timeval start_time, stop_time;
#endif 

#ifndef _OMPSS
int nanos_omp_get_thread_num () { return 0; }
int nanos_omp_get_num_threads() { return 1; }
#endif
int get_rank_num () { int mpi_here; MPI_Comm_rank(MPI_COMM_WORLD, &mpi_here); return mpi_here; }
int get_num_ranks() { int mpi_size; MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); return mpi_size; }


#if !defined PERFORMANCE || defined EXTRAE_EVENTS
void log_convergence(const int r UNUSED, const double e UNUSED, const int f UNUSED)
{
	#ifdef EXTRAE_EVENTS
	int exp;
	double significand = frexp(e, &exp);
	significand = ldexp(significand, 63);
	log_conv_vals[0] = r;
	log_conv_vals[1] = (exp+960);
	log_conv_vals[2] = (long long)significand;
	log_conv_vals[3] = f;
	Extrae_nevent(4, log_conv_types, log_conv_vals);
	#endif
	log_out("%d, % e %d\n", r, e, f);
}

void log_sdc(const double e UNUSED, const int f UNUSED)
{
	#ifdef EXTRAE_EVENTS
	int exp;
	double significand = frexp(e, &exp);
	significand = ldexp(significand, 63);
	log_sdc_vals[0] = (exp+960);
	log_sdc_vals[1] = (long long)significand;
	log_sdc_vals[2] = f;
	Extrae_nevent(3, log_sdc_types, log_sdc_vals);
	#endif
	log_out("SDC_CHECK % e %d\n", e, f);
}
#endif

#ifdef EXTRAE_EVENTS
void out_of_time_hdlr(int sig_num UNUSED, siginfo_t * info UNUSED, void * context UNUSED)
{
	fprintf(stderr, "Interrupted by LSF because of time limit !\n");
	Extrae_fini();
	Extrae_shutdown();
	exit(12);
}

void register_sigusr2_handler()
{
	struct sigaction sigact;
	sigact.sa_sigaction = out_of_time_hdlr;
	sigact.sa_flags = SA_SIGINFO;

	sigaction(SIGUSR2, &sigact, (struct sigaction*)NULL);
}
#endif

void unset_measure()
{
	#ifdef EXTRAE_EVENTS
	Extrae_fini();
	Extrae_shutdown();
	#endif
}

void setup_measure()
{
	#ifdef EXTRAE_EVENTS
	if( !Extrae_is_initialized() )
	{
		Extrae_set_threadid_function ((unsigned int (*)(void))&nanos_omp_get_thread_num);
		Extrae_set_numthreads_function ((unsigned int (*)(void))&nanos_omp_get_num_threads);
		Extrae_set_taskid_function ((unsigned int (*)(void))&get_rank_num);
		Extrae_set_numtasks_function ((unsigned int (*)(void))&get_num_ranks);
		Extrae_init();
	}
	#ifdef PERFORMANCE
	else
	{
		Extrae_shutdown();
		Extrae_set_options(0);
		Extrae_restart();
	}
	#endif

	register_sigusr2_handler();

	unsigned int nvals = 3;
	extrae_value_t vals[] = {extrae_measure_start, extrae_measure_none, extrae_measure_stop};
	char* explanations[] = {"solving", "other", "end solving"};

	Extrae_define_event_type( &extrae_measure_event, "measure (P)CG solving", &nvals, vals, explanations);

	nvals = 0;
	Extrae_define_event_type( &extrae_iteration_event, "iteration", &nvals, NULL, NULL);
	Extrae_define_event_type( &extrae_convergence_power_event, "log2 of CG residual norm + 1023", &nvals, NULL, NULL);
	Extrae_define_event_type( &extrae_convergence_significand_event, "significand of CG residual norm", &nvals, NULL, NULL);
	Extrae_define_event_type( &extrae_failures_event, "number of failures in iteration", &nvals, NULL, NULL);

	#endif
}

void start_measure()
{
	#if VERBOSE >= SHOW_DBGINFO
	log_out("Starting measures\n");
	#endif

	#ifdef GETTIMEOFDAY
	gettimeofday( &start_time, NULL );
	#endif 

	#ifdef EXTRAE_EVENTS
	Extrae_event(extrae_measure_event, extrae_measure_start);
	#endif
}

void stop_measure()
{
	#ifdef EXTRAE_EVENTS
	Extrae_event(extrae_measure_event, extrae_measure_stop);
	#endif

	#ifdef GETTIMEOFDAY
	gettimeofday( &stop_time, NULL );
	printf("gettimeofday_Usecs:%e\n", (1e6 * (stop_time.tv_sec - start_time.tv_sec)) + stop_time.tv_usec - start_time.tv_usec);
	#endif 

	#if VERBOSE >= SHOW_DBGINFO
	log_out("Ended measures\n");
	#endif

	#ifdef EXTRAE_EVENTS
	#ifdef _OMPSS
		int i;
		#pragma omp for schedule(static,1)
		for(i=0;i<nanos_omp_get_num_threads();i++)
			Extrae_flush();
	#else
		Extrae_flush();
	#endif
	#endif
}

