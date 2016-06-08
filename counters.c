#include <stdlib.h>
#include <stdio.h>

#include "debug.h"

#include "counters.h"

// measures are chosen by defining EXTRAE_EVENTS, PAPICOUNTERS, or GETTIMEOFDAY (this latter one serving as fallback)
#define FALLBACK_GETTIMEOFDAY

#ifdef EXTRAE_EVENTS
#undef FALLBACK_GETTIMEOFDAY
#include <extrae_user_events.h>
extrae_type_t extrae_measure_event = 1234567;
extrae_value_t extrae_measure_start = 1, extrae_measure_stop = 0;
#endif


#ifdef PAPICOUNTERS
#undef FALLBACK_GETTIMEOFDAY
#include <papi.h>
long long papi_cycles_start, papi_usec_start;
#endif


#ifdef FALLBACK_GETTIMEOFDAY
#ifndef GETTIMEOFDAY
#define GETTIMEOFDAY
#endif
#endif

#ifdef GETTIMEOFDAY
#include <sys/time.h>
struct timeval start_time, stop_time;
#endif


void unset_measure()
{
	#ifdef EXTRAE_EVENTS
	Extrae_fini();
	Extrae_shutdown();
	#endif

	#ifdef PAPICOUNTERS
	PAPI_shutdown();
	#endif
}

void setup_measure()
{
	#ifdef EXTRAE_EVENTS
	Extrae_init();

	unsigned nvals = 2;
	extrae_value_t vals[] = {extrae_measure_start, extrae_measure_stop};
	char* explanations[] = {"solving", "other"};

	Extrae_define_event_type( &extrae_measure_event, "measure Krylov-method solving", &nvals, vals, explanations);
	#endif

	#ifdef PAPICOUNTERS
	if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT )
	{
		printf("Library initialization error! \n");
		exit(1);
	}
	#endif
}

void start_measure()
{
	log_out("Starting measures\n");
	#ifdef EXTRAE_EVENTS
	Extrae_event(extrae_measure_event, extrae_measure_start);
	#endif

	#ifdef PAPICOUNTERS
	papi_cycles_start = PAPI_get_real_cyc();
	papi_usec_start = PAPI_get_real_usec();
	#endif

	#ifdef GETTIMEOFDAY
	gettimeofday( &start_time, NULL );
	#endif
}

void stop_measure()
{
	#ifdef EXTRAE_EVENTS
	Extrae_event(extrae_measure_event, extrae_measure_stop);
	log_out("Ended measures\n");
	#endif

	#ifdef PAPICOUNTERS
	long long papi_usec_end = PAPI_get_real_usec(), papi_cycles_end = PAPI_get_real_cyc();
	printf("Cycles:%lld\n", papi_cycles_end - papi_cycles_start);
	printf("Usecs:%lld\n", papi_usec_end - papi_usec_start);
	#endif

	#ifdef GETTIMEOFDAY
	gettimeofday( &stop_time, NULL );
	printf("gettimeofday_Usecs:%e\n", (1e6 * (stop_time.tv_sec - start_time.tv_sec)) + stop_time.tv_usec - start_time.tv_usec);
	#endif
}

