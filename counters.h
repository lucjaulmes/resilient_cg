#ifndef COUNTERS_H
#define COUNTERS_H

#include <stdio.h>
#include <papi.h>

#include "debug.h"

int counters[] = {PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_L1_DCM,PAPI_L1_ICM,PAPI_L2_TCM}, nb_counters = sizeof(counters)/sizeof(int);
long long *values;
char *names;

void setup_papi()
{
	int i;
	PAPI_event_info_t info;

	// next line also implicitly initializes papi library
	i = PAPI_num_counters();
	
	if( nb_counters > i )
	{
		nb_counters = i;
		log_err(0,"Only %d of the %d PAPI counters will be used\n");
		exit(1);
	}
	
	names = (char*)malloc( PAPI_MAX_STR_LEN * nb_counters * sizeof(char) );
	values = (long long*) malloc( nb_counters * sizeof(long long) );

	/* Check to see if the presets exist */
	for(i=0; i < nb_counters; i++)
	{
		PAPI_event_code_to_name(counters[i], &names[i*PAPI_MAX_STR_LEN]);

		if (PAPI_query_event(counters[i]) != PAPI_OK || PAPI_get_event_info(counters[i], &info) != PAPI_OK || info.count == 0)
		{
			fprintf (stderr,"Counter %d \"%s\" doesn't exist.\n", i, &names[i*PAPI_MAX_STR_LEN]);
			exit(1);
		}
	}
}

void unset_papi()
{
	free(values);
	free(names);
}

void start_papi()
{
	//PAPI_start_counters(counters, nb_counters);
	float realtime, proctime, ipc;
	long long instructions;
	
	if( PAPI_ipc( &realtime, &proctime, &instructions, &ipc ) != PAPI_OK )	
	{
		fprintf (stderr,"Error initializing PAPI_ipc\n");
		exit(1);
	}
}

void reset_papi()
{
	/*
	PAPI_read_counters(values, nb_counters);

	int i;
	for(i=0; i<nb_counters; i++)
		log_out("%s : %lld\n", &names[i*PAPI_MAX_STR_LEN], values[i]);
	*/

	float realtime, proctime, ipc;
	long long instructions;
	
	if( PAPI_ipc( &realtime, &proctime, &instructions, &ipc ) == PAPI_OK )	
		log_out("Since last PAPI_ipc call, we have had : \n%'lld instructions\n%e realtime\n%e processor time\n%e IPC\n");
	else
	{
		fprintf (stderr,"Error initializing PAPI_ipc\n");
		exit(1);
	}
}

void stop_papi()
{
	/*
	PAPI_stop_counters(values, nb_counters);

	int i;
	for(i=0; i<nb_counters; i++)
		log_out("%s : %lld\n", &names[i*PAPI_MAX_STR_LEN], values[i]);
	*/

	float realtime, proctime, ipc;
	long long instructions;
	
	if( PAPI_ipc( &realtime, &proctime, &instructions, &ipc ) == PAPI_OK )	
		log_out("Since last PAPI_ipc call, we have had : \n%'lld instructions\n%e realtime\n%e processor time\n%e IPC\n");
	else
	{
		fprintf (stderr,"Error initializing PAPI_ipc\n");
		exit(1);
	}
}

#endif // COUNTERS_H

