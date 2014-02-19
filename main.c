#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "global.h"
#include "cg.h"
#include "gmres.h"
#include "mmio.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"
#include "counters.h"

#ifdef BACKTRACE
	#include "backtrace.h"
#else 
	#define register_sigsegv_handler()
#endif

int MAGIC_ITERATION = 212;
int MAGIC_BLOCKTORECOVER = 0;

char fault_strat;

// some self-explanatory text functions
void usage(char* arg0)
{
	printf("Usage: %s [options] <matrix-market-filename> [, ...] \n"
			"Possible options are : \n"
			"  -l   lambda       Number (double), meaning 1/mtbf in usec.\n"
			"  -nf               Disabling faults simulation (default).\n"
			"  -sf               Forcing faults to happen no more than one at a time.\n"
			"  -mf  strategy     Enabling multiple faults to happen.\n "
			"                   'strategy' must be one of global, uncorrelated, decorrelated.\n"
			"                    Note : the options -nf, -sf and -mf are mutually exclusive.\n"
			"  -ld  size         Defines size of lost data on failure (in bytes).\n"
			"  -run runs         number of times to run a matrix solving.\n"
			"  -r   restart      number of steps for the restarted gmres.\n"
			"                    0 means standard gmres, without restarting (default).\n\n"
			"All options apply to every following input file. You may re-specify them for each file.\n\n", arg0);
	exit(1);
}

void name_strategy(const char n, char* name)
{
	if( n == NOFAULT )
		strcpy(name, "no_fault");	
	else if( n == SINGLEFAULT )
		strcpy(name, "single_fault");	
	else if( n == MULTFAULTS_GLOBAL )
		strcpy(name, "multiple_faults_global_recovery");	
	else if( n == MULTFAULTS_UNCORRELATED )
		strcpy(name, "multiple_faults_uncorrelated_recovery");	
	else if( n == MULTFAULTS_DECORRELATED )
		strcpy(name, "multiple_faults_decorrelated_recovery");	
	else
		strcpy(name, "unknown_fault_strategy_expect_crashes");	
}

int read_param(int argsleft, char* argv[], double *lambda, int *restart, int *runs, int *fail_size, char *fault_strat)
{
	if( strcmp(argv[0], "-it") == 0 )
	{
		// we want at least the double and a matrix market file after
		if( argsleft <= 2 )
			usage(argv[0]);

		MAGIC_ITERATION = strtod(argv[1], NULL);

		if( MAGIC_ITERATION <= 0 )
			usage(argv[0]);

		return 2;
	}
	else if( strcmp(argv[0], "-btr") == 0 )
	{
		// we want at least the double and a matrix market file after
		if( argsleft <= 2 )
			usage(argv[0]);

		MAGIC_BLOCKTORECOVER = strtod(argv[1], NULL);

		if( MAGIC_BLOCKTORECOVER <= 0 )
			usage(argv[0]);

		return 2;
	}
	else if( strcmp(argv[0], "-l") == 0 )
	{
		// we want at least the double and a matrix market file after
		if( argsleft <= 2 )
			usage(argv[0]);

		*lambda = strtod(argv[1], NULL);

		if( *lambda <= 0 )
			usage(argv[0]);

		return 2;
	}
	else if( strcmp(argv[0], "-ld") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			usage(argv[0]);

		*fail_size = (int) strtol(argv[1], NULL, 10);

		if( *fail_size <= 0 )
			usage(argv[0]);

		return 2;
	}
	else if( strcmp(argv[0], "-runs") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			usage(argv[0]);

		*runs = (int) strtol(argv[1], NULL, 10);

		if( *runs < 0 )
			usage(argv[0]);

		return 2;
	}
	else if( strcmp(argv[0], "-r") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			usage(argv[0]);

		*restart = (int) strtol(argv[1], NULL, 10);

		if( *restart < 0 )
			usage(argv[0]);

		return 2;
	}
	else if( strcmp(argv[0], "-sf") == 0 )
	{
		// we want at least a matrix market file after the switch
		if( argsleft <= 1 )
			usage(argv[0]);

		*fault_strat = SINGLEFAULT;
		// (all strategies equivalent for 1 fault)

		return 1;
	}
	else if( strcmp(argv[0], "-mf") == 0 )
	{
		// we want at least the strategy and a matrix market file after
		if( argsleft <= 2 )
			usage(argv[0]);

		if( strcmp(argv[1], "global") == 0 )
			*fault_strat = MULTFAULTS_GLOBAL;
		else if( strcmp(argv[1], "uncorrelated") == 0 )
			*fault_strat = MULTFAULTS_UNCORRELATED;
		else if( strcmp(argv[1], "decorrelated") == 0 )
			*fault_strat = MULTFAULTS_DECORRELATED;
		else
			usage(argv[0]);

		return 2;
	}
	else
		return 0; // no option regognized, consumed 0 parameters
}

FILE* get_infos_matrix(char *filename, int *n, int *m, int *nnz, int *symmetric)
{
	FILE* input_file = fopen(filename, "r");
	MM_typecode matcode;
	*nnz = 0;

	if(input_file == NULL)
	{
		printf("Error : file %s not valid (check path/read permissions)\n", filename);
		return NULL;
	}

	else if (mm_read_banner(input_file, &matcode) != 0)
		printf("Could not process Matrix Market banner of file %s.\n", filename);

	else if (mm_is_complex(matcode))
		printf("Sorry, this application does not support Matrix Market type of file %s : [%s]\n", 
			filename, mm_typecode_to_str(matcode));

	else if( !mm_is_array(matcode) && (mm_read_mtx_crd_size(input_file, m, n, nnz) != 0 || *m != *n) )
		printf("Sorry, this application does not support the not-array matrix in file %s\n", filename);

	else if( mm_is_array(matcode) && (mm_read_mtx_array_size(input_file, m, n) != 0 || *m != *n) )
		printf("Sorry, this application does not support the array matrix in file %s\n", filename);

	else // hurray, no reasons to fail
	{
		if( *nnz == 0 )
			*nnz = (*m) * (*n);
		*symmetric = mm_is_symmetric(matcode);
		return input_file;
	}

	// if we're here we failed at some point but opened the file
	fclose(input_file);
	return NULL;
}

// main function, where we parse arguments, read files, setup stuff and start the recoveries
int main(int argc, char* argv[])
{
	setbuf(stdout, NULL);

	// if we want to autmatically print bactraces, regsiter handler 
	register_sigsegv_handler();

	if(argc < 2)
		usage(argv[0]);

	int i, j, f, nb_read;
	//srand(time(NULL));
	srand(1591613054);

	double lambda = 100;
	int restart = 0, runs = 1, fail_size = 4096; // default page size ?
	fault_strat = NOFAULT;

	// Iterate over parameters (usually open files)
	for(f=1; f<argc; f += nb_read )

		if( (nb_read = read_param(argc - f, &argv[f], &lambda, &restart, &runs, &fail_size, &fault_strat)) == 0 )
		{
			// if it's not an option, it's a file. Read it (and consume parameter)
			int n, m, nnz, symmetric;
			nb_read = 1;
			FILE* input_file = get_infos_matrix(argv[f], &n, &m, &nnz, &symmetric);

			if( input_file == NULL )
				continue;

			// DEBUG TO MODIFY PROBLEM
			// n = m = 16;

			Matrix matrix;

			#ifdef MATRIX_DENSE // using dense matrices
				allocate_dense_matrix(n, m, &matrix);
				read_dense_Matrix(n, m, nnz, symmetric, &matrix, input_file);

				//compute_neighbourhoods_dense(&dA, fail_size);
				printf("matrix_format:DENSE ");

			#else // by default : using sparse matrices
				allocate_sparse_matrix(n, m, nnz * (1+symmetric), &matrix);
				read_sparse_Matrix(n, m, nnz, symmetric, &matrix, input_file);

				//compute_neighbourhoods_sparse(&dA, fail_size);
				printf("matrix_format:SPARSE ");

			#endif 

			fclose(input_file);

			// now show infos
			printf("executable:%s File:%s problem_size:%d ", argv[0], argv[f], n);

			if(symmetric)
				printf("matrix_symmetric:yes method:ConjugateGradient ");
			else if( restart > 0 )
				printf("matrix_symmetric:no method:restarted_GMRES(%d) ", restart);
			else
				printf("matrix_symmetric:no method:full_GMRES ");

			char strat[40];
			name_strategy(fault_strat, strat);

			printf("lambda:%e strategy:%s\n", lambda, strat);

			#if VERBOSE > FULL_VERBOSE
			print(&matrix);
			#endif

			setup_measure();

			// a few vectors for rhs of equation, solution and verification
			double *b, *x, *s;
			b = (double*)malloc( n * sizeof(double) );
			x = (double*)malloc( n * sizeof(double) );
			s = (double*)malloc( n * sizeof(double) );

			// interesting stuff is here
			for(j=0;j<runs;j++)
			{
				if( runs > 1 )
					printf("run:%d ", j);

				// generate random rhs to problem, and initialize first guess to 0
				double range = (double) 1;

				for(i=0; i<n; i++)
				{
					b[i] = ((double)rand() / (double)RAND_MAX ) * range - range/2;
					x[i] = 0.0;
				}

				// do some setup for the resilience part
				setup(n, fail_size, lambda, 0.7);

				// if symmetric, solve with conjugate gradient method
				if(symmetric)
					solve_cg(n, &matrix, b, x, 1e-10 );
				//otherwise, gmres
				else
					solve_gmres(n, &matrix, b, x, 1e-10 , restart);

				// compute verification
				mult(&matrix, x, s);

				// deallocate everything we have allocated for this solving
				unset();

				// do displays (verification, error)
				double err = 0, norm_b = scalar_product(n, b, b);
				for(i=0; i < n; i++)
				{
					double e_i = b[i] - s[i];
					err += e_i * e_i;
				}

				printf("Verification : euclidian distance to solution ||Ax-b||^2 = %e , ||Ax-b||/||b|| = %e\n", err, sqrt(err/norm_b));
			}

			unset_measure();

			deallocate_matrix(&matrix);
			free(b);
			free(x);
			free(s);
		}

	return 0;
}

