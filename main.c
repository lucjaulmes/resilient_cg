#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "global.h"
#include "cg.h"
#include "gmres.h"
#include "mmio.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"

#ifdef PAPICOUNTERS
#include "counters.h"
#endif

// pointers to functions
char fault_strat;
int BS;

// useful vector functions
double scalar_product( const int n, const double *v, const double *w )
{
	int i;
	double r = 0;

	for(i=0; i<n; i++)
		r += v[i] * w[i];

	return r;
}

struct timeval start_time, stop_time;
void start_measure()
{
	log_out("starting the measures\n");
	#ifdef PAPICOUNTERS
	start_papi();
	#endif
	gettimeofday( &start_time, NULL );
}

double stop_measure()
{
	log_out("ending the measures\n");
	gettimeofday( &stop_time, NULL );
	#ifdef PAPICOUNTERS
	stop_papi();
	#endif
	return (1e6 * (stop_time.tv_sec - start_time.tv_sec)) + stop_time.tv_usec - start_time.tv_usec;
}

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
			"  -bs  blocksize    force size of the blocks in block-row operations ;\n"
			"                    also size of lost data on failure.\n"
			"  -nb  blocks       Defines the number of blocks in which to divide operations ;\n"
			"                    their size will depdend on the matrix' size.\n"
			"                    Note : the options -bs and -nb are mutually exclusive.\n"
			"  -r   restart      number of steps for the restarted gmres.\n"
			"                    0 means standard gmres, without restarting (default).\n\n"
			"Options apply to all following input files. You may re-specify them for each file.\n\n", arg0);
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

// main function, where we parse arguments, read files, setup stuff and start the recoveries
int main(int argc, char* argv[])
{
	setbuf(stdout, NULL);

	if(argc < 2)
		usage(argv[0]);

	int i, f;
	srand(time(NULL));

	double lambda = 100;
	int restart = 0, blocks = 0;
	BS = 64; // educated guess ? could we get something from a like of omp_get_num_threads() ?
	fault_strat = NOFAULT;
	char BS_defined = 0;

	// Iterate over parameters (usually open files)
	for(f=1; f<argc; f++)
		if( strcmp(argv[f], "-l") == 0 )
		{
			// we want at least the double and a matrix market file after
			if( f+2 >= argc )
				usage(argv[0]);

			lambda = strtod(argv[f+1], NULL);

			if( lambda <= 0 )
				usage(argv[0]);

			f++;
			continue;
		}
		else if( strcmp(argv[f], "-nb") == 0 )
		{
			// we want at least the integer and a matrix market file after
			if( f+2 >= argc || BS_defined )
				usage(argv[0]);

			blocks = (int) strtol(argv[f+1], NULL, 10);

			if( blocks <= 0 )
				usage(argv[0]);

			f++;
			BS_defined = 1;
			continue;
		}
		else if( strcmp(argv[f], "-bs") == 0 )
		{
			// we want at least the integer and a matrix market file after
			if( f+2 >= argc || BS_defined )
				usage(argv[0]);

			BS = (int) strtol(argv[f+1], NULL, 10);

			if( BS <= 0 )
				usage(argv[0]);

			f++;
			BS_defined = 1;
			continue;
		}
		else if( strcmp(argv[f], "-r") == 0 )
		{
			// we want at least the integer and a matrix market file after
			if( f+2 >= argc )
				usage(argv[0]);

			restart = (int) strtol(argv[f+1], NULL, 10);

			if( restart < 0 )
				usage(argv[0]);

			f++;
			continue;
		}
		else if( strcmp(argv[f], "-sf") == 0 )
		{
			// we want at least the switch and a matrix market file after
			if( f+1 >= argc )
				usage(argv[0]);

			fault_strat = SINGLEFAULT;
			// (all strategies equivalent for 1 fault)
			
			continue;
		}
		else if( strcmp(argv[f], "-mf") == 0 )
		{
			// we want at least the strategy and a matrix market file after
			if( f+2 >= argc )
				usage(argv[0]);
			
			if( strcmp(argv[f+1], "global") == 0 )
				fault_strat = MULTFAULTS_GLOBAL;
			else if( strcmp(argv[f+1], "uncorrelated") == 0 )
				fault_strat = MULTFAULTS_UNCORRELATED;
			else if( strcmp(argv[f+1], "decorrelated") == 0 )
				fault_strat = MULTFAULTS_DECORRELATED;
			else
				usage(argv[0]);

			f++;
			continue;
		}
		
		// if it's not an option, it's a file
		else
		{
			FILE* input_file = fopen(argv[f], "r");
			if(input_file == NULL)
			{
				printf("Error : file %s not valid (check path/read permissions)\n", argv[f]);
				continue;
			}

			MM_typecode matcode;
			if (mm_read_banner(input_file, &matcode) != 0)
			{
				printf("Could not process Matrix Market banner of file %s.\n", argv[f]);
				exit(1);
			}

			if (mm_is_complex(matcode))
			{
				printf("Sorry, this application does not support "
						"Matrix Market type of file %s : [%s]\n", argv[f], mm_typecode_to_str(matcode));
				continue;
			}

			int m, n, nnz, symmetric = mm_is_symmetric(matcode);


			if( !mm_is_array(matcode) && ( mm_read_mtx_crd_size(input_file, &m, &n, &nnz) != 0 || m != n ) )
			{
				printf("Sorry, this application does not support the matrix in file %s\n", argv[f]);
				continue;
			}
			else if( mm_is_array(matcode) )
			{
				if( mm_read_mtx_array_size(input_file, &m, &n) != 0 || m != n )
				{
				printf("Sorry, this application does not support the matrix in file %s\n", argv[f]);
					continue;
				}

				nnz = m*n;
			}

			// DEBUG TO MODIFY PROBLEM
			// n = m = 16;

			// compute now we have n, and reset for next matrix
			if( BS_defined )
			{
				BS_defined = 0;
				if( blocks > 0 )
					BS = (n+blocks-1) / blocks;
			}
			else
				printf("WARNING : block size way be unadapted.\n");

			Matrix matrix;

			#ifdef MATRIX_DENSE // using dense matrices
				allocate_dense_matrix(n, m, &matrix);
				read_dense_Matrix(n, m, nnz, symmetric, &matrix, input_file);

				//compute_neighbourhoods_dense(&dA, BS);
				printf("matrix_format:DENSE ");

			#else // by default : using sparse matrices
				allocate_sparse_matrix(n, m, nnz * (1+symmetric), &matrix);
				read_sparse_Matrix(n, m, nnz, symmetric, &matrix, input_file);

				//compute_neighbourhoods_sparse(&dA, BS);
				printf("matrix_format:SPARSE ");

			#endif 

			fclose(input_file);

			// now show infos
			printf("File:%s problem_size:%d ", argv[f], n);

			if(symmetric)
				printf("matrix_symmetric:yes method:ConjugateGradient\n");
			else if( restart > 0 )
				printf("matrix_symmetric:no method:restarted_GMRES(%d)\n", restart);
			else
				printf("matrix_symmetric:no method:full_GMRES\n");

			char strat[40];
			name_strategy(fault_strat, strat);

			printf("lambda:%e block_size:%d strategy:%s\n", lambda, BS, strat);

			#if VERBOSE > FULL_VERBOSE
			print(&matrix);
			#endif
		
			// a few vectors for rhs of equation, solution and verification
			double *b, *x, *s;
			b = (double*)malloc( n * sizeof(double) );
			x = (double*)malloc( n * sizeof(double) );
			s = (double*)malloc( n * sizeof(double) );

			// generate random rhs to problem, and initialize first guess to 0
			double range = (double) 1;

			for(i=0; i<n; i++)
			{
				b[i] = ((double)rand() / (double)RAND_MAX ) * range - range/2;
				x[i] = 0.0;
			}

			// do some setup for the resilience part
			#ifdef PAPICOUNTERS
			setup_papi();
			#endif
			setup(n, lambda, 0.7);

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
			#ifdef PAPICOUNTERS
			unset_papi();
			#endif
			deallocate_matrix(&matrix);

			// do displays (verification, error)
			double err = 0, norm_b = scalar_product(n, b, b);
			for(i=0; i < n; i++)
			{
				double e_i = b[i] - s[i];
				err += e_i * e_i;
			}

			printf("Verification : euclidian distance to solution ||Ax-b||^2 = %e , ||Ax-b||/||b|| = %e\n", err, sqrt(err/norm_b));

			free(b);
			free(x);
			free(s);
		}

	return 0;
}

