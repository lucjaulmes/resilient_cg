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

// pointers to functions
MultFunction mult;
RhsFunction get_rhs;
SubmatrixFunction get_submatrix;

char fault_strat;

// useful vector functions
double scalar_product( const int n, const double *v, const double *w )
{
	int i;
	double r = 0;

	for(i=0; i<n; i++)
		r += v[i] * w[i];

	return r;
}

struct timeval start, stop;
void start_measure()
{
	gettimeofday( &start, NULL );
}

double stop_measure()
{
	gettimeofday( &stop, NULL );
	return (1e6 * (stop.tv_sec - start.tv_sec)) + stop.tv_usec - start.tv_usec;
}

// some self-explanatory text functions
void usage(char* arg0)
{
	printf("Usage: %s [options] <matrix-market-filename> [, ...] \n"
			"Possible options are : \n"
			"  -l lambda		 Number (double), meaning 1/mtbf in usec.\n"
			"  -sf		       Forcing faults to happen no more than one at a time (default).\n"
			"  -mf strategy	  Enabling multiple faults to happen.\n "
			"		           'strategy' must be one of global, uncorrelated, decorrelated.\n"
			"  -bs blocksize	 size of the blocks in block-row operations ;\n"
			"		            also size of lost data on failure.\n"
			"  -r restart		number of steps for the restarted gmres.\n"
			"		           0 means standard gmres, without restarting (default).\n\n"
			"Options apply to all following input files. You may re-specify them for each file.", arg0);
	exit(1);
}

void name_strategy(const char n, char* name)
{
	if( n == SINGLEFAULT )
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
	//srand(time(NULL));
	srand(0);

	double lambda = 500;
	int block_size = 8, restart = 0;
	fault_strat = SINGLEFAULT;

	// Iterate over parameters (usually open files)
	for(f=1; f<argc; f++)
		if( strcmp(argv[f], "-l") == 0 )
		{
			// we want at least the double and a matrix market file after
			if( f+2 >= argc )
				usage(argv[0]);

			lambda = strtod(argv[f+1], NULL);

			if( lambda == 0 )
				usage(argv[0]);

			f++;
			continue;
		}
		else if( strcmp(argv[f], "-bs") == 0 )
		{
			// we want at least the integer and a matrix market file after
			if( f+2 >= argc )
				usage(argv[0]);

			block_size = (int) strtol(argv[f+1], NULL, 10);

			if( block_size == 0 )
				usage(argv[0]);

			f++;
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
			printf("\nFile:%s ", argv[f]);
			FILE* input_file = fopen(argv[f], "r");
			if(input_file == NULL)
			{
				printf("Error : file %s not valid (check path/read permissions)\n", argv[f]);
				continue;
			}

			MM_typecode matcode;
			if (mm_read_banner(input_file, &matcode) != 0)
			{
				printf("Could not process Matrix Market banner.\n");
				exit(1);
			}

			if (mm_is_complex(matcode))
			{
				printf("Sorry, this application does not support ");
				printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
				continue;
			}

			int m, n, nnz, symmetric = mm_is_symmetric(matcode);

			
			if( !mm_is_array(matcode) && ( mm_read_mtx_crd_size(input_file, &m, &n, &nnz) != 0 || m != n ) )
			{
				printf("Sorry, this application does not support this matrix");
				continue;
			}
			else if( mm_is_array(matcode) )
			{
				if( mm_read_mtx_array_size(input_file, &m, &n) != 0 || m != n )
				{
					printf("Sorry, this application does not support this matrix");
					continue;
				}

				nnz = m*n;
			}

			// DEBUG TO MODIFY PROBLEM
			// n = m = 15;
			//mm_set_dense(&matcode);

			printf("problem_size:%d ", n);

			// verifications are done and we've got the size of the matrix
			DenseMatrix dA;

			// read the matrix
			// no ordering implied by MM format. No choice but to allocate a n*n space
			// and then to transform to CSR format for sparse matrices.
			allocate_dense_matrix(n, m, &dA);
			read_dense_Matrix(n, m, nnz, symmetric, &dA, input_file);

			fclose(input_file);


			// a few vectors for rhs of equation, solution and verification
			double b[n], x[n], s[n];
			void *matrix;
			SparseMatrix sA;

			// generate random rhs to problem, and initialize first guess to 0
			double range = (double) 1;

			for(i=0; i<n; i++)
			{
				b[i] = ((double)rand() / (double)RAND_MAX ) * range - range/2;
				x[i] = 0.0;
			}

			// do some setup for the resilience part
			setup(n, block_size, lambda, 0.7, fault_strat);


			if(mm_is_sparse(matcode))
			{
				allocate_sparse_matrix(n, m, nnz * (1+symmetric), &sA);

				dense_to_sparse_Matrix(&dA, &sA);
				deallocate_dense_matrix(&dA);

				matrix = &sA;
				mult = &mult_sparse;
				get_rhs = &rhs_sparse;
				get_submatrix = &submatrix_sparse_to_dense;

				compute_neighbourhoods_sparse(&sA, block_size);

				printf("matrix_format:SPARSE ");
			}
			else
			{
				matrix = &dA;
				mult = &mult_dense;
				get_rhs = &rhs_dense;
				get_submatrix = &submatrix_dense;

				compute_neighbourhoods_dense(&dA, block_size);

				printf("matrix_format:DENSE ");
			}

			if(symmetric)
				printf("matrix_symmetric:yes method:ConjugateGradient\n");
			else if( restart > 0 )
				printf("matrix_symmetric:no method:restarted_GMRES(%d)\n", restart);
			else
				printf("matrix_symmetric:no method:full_GMRES\n");

			char strat[40];
			name_strategy(fault_strat, strat);

			printf("lambda:%e block_size:%d strategy:%s\n", lambda, block_size, strat);

			// if symmetric, solve with conjugate gradient method
			if(symmetric)
				solve_cg(n, matrix, b, x, 1e-10 );
			//otherwise, gmres
			else
				solve_gmres(n, matrix, b, x, 1e-10 , restart);

			// compute verification
			mult(matrix, x, s);

			// deallocate everything we have allocated for this solving
			unset();
			if(mm_is_sparse(matcode))
				deallocate_sparse_matrix(&sA);
			else
				deallocate_dense_matrix(&dA);

			// do displays (solution, error)
			log_err("\nsolution : \n\n(  ");

			for(i=0; i < n; i++)
				log_err("%.2e\t", x[i]);
			log_err(")\n");

			double err = 0, norm_b = scalar_product(n, b, b);
			for(i=0; i < n; i++)
			{
				double e_i = b[i] - s[i];
				err += e_i * e_i;
			}

			printf("Verification : euclidian distance to solution ||Ax-b||^2 = %e , ||Ax-b||/||b|| = %e\n", err, sqrt(err/norm_b));
		}

	return 0;
}

