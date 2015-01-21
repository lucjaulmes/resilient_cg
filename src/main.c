#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <float.h>
#include <signal.h>
#include <assert.h>

#include "mpi.h"

#include "global.h"

#if CKPT == CKPT_TO_DISK
#include <sys/stat.h>
#endif

#include "matrix.h"
#include "cg.h"
#include "mmio.h"
#include "failinfo.h"
#include "recover.h"
#include "debug.h"
#include "counters.h"
#include "backtrace.h"

#ifdef _OMPSS
	#include <nanos_omp.h>
#endif

// globals
int nb_blocks;
int *block_bounds;
int mpi_rank = 0, *mpi_zonestart, *mpi_zonesize;

void set_blocks_sparse(Matrix *A, int nb_blocks, const int fail_size, const int mpi_rank, const int mpi_size)
{
	block_bounds  = (int*)malloc( (nb_blocks+1) * sizeof(int));
	mpi_zonestart = (int*)malloc( (  mpi_size  ) * sizeof(int));
	mpi_zonesize  = (int*)malloc( (  mpi_size  ) * sizeof(int));
	int mpiworld_blocks = (nb_blocks) * mpi_size;

	printf("mpi rank is %d out of %d : with %d blocks per rank, we have %d total blocks\n", mpi_rank, mpi_size, nb_blocks, mpiworld_blocks);

	// compute block repartition now we have the matrix, arrange for block limits to be on fail block limits
	int i, r, b, pos = 0, next_stop = 0, ideal_bs, inc_pos;

	ideal_bs = (A->nnz + mpiworld_blocks / 2) / mpiworld_blocks;
	inc_pos = fail_size / sizeof(double);

	for(r=0, i=0; r<mpi_size; r++)
	{
		mpi_zonestart[r] = pos;
		if( r == mpi_rank )
			block_bounds[0] = pos;

		for(b=0; b<nb_blocks; b++, i++)
		{
			next_stop += ideal_bs;

			// force to increment by at least 1
			pos += inc_pos;

			// increment until we are highest possible below next_stop
			while( pos + inc_pos <= A->n && A->r[pos + inc_pos] < next_stop )
				pos += inc_pos;

			// choose which of just below or just above next_stop is closest
			if( pos + inc_pos <= A->n && A->r[pos + inc_pos] - next_stop < -(A->r[pos] - next_stop) )
				pos += inc_pos;

			if(pos >= A->n && i+1 != mpiworld_blocks)
			{
				fprintf(stderr, "Error while making blocks : end of block %d/%d is %d, beyond size of matrix %d."
								" Try reducing -ps\n", i+1, nb_blocks, pos, A->n);

				exit(EXIT_FAILURE);
			}

			if( r == mpi_rank )
				block_bounds[b+1] = pos;
		}

		mpi_zonesize[r] = pos - mpi_zonestart[r];
	}

	if( mpi_rank == mpi_size-1 )
		block_bounds[ nb_blocks ] = A->n;
}
 
int MAXIT = 0;

void usage(const char *arg0, const char *arg_err)
{
	if( arg_err != NULL )
		printf("Error near (or after) argument \"%s\"\n\n", arg_err);
	
	printf("Usage: %s [options] [<matrix-market-filename>|-synth name param] [, ...]\n"
			" === Matrix === \n"
			"  Either provide a path to a symmetric positive definite matrix in Matrix Market format\n"
			"  or provide the -synth option for a synthetic matrix. Arguments are name param pairs :\n"
			"    Poisson3D  p n  3D Poisson's equation using finite differences, matrix size n^3\n"
			"                    with a p-points stencil : p one of 7, 19, 27 (TODO : 9 BCC, 13 FCC)\n"
			" ===  fault injection options === \n"
			"  -nf               Disabling faults simulation (default).\n"
			"  -l     lambda     Inject errors with lambda meaning MTBE in usec.\n"
			"  -nerr  N duration Inject N errors over a period of duration in usec.\n"
			"                    Note : the options -nf, -l and -nerr are mutually exclusive.\n"
			"  -mfs   strategy   Select an alternate (cf Agullo2013) strategy for multiple faults.\n "
			"                   'strategy' must be one of global, uncorrelated, decorrelated.\n"
			"                    Note : has no effect without errors. global is default.\n"
			" === run configuration options === \n"
			"  -th    threads    Manually define number of threads PER RANK.\n"
			"  -nb    blocks     Defines the number of blocks PER RANK in which to divide operations ;\n"
			"                    their size will depend on the matrix' size.\n"
			"  -r     runs       number of times to run a matrix solving.\n"
			"  -cv    thres      Run until the error verifies ||b-Ax|| < thres * ||b|| (default 1e-10).\n"
			"  -maxit N          Run no more than N iterations (default no limit).\n"
			"  -seed  s          Initialize seed of each run with s. If 0 use different (random) seeds.\n"
			" === resilience method options === \n"
			"  -ps    size       Defines page size (used on failure, in bytes, defaults to 4K).\n"
			"                    Must be a multiple of the system page size (and a power of 2).\n"
	#if CKPT == CKPT_TO_DISK
			"  -disk  /path/dir  Path to a directory on local disk for checkpointing (default $TMPDIR).\n"
			"  -ckpt             Prefix of the name of checkpoint files.\n"
	#endif
	#if SDC
			"  -sdc    tol       Consider SDC occured if detector yields a value > tol (default 1e-12).\n"
	#endif
			"All options apply to every following input file. You may re-specify them for each file.\n\n", arg0);
	exit(1);
}

// we return how many parameters we consumed, -1 for error
int read_param(int argsleft, char* argv[], double *lambda, int *runs, int *threads UNUSED, int *blocks, long *fail_size, int *fault_strat, int *nerr, unsigned int *seed, 
			double *cv_thres, double *err_thres, char **checkpoint_path UNUSED, char **checkpoint_prefix UNUSED, matrix_type *matsource, int *stencil_points, int *matrix_size)
{
	if( strcmp(argv[0], "-r") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		*runs = (int) strtol(argv[1], NULL, 10);

		if( *runs < 0 )
			return -1;

		return 2;
	}
	else if( strcmp(argv[0], "-maxit") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		MAXIT = (int) strtol(argv[1], NULL, 10);

		if( MAXIT <= 0 )
			return -1;

		return 2;
	}
	else if( strcmp(argv[0], "-seed") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		*seed = (unsigned int) strtol(argv[1], NULL, 10);

		return 2;
	}
	else if( strcmp(argv[0], "-cv") == 0 )
	{
		// we want at least the double and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		*cv_thres = strtod(argv[1], NULL);

		if( *cv_thres <= 1e-15 )
			return -1;

		return 2;
	}
	else if( strcmp(argv[0], "-th") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		int th = (int) strtol(argv[1], NULL, 10);

		if( th <= 0 )
			return -1;

		#ifdef _OMPSS
		nanos_omp_set_num_threads(th);
		*threads = 1;
		#else
		if( th != 1 )
			fprintf(stderr, "DO NOT DEFINE THREADS FOR THE SEQUENTIAL VERSION !\n");
		#endif

		return 2;
	}
	else if( strcmp(argv[0], "-nb") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		*blocks = (int) strtol(argv[1], NULL, 10);

		if( *blocks <= 0 )
			return -1;

		return 2;
	}
	else if( strcmp(argv[0], "-ps") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		*fail_size = strtol(argv[1], NULL, 10);
		const long divisor = sysconf(_SC_PAGESIZE);

		if( *fail_size <= 0 || *fail_size % divisor || *fail_size ^ (1 << (ffs((int)*fail_size)-1)) )
			return -1;

		return 2;
	}
	else if( strcmp(argv[0], "-l") == 0 )
	{
		// we want at least the double and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		*nerr = 0;
		*lambda = strtod(argv[1], NULL);

		if( *lambda <= 0 )
			return -1;

		return 2;
	}
	else if( strcmp(argv[0], "-nf") == 0 )
	{
		// we want at least a matrix market file after the switch
		if( argsleft <= 1 )
			return -1;

		*lambda = 0;
		*nerr   = 0;
		// (all strategies equivalent for 1 fault)

		return 1;
	}
	else if( strcmp(argv[0], "-mfs") == 0 )
	{
		// we want at least the strategy and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		if( strcmp(argv[1], "global") == 0 )
			*fault_strat = MULTFAULTS_GLOBAL;
		else if( strcmp(argv[1], "uncorrelated") == 0 )
			*fault_strat = MULTFAULTS_UNCORRELATED;
		else if( strcmp(argv[1], "decorrelated") == 0 )
			*fault_strat = MULTFAULTS_DECORRELATED;
		else
			return -1;

		return 2;
	}
	else if( strcmp(argv[0], "-nerr") == 0 )
	{
		// we want at least the integer, duration and a matrix market file after
		if( argsleft <= 3 )
			return -1;

		*nerr = (int) strtol(argv[1], NULL, 10);
		*lambda = (double) strtod(argv[2], NULL);

		if( *nerr <= 0 || *lambda <= 0 )
			return -1;

		return 3;
	}
	else if( strcmp(argv[0], "-sdc") == 0 )
	{
		// we want at least the double and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		*err_thres = strtod(argv[1], NULL);

		#if SDC == 0
		if( *err_thres <= DBL_EPSILON )
		#else
		if( *err_thres <= 0.0 )
		#endif
			return -1;

		return 2;
	}
	#if CKPT == CKPT_TO_DISK
	else if( strcmp(argv[0], "-disk") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			return -1;

		struct stat file_infos;
		stat(argv[1], &file_infos);
	
		mode_t required_flags = S_IFDIR | S_IROTH | S_IWOTH;
		if( (file_infos.st_mode & required_flags) == required_flags )
			return -1;

		*checkpoint_path = strdup(argv[1]);

		return 2;
	}
	else if( strcmp(argv[0], "-ckpt") == 0 )
	{
		// we want at least the integer and a matrix market file after
		if( argsleft <= 2 )
			return -1;


		*checkpoint_prefix = strdup(argv[1]);

		return 2;
	}
	#endif
	else if( strcmp(argv[0], "-synth") == 0 )
	{
		// we want at least the name and size and points parameter after
		if( argsleft <= 3 )
			return -1;
		
		if( strcmp(argv[1], "Poisson3D") == 0 )
		{
			*matsource = POISSON3D;

			*stencil_points = (int)strtol(argv[2], NULL, 10);

			if( *stencil_points != 7 && *stencil_points != 19 && *stencil_points != 27 )
				return -1;
		}
		else
			return -1; // unrecognized

		*matrix_size = (int)strtol(argv[3], NULL, 10);

		if( *matrix_size <= 0 )
			return -1;

		return 4;
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
		printf("Error : file \"%s\" not valid (check path/read permissions)\n", filename);
		return NULL;
	}

	else if (mm_read_banner(input_file, &matcode) != 0)
		printf("Could not process Matrix Market banner of file \"%s\".\n", filename);

	else if (mm_is_complex(matcode))
		printf("Sorry, this application does not support Matrix Market type of file \"%s\" : [%s]\n", 
			filename, mm_typecode_to_str(matcode));

	else if( !mm_is_array(matcode) && (mm_read_mtx_crd_size(input_file, m, n, nnz) != 0 || *m != *n) )
		printf("Sorry, this application does not support the not-array matrix in file \"%s\"\n", filename);

	else if( mm_is_array(matcode) && (mm_read_mtx_array_size(input_file, m, n) != 0 || *m != *n) )
		printf("Sorry, this application does not support the array matrix in file \"%s\"\n", filename);

	else if( !mm_is_symmetric(matcode) )
		printf("Sorry, this application does not support the non-symmetric matrix in file \"%s\"\n", filename);

	else // hurray, no reasons to fail
	{
		if( *nnz == 0 )
			*nnz = (*m) * (*n);
		*symmetric = mm_is_symmetric(matcode);
		if( MAXIT == 0 )
			MAXIT = 10 * (*n);
		return input_file;
	}

	// if we're here we failed at some point but opened the file
	fclose(input_file);
	return NULL;
}

// main function, where we parse arguments, read files, setup stuff and start the recoveries
int main(int argc, char* argv[])
{
	if(argc < 2)
		usage(argv[0], NULL);

	int i, j, f, nb_read, runs = 1;

	#ifdef _OMPSS
	int nb_threads = nb_blocks = nanos_omp_get_num_threads();
	#else
	int nb_threads = nb_blocks = 1;
	#endif

	int fault_strat = MULTFAULTS_GLOBAL, nerr = 0, stencil_points, matrix_size;
	matrix_type matsource = FROM_FILE;
	long fail_size;
	double lambda = 0, cv_thres = 1e-10, err_thres = 1e-12;
	#if CKPT == CKPT_TO_DISK
	char *checkpoint_path = getenv("TMPDIR"), *checkpoint_prefix = "", ckpt[50];
	#else
	char *checkpoint_path = NULL, *checkpoint_prefix = NULL;
	#endif


	fail_size = sysconf(_SC_PAGESIZE); // default page size ?
	unsigned int seed = 1591613054 ;// time(NULL);

	int mpi_args = 0, mpi_thread_level = MPI_THREAD_FUNNELED/*SERIALIZED*/, mpi_size = 1;
	MPI_Init_thread(&mpi_args, NULL, mpi_thread_level, &mpi_thread_level);
	//assert( mpi_thread_level == MPI_THREAD_SERIALIZED );
	printf("Asked for mpi thread level %d, got mpi_thread_level:%d\n", MPI_THREAD_FUNNELED, mpi_thread_level);

	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// Iterate over parameters (usually open files)
	for(f=1; f<argc; f += nb_read )
	{
		nb_read = read_param(argc - f, &argv[f], &lambda, &runs, &nb_threads, &nb_blocks, &fail_size, &fault_strat, &nerr, &seed,
							&cv_thres, &err_thres, &checkpoint_path, &checkpoint_prefix, &matsource, &stencil_points, &matrix_size);

		// error happened
		if( nb_read < 0 )
			usage(argv[0], argv[f]);

		// no parameters read : next param must be a matrix file. Read it (and consume parameter)
		else if( nb_read == 0 || matsource != FROM_FILE)
		{
			int n;
			Matrix matrix;
			char mat_name[200];

			if( matsource == FROM_FILE )
			{
				nb_read = 1;
				int m, lines_in_file, symmetric;
				FILE* input_file = get_infos_matrix(argv[f], &n, &m, &lines_in_file, &symmetric);

				if( input_file == NULL )
					usage(argv[0], NULL);

				// DEBUG TO MODIFY PROBLEM
				// n = m = 128;

				allocate_matrix(n, m, lines_in_file * (1 + symmetric), &matrix, fail_size);
				read_matrix(n, m, lines_in_file, symmetric, &matrix, input_file);

				set_blocks_sparse(&matrix, nb_blocks, fail_size, mpi_rank, mpi_size);

				fclose(input_file);

				strcpy(mat_name, argv[f]);
			}
			else // if matsource == POISSON3D
			{
				matsource = FROM_FILE;

				// here all block etc. repartition is static, so we can load balance in advance
				n = mpi_size * matrix_size * matrix_size;

				// number of mem. pages per vector on one side, total number of threads on the other
				if( n / (fail_size/sizeof(double)) % (mpi_size * nb_blocks) != 0 )
					usage(argv[0], "Error : size of Poisson3D matrix incompatible with alignment of memory page per block\n");

				int rows_per_rank = n / mpi_size, rows_per_block = rows_per_rank / nb_blocks;
				long nnz_here = stencil_points * rows_per_rank;

				block_bounds  = (int*)malloc( (nb_blocks+1) * sizeof(int));
				mpi_zonestart = (int*)malloc( (  mpi_size  ) * sizeof(int));
				mpi_zonesize  = (int*)malloc( (  mpi_size  ) * sizeof(int));

				mpi_zonestart[0] = 0;
				for(i=1; i<mpi_size; i++)
				{
					mpi_zonestart[i] = mpi_zonestart[i-1] + rows_per_rank;
					mpi_zonesize[i-1] = rows_per_rank;
				}
				mpi_zonesize[i-1] = rows_per_rank;

				block_bounds[0] = mpi_zonestart[mpi_rank];
				for(i=1; i<nb_blocks+1; i++)
					block_bounds[i] = block_bounds[i-1] + rows_per_block;


				allocate_matrix(n, n, nnz_here, &matrix, fail_size);
				generate_Poisson3D(&matrix, matrix_size, stencil_points, mpi_zonestart[mpi_rank], mpi_zonestart[mpi_rank] + rows_per_rank);

				sprintf(mat_name, "Poisson3D_%dpt", stencil_points);
			}

			#if VERBOSE >= SHOW_TOOMUCH
			print_matrix(stderr, &matrix);
			#endif

			// now show infos
			#ifndef DUE
			#define DUE 0
			#endif

			char header[500];
			const char * const due_names[] = {"none", "async", "in_path", "rollback", "lossy"};
			const char * const fault_strat_names[] = {"global", "uncorrelated", "decorrelated"};

			sprintf(header, "matrix_format:SPARSE executable:%s File:%s problem_size:%d nb_threads:%d nb_blocks:%d due:%s strategy:%s failure_size:%ld srand_seed:%u maxit:%d convergence_at:%e\n",
					argv[0], mat_name, n, nb_threads, nb_blocks, due_names[DUE], fault_strat_names[fault_strat-1], fail_size, seed, MAXIT, cv_thres);

			if( nerr )
				sprintf(strchr(header, '\n'), " inject_errors:%d inject_duration:%e\n", nerr, lambda);
			else
				sprintf(strchr(header, '\n'), " lambda:%e\n", lambda);

			#if SDC
			const char * const sdc_names[] = {"none", "alpha", "gradient", "orthogonal"};
			sprintf(strchr(header, '\n'), " sdc:%s sdc_freq:%d sdc_thres:%e\n", sdc_names[SDC], CHECK_SDC_FREQ, err_thres);
			#endif

			#if CKPT
			const char * const ckpt_names[] = {"none", "in_memory", "to_disk"};
			sprintf(strchr(header, '\n'), " ckpt:%s ckpt_freq:%d\n", ckpt_names[CKPT], CHECKPOINT_FREQ);
			#endif

			// set some parameters that we don't want to pass through solve_cg
			#if CKPT == CKPT_TO_DISK
			sprintf(ckpt, "%s/%s", checkpoint_path, checkpoint_prefix);
			sprintf(strchr(header, '\n'), " ckpt_path:%s\n", ckpt);
			#else
			const char *ckpt = NULL;
			#endif

			printf(header);

			populate_global(matrix.n, fail_size, fault_strat, nerr, lambda, ckpt);

			// if using fancy ways of measuring (e.g. extrae events)
			setup_measure();
		
			// a few vectors for rhs of equation, solution and verification
			double *b, *x, *s;
			b = (double*)aligned_calloc( fail_size, n * sizeof(double));
			x = (double*)aligned_calloc( fail_size, n * sizeof(double));
			s = (double*)aligned_calloc( fail_size, n * sizeof(double));

			// interesting stuff is here
			for(j=0;j<runs;j++)
			{
				// seed = 0 -> random : time for randomness, +j to get different seeds even if solving < 1s
				unsigned int real_seed = seed == 0 ? time(NULL) + j : seed;
				if( runs > 1 )
					printf("run:%d seed:%u ", j, real_seed);

				srand(real_seed);

				// generate random rhs to problem
				double range = (double) 1;

				for(i=mpi_zonestart[mpi_rank]; i<mpi_zonestart[mpi_rank]+mpi_zonesize[mpi_rank]; i++)
				{
					b[i] = ((double)rand() / (double)RAND_MAX ) * range - range/2;
					x[i] = 0.0;
				}

				solve_cg(&matrix, b, x, cv_thres, err_thres);

				// get all x, for final computation of error
				MPI_Allgatherv(MPI_IN_PLACE, 0/*ignored*/, MPI_DOUBLE, x, mpi_zonesize, mpi_zonestart, MPI_DOUBLE, MPI_COMM_WORLD);

				// compute verification
				mult(&matrix, x, s);

				// do displays (verification, error)
				double t, err = 0, norm_b = norm(mpi_zonesize[mpi_rank], b + mpi_zonestart[mpi_rank]);
				for(i=mpi_zonestart[mpi_rank]; i<mpi_zonestart[mpi_rank]+mpi_zonesize[mpi_rank]; i++)
				{
					double e_i = b[i] - s[i];
					err += e_i * e_i;
				}
				t = err;
				MPI_Allreduce(&t, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

				printf("Verification : euclidian distance to solution ||Ax-b||^2 = %e , ||Ax-b||/||b|| = %e\n", err, sqrt(err/norm_b));
			}

			// deallocate everything we have allocated for several solvings
			unset_measure();

			deallocate_matrix(&matrix);
			free(b);
			free(x);
			free(s);
			free(block_bounds);
			free(mpi_zonestart);
			free(mpi_zonesize);
		}
	}

	MPI_Finalize();

	return 0;
}

