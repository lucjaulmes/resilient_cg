#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <float.h>

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

#ifdef _OMPSS
	#include <nanos_omp.h>
#endif

// globals
int nb_blocks;
int *block_ends;

void set_blocks_sparse(Matrix *A, int *nb_blocks, const int fail_size)
{
	block_ends = (int*)malloc(*nb_blocks * sizeof(int));

	// compute block repartition now we have the matrix, arrange for block limits to be on fail block limits
	int i, pos = 0, next_stop = 0, ideal_bs = (A->nnz + *nb_blocks / 2) / *nb_blocks, inc_pos = fail_size / sizeof(double);
	for(i=0; i<*nb_blocks-1; i++)
	{
		next_stop += ideal_bs;

		while(pos + inc_pos <= A->n && A->r[pos + inc_pos] < next_stop)
			pos += inc_pos;

		if(pos + inc_pos <= A->n && A->r[pos + inc_pos] - next_stop < next_stop - A->r[pos])
			pos += inc_pos;

		if(pos >= A->n)
		{
			fprintf(stderr, "Error while making blocks : end of block %d/%d is %d, beyond size of matrix %d ;"
							" nb_blocks reduced to %d. You could try reducing -ps\n", i+1, *nb_blocks, pos, A->n, i+1);
			*nb_blocks=i+1;
			break;
		}

		set_block_end(i, pos);

		// force to increment by at least 1
		pos += inc_pos;
	}

	set_block_end(*nb_blocks -1, A->n);
}

int MAXIT = 0;

void usage(char* arg0)
{
	printf("Usage: %s [options] <matrix-market-filename> [, ...] \n"
			"Possible options are : \n"
			" ===  fault injection  === \n"
			"  -nf               Disabling faults simulation (default).\n"
			"  -l     lambda     Inject errors with lambda meaning MTBE in usec.\n"
			"  -nerr  N duration Inject N errors over a period of duration in usec.\n"
			"                    Note : the options -nf, -l and -nerr are mutually exclusive.\n"
			"  -mfs   strategy   Select an alternate (cf Agullo2013) strategy for multiple faults.\n "
			"                   'strategy' must be one of global, uncorrelated, decorrelated.\n"
			"                    Note : has no effect without errors. global is default.\n"
			" === run configuration === \n"
			"  -th    threads    Manually define number of threads.\n"
			"  -nb    blocks     Defines the number of blocks in which to divide operations ;\n"
			"                    their size will depend on the matrix' size.\n"
			"  -r     runs       number of times to run a matrix solving.\n"
			"  -cv    thres      Run until the error verifies ||b-Ax|| < thres * ||b|| (default 1e-10).\n"
			"  -maxit N          Run no more than N iterations (default no limit).\n"
			"  -seed  s          Initialize seed of each run with s. If 0 use different (random) seeds.\n"
			" === resilience method === \n"
			"  -ps    size       Defines page size (used on failure, in bytes, defaults to 4K).\n"
			"                    Must be a multiple of the system page size (and a power of 2).\n"
	#if CKPT == CKPT_TO_DISK
			"  -disk  /path/dir  Path to a directory on local disk for checkpointing (default $TMPDIR).\n"
			"  -ckpt             Prefix of the name of checkpoint files.\n"
	#endif
			"All options apply to every following input file. You may re-specify them for each file.\n\n", arg0);
	exit(1);
}

// we return how many parameters we consumed, -1 for error
int read_param(int argsleft, char* argv[], double *lambda, int *runs, int *threads UNUSED, int *blocks, long *fail_size, int *fault_strat,
			int *nerr, unsigned int *seed, double *cv_thres, char **checkpoint_path UNUSED, char **checkpoint_prefix UNUSED)
{
	if(strcmp(argv[0], "-r") == 0)
	{
		// we want at least the integer and a matrix market file after
		if(argsleft <= 2)
			return -1;

		*runs = (int) strtol(argv[1], NULL, 10);

		if(*runs < 0)
			return -1;

		return 2;
	}
	else if(strcmp(argv[0], "-maxit") == 0)
	{
		// we want at least the integer and a matrix market file after
		if(argsleft <= 2)
			return -1;

		MAXIT = (int) strtol(argv[1], NULL, 10);

		if(MAXIT <= 0)
			return -1;

		return 2;
	}
	else if(strcmp(argv[0], "-seed") == 0)
	{
		// we want at least the integer and a matrix market file after
		if(argsleft <= 2)
			return -1;

		*seed = (unsigned int) strtol(argv[1], NULL, 10);

		return 2;
	}
	else if(strcmp(argv[0], "-cv") == 0)
	{
		// we want at least the double and a matrix market file after
		if(argsleft <= 2)
			return -1;

		*cv_thres = strtod(argv[1], NULL);

		if(*cv_thres <= 1e-15)
			return -1;

		return 2;
	}
	else if(strcmp(argv[0], "-th") == 0)
	{
		// we want at least the integer and a matrix market file after
		if(argsleft <= 2)
			return -1;

		int th = (int) strtol(argv[1], NULL, 10);

		if(th <= 0)
			return -1;

		#ifdef _OMPSS
		nanos_omp_set_num_threads(th);
		*threads = th;
		#else
		if(th != 1)
			fprintf(stderr, "DO NOT DEFINE THREADS FOR THE SEQUENTIAL VERSION !\n");
		#endif

		return 2;
	}
	else if(strcmp(argv[0], "-nb") == 0)
	{
		// we want at least the integer and a matrix market file after
		if(argsleft <= 2)
			return -1;

		*blocks = (int) strtol(argv[1], NULL, 10);

		if(*blocks <= 0)
			return -1;

		return 2;
	}
	else if(strcmp(argv[0], "-ps") == 0)
	{
		// we want at least the integer and a matrix market file after
		if(argsleft <= 2)
			return -1;

		*fail_size = strtol(argv[1], NULL, 10);
		const long divisor = sysconf(_SC_PAGESIZE);

		if(*fail_size <= 0 || *fail_size % divisor || *fail_size ^ (1 << (ffs((int)*fail_size)-1)))
			return -1;

		return 2;
	}
	else if(strcmp(argv[0], "-l") == 0)
	{
		// we want at least the double and a matrix market file after
		if(argsleft <= 2)
			return -1;

		*nerr = 0;
		*lambda = strtod(argv[1], NULL);

		if(*lambda <= 0)
			return -1;

		return 2;
	}
	else if(strcmp(argv[0], "-nf") == 0)
	{
		// we want at least a matrix market file after the switch
		if(argsleft <= 1)
			return -1;

		*lambda = 0;
		*nerr   = 0;
		// (all strategies equivalent for 1 fault)

		return 1;
	}
	else if(strcmp(argv[0], "-mfs") == 0)
	{
		// we want at least the strategy and a matrix market file after
		if(argsleft <= 2)
			return -1;

		if(strcmp(argv[1], "global") == 0)
			*fault_strat = MULTFAULTS_GLOBAL;
		else if(strcmp(argv[1], "uncorrelated") == 0)
			*fault_strat = MULTFAULTS_UNCORRELATED;
		else if(strcmp(argv[1], "decorrelated") == 0)
			*fault_strat = MULTFAULTS_DECORRELATED;
		else
			return -1;

		return 2;
	}
	else if(strcmp(argv[0], "-nerr") == 0)
	{
		// we want at least the integer, duration and a matrix market file after
		if(argsleft <= 3)
			return -1;

		*nerr = (int) strtol(argv[1], NULL, 10);
		*lambda = (double) strtod(argv[2], NULL);

		if(*nerr <= 0 || *lambda <= 0)
			return -1;

		return 3;
	}
	#if CKPT == CKPT_TO_DISK
	else if(strcmp(argv[0], "-disk") == 0)
	{
		// we want at least the integer and a matrix market file after
		if(argsleft <= 2)
			return -1;

		struct stat file_infos;
		stat(argv[1], &file_infos);

		mode_t required_flags = S_IFDIR | S_IROTH | S_IWOTH;
		if((file_infos.st_mode & required_flags) == required_flags)
			return -1;

		*checkpoint_path = strdup(argv[1]);

		return 2;
	}
	else if(strcmp(argv[0], "-ckpt") == 0)
	{
		// we want at least the integer and a matrix market file after
		if(argsleft <= 2)
			return -1;


		*checkpoint_prefix = strdup(argv[1]);

		return 2;
	}
	#endif
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

	else if(!mm_is_array(matcode) && (mm_read_mtx_crd_size(input_file, m, n, nnz) != 0 || *m != *n))
		printf("Sorry, this application does not support the not-array matrix in file \"%s\"\n", filename);

	else if(mm_is_array(matcode) && (mm_read_mtx_array_size(input_file, m, n) != 0 || *m != *n))
		printf("Sorry, this application does not support the array matrix in file \"%s\"\n", filename);

	else if(!mm_is_symmetric(matcode))
		printf("Sorry, this application does not support the non-symmetric matrix in file \"%s\"\n", filename);

	else // hurray, no reasons to fail
	{
		if(*nnz == 0)
			*nnz = (*m) * (*n);
		*symmetric = mm_is_symmetric(matcode);
		if(MAXIT == 0)
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
	// no buffer on stdout so messages interleaved with stderr will be in right order
	setbuf(stdout, NULL);

	if(argc < 2)
		usage(argv[0]);

	int i, j, f, nb_read, runs = 1;

	#ifdef _OMPSS
	int nb_threads = nb_blocks = nanos_omp_get_num_threads();
	#else
	int nb_threads = nb_blocks = 1;
	#endif

	int fault_strat = MULTFAULTS_GLOBAL, nerr = 0;
	long fail_size;
	double lambda = 0, cv_thres = 1e-10;
	#if CKPT == CKPT_TO_DISK
	char *checkpoint_path = getenv("TMPDIR"), *checkpoint_prefix = "", ckpt[50];
	#else
	char *checkpoint_path = NULL, *checkpoint_prefix = NULL;
	#endif


	fail_size = sysconf(_SC_PAGESIZE); // default page size ?

	unsigned int seed = 1591613054 ;// time(NULL);

	// Iterate over parameters (usually open files)
	for(f=1; f<argc; f += nb_read)
	{
		nb_read = read_param(argc - f, &argv[f], &lambda, &runs, &nb_threads, &nb_blocks, &fail_size, &fault_strat, &nerr, &seed, &cv_thres, &checkpoint_path, &checkpoint_prefix);

		// error happened
		if(nb_read < 0)
			usage(argv[0]);

		// no parameters read : next param must be a matrix file. Read it (and consume parameter)
		else if(nb_read == 0)
		{
			nb_read = 1;
			int n, m, lines_in_file, symmetric;
			FILE* input_file = get_infos_matrix(argv[f], &n, &m, &lines_in_file, &symmetric);

			if(input_file == NULL)
				usage(argv[0]);

			// DEBUG TO MODIFY PROBLEM
			// n = m = 128;

			Matrix matrix;
			allocate_matrix(n, m, lines_in_file * (1 + symmetric), &matrix, fail_size);
			read_matrix(n, m, lines_in_file, symmetric, &matrix, input_file);

			set_blocks_sparse(&matrix, &nb_blocks, fail_size);

			fclose(input_file);

			// now show infos
			#ifndef DUE
			#define DUE 0
			#endif

			char header[500];
			const char * const due_names[] = {"none", "async", "in_path", "rollback", "lossy"};
			const char * const fault_strat_names[] = {"global", "uncorrelated", "decorrelated"};

			sprintf(header, "matrix_format:SPARSE executable:%s File:%s problem_size:%d nb_threads:%d nb_blocks:%d due:%s strategy:%s failure_size:%ld srand_seed:%u maxit:%d convergence_at:%e\n",
					argv[0], argv[f], n, nb_threads, nb_blocks, due_names[DUE], fault_strat_names[fault_strat-1], fail_size, seed, MAXIT, cv_thres);

			if(nerr)
				sprintf(strchr(header, '\n'), " inject_errors:%d inject_duration:%e\n", nerr, lambda);
			else
				sprintf(strchr(header, '\n'), " lambda:%e\n", lambda);

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
			#if VERBOSE >= SHOW_TOOMUCH
			print_matrix(stderr, &matrix);
			#endif

			populate_global(matrix.n, fail_size, fault_strat, nerr, lambda, ckpt);

			// if using fancy ways of measuring (e.g. extrae events)
			setup_measure();

			// a few vectors for rhs of equation, solution and verification
			double *b, *x, *s;
			b = (double*)aligned_calloc(fail_size, n * sizeof(double));
			x = (double*)aligned_calloc(fail_size, n * sizeof(double));
			s = (double*)aligned_calloc(fail_size, n * sizeof(double));

			// interesting stuff is here
			for(j=0;j<runs;j++)
			{
				// seed = 0 -> random : time for randomness, +j to get different seeds even if solving < 1s
				unsigned int real_seed = seed == 0 ? time(NULL) + j : seed;
				if(runs > 1)
					printf("run:%d seed:%u\n", j, real_seed);

				srand(real_seed);

				// generate random rhs to problem
				double range = (double) 1;

				for(i=0; i<n; i++)
				{
					b[i] = ((double)rand() / (double)RAND_MAX ) * range - range/2;
					x[i] = 0.0;
				}

				solve_cg(&matrix, b, x, cv_thres);

				// compute verification
				mult(&matrix, x, s);

				// do displays (verification, error)
				double err = 0, norm_b = scalar_product(n, b, b);
				for(i=0; i < n; i++)
				{
					double e_i = b[i] - s[i];
					err += e_i * e_i;
				}

				printf("Verification : euclidian distance to solution ||Ax-b||^2 = %e , ||Ax-b||/||b|| = %e\n", err, sqrt(err/norm_b));
			}

			// deallocate everything we have allocated for several solvings
			unset_measure();

			deallocate_matrix(&matrix);
			free(b);
			free(x);
			free(s);
			free(block_ends);
		}
	}

	return 0;
}

