#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <signal.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>
#include <limits.h>
#include <err.h>

#include "global.h"
#include "debug.h"
#include "cg.h"
#include "backtrace.h"

const char * const mask_names[] = { "0 ",
	"X ", "Ax", "G ", "P4", "P5", "Ap", "7 ", "8 ",
	"Sx", "10", "Sg", "Sp", "13", "Tp", "15", "16",
	"Ng", "Np", "19", "RC", "CP", "22", "23", "24",
	"25", "26", "27", "28", "Fg", "Fp", "Sh" };

#include "failinfo.h"

error_sim_data sim_err;
analyze_err errinfo;

// these are used to communicate between a thread and its tasks and vice versa, but not between threads
// N.B this is still __thread and not _Thread_local until mcc supports it : https://pm.bsc.es/projects/mcxx/ticket/404
__thread sig_atomic_t out_vect = 0, exception_happened = 0;

// from x a uniform distribution between 0 and 1, the weibull distribution
// is given by lambda * ( -ln( 1 - x ) )^(1/k)
double weibull(const double lambda, const double k, const double x)
{
	double y, inv_k = 1 / k;
	y = - log1p( - x ); // - log ( 1 - x )
	y = pow(y, inv_k);
	y *= lambda; // where lambda ~ mean time between faults

	return y;
}

// lambda is as in weibull (so inverse to usual in exp) ~ mtbf
// (i.e. scale parameter, not rate)
// so if x uniform between 0 and 1, return - lambda * log ( 1 - x )
double exponential(const double lambda, const double x)
{
	double y = - log1p( - x ); // - log ( 1 - x )
	y *= lambda;

	return y;
}

void populate_global(const int n, const int fail_size_bytes, const int fault_strat, const int nerr, const double lambda, const int checkpoint_freq UNUSED, const char *checkpoint_path UNUSED)
{
	const int fail_size = fail_size_bytes / sizeof(double);
	errinfo = (analyze_err){ .failblock_size = fail_size, .log2fbs = ffs(fail_size)-1, .nb_failblocks = (n + fail_size -1) / fail_size, .fault_strat = fault_strat,
		#if CKPT
		.ckpt_freq = checkpoint_freq,
		#endif
		#if CKPT == CKPT_TO_DISK
		.ckpt = checkpoint_path
		#endif
	};

	sim_err = (error_sim_data){ .lambda = lambda, .nerr = nerr, .info = &errinfo };
}

void setup_resilience(const Matrix *A UNUSED, const int nb, magic_pointers *mp)
{
	// various allocations

	#if DUE
	// neighbourhood stuff in errinfo
	errinfo.neighbours = (Matrix*)calloc( 1, sizeof(Matrix) );
	if (errinfo.neighbours == NULL)
		err(1, "Failed to allocate errinfo.neighbours");
	// don't want A->v so we allocate manually
	errinfo.neighbours->nnz = errinfo.nb_failblocks * errinfo.nb_failblocks;
	errinfo.neighbours->n = errinfo.neighbours->m = errinfo.nb_failblocks;
	errinfo.neighbours->r = (int*)calloc( (errinfo.nb_failblocks+1), sizeof(int) );
	errinfo.neighbours->c = (int*)calloc( errinfo.nb_failblocks * errinfo.nb_failblocks, sizeof(int) );
	errinfo.neighbours->v = NULL;
	if (errinfo.neighbours->r == NULL || errinfo.neighbours->c == NULL)
		err(1, "Failed to allocate errinfo.neighbours");

	compute_neighbourhoods(A, errinfo.failblock_size, errinfo.neighbours);

	// now for storing infos about errors
	errinfo.skipped_blocks = (int*)calloc( errinfo.nb_failblocks, sizeof(int) );
	if (errinfo.skipped_blocks == NULL)
		err(1, "Failed to allocate errinfo.skipped_blocks");
	#endif

	#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
	mp->shared_page_reductions = (double*)calloc(2 * nb_blocks, sizeof(double));
	if (mp->shared_page_reductions == NULL)
		err(1, "Failed to allocate mp->shared_page_reductions");
	int b;
	for (b = 0; b < nb_blocks; b++)
		if (get_block_end(b) & (errinfo.failblock_size - 1))
			errinfo.skipped_blocks[get_block_end(b) >> errinfo.log2fbs] |= SHARED_BLOCK;
	#endif

	#if CKPT == CKPT_TO_DISK
	mp->ckpt_data->checkpoint_path = errinfo.ckpt;
	#endif

	// now using the variable number of args set the pointers in errinfo.data for bit flipping / finding errors
	errinfo.nb_data = nb;
	errinfo.data = (double **) calloc(nb, sizeof(double *));
	if (errinfo.data == NULL)
		err(1, "Failed to allocate errinfo.data");

	#define X(constant, name) errinfo.data[constant-1] = mp->name;
	ASSOC_CONST_MP
	#undef X

	errinfo.in_recovery_errors = 0;
	errinfo.errors = 0;
	errinfo.skips = 0;
	sim_err.inj = 0;

	// finally set the handler for signals that will simulate (SIGSEGV) or report real errors (SIGBUS)
	struct sigaction sigact;
	sigact.sa_sigaction = resilience_sighandler;
	sigact.sa_flags = SA_SIGINFO;
	sigemptyset(&sigact.sa_mask);

	if( sigaction(SIGBUS, &sigact, NULL) !=  0)
		fprintf(stderr, "error setting signal handler for %d (%s)\n", SIGBUS, strsignal(SIGBUS));
	if( sigaction(SIGSEGV, &sigact, NULL) !=  0)
		fprintf(stderr, "error setting signal handler for %d (%s)\n", SIGSEGV, strsignal(SIGSEGV));

	// start semaphore locked : released in release_error_injection
	sem_init(&sim_err.start_sim, 0, 0);

	// if simulating faults, create thread to do so
	if( sim_err.lambda != 0 )
		pthread_create(&sim_err.th, NULL, &simulate_failures, (void*)&sim_err);
}

void start_error_injection()
{
	sem_post(&sim_err.start_sim);
}

void unset_resilience(magic_pointers *mp UNUSED)
{
	if( sim_err.lambda != 0 && sim_err.th )
	{
		pthread_cancel(sim_err.th);
		pthread_join(sim_err.th, NULL);
	}

	sem_destroy(&sim_err.start_sim);

	// undo all potentially undetected but already injected errors in memory from previous runs
	int i;
	const intptr_t vect_size = errinfo.nb_failblocks * (sizeof(double) << get_log2_failblock_size());
	for(i=0; i<errinfo.nb_data; i++)
		mprotect(errinfo.data[i], vect_size, PROT_READ | PROT_WRITE);

	// now stop handling errors
	struct sigaction sigact;
	sigset_t empty;
	sigemptyset(&empty);
	sigact.sa_sigaction = silent_deallocating_sighandler;
	sigact.sa_mask = empty;
	sigact.sa_flags = SA_SIGINFO | SA_NODEFER;

	sigaction(SIGBUS, &sigact, NULL);
	sigaction(SIGSEGV, &sigact, NULL);

	#if DUE
	deallocate_matrix(errinfo.neighbours);
	free(errinfo.neighbours);
	free((void*)errinfo.skipped_blocks);
	#endif

	#if DUE == DUE_ASYNC || DUE == DUE_IN_PATH
	free((void*)mp->shared_page_reductions);
	#endif

	free(errinfo.data);
}

void resilience_sighandler(int signum, siginfo_t *info, void *context UNUSED)
{
	if( (signum == SIGBUS /* && (info->si_code == BUS_MCEER_AR || info->si_code == BUS_MCEER_A0 )*/) ||
		(signum == SIGSEGV && info->si_code == SEGV_ACCERR) )
	{
		void * page = (void*)((long)info->si_addr - ((long)info->si_addr % (sizeof(double) << get_log2_failblock_size())));
		//info.si_add_lsb contains lsb of corrupted data, e.g. log2(sysconf(_SC_PAGESIZE)) for a full page
		// so long lastpage = (long)info.si_addr + (long)(1 << info.si_add_lsb);
		// and we should report all pages from page(page) to page(lastpage)

		// check if error was in data that we know to recover
		int block, vect = get_data_blockptr(page, &block);

		if( vect < 0 )
		{
			fprintf(stderr, "Error happened in memory that is not recoverable data : %p\n", page);
			crit_err_hdlr(signum, info, context);
			return;
		}

	#if DUE
		// mark vector of error and (pseudo-?)vector of output with error
		// TODO shouldn't it be this ? mark_to_skip(block, COMPLETE_WITH_FAIL(((1 << out_vect) | (1 << vect))) );
		mark_to_skip(block, (1 << out_vect) | (1 << vect) );
	#endif

		// notify globally
		__sync_fetch_and_add(&errinfo.errors, 1);

	#if DUE
		// notify this thread
		exception_happened++;
		if( out_vect == RECOVERY || out_vect == CHECKPOINT )
			errinfo.in_recovery_errors++;
	#endif


		// old : unprotect, mess with data in the page
		//if( signum == SIGSEGV )
		//	mprotect(page, sizeof(double) << get_log2_failblock_size(), PROT_READ | PROT_WRITE);
		//memset(page, 0x00, sizeof(double) << get_log2_failblock_size());

		// new : plain replace memory page
		void *ret = mmap(page, sizeof(double) << get_log2_failblock_size(), PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

		if (ret == MAP_FAILED)
			err(1, "Failed to remap failed page");


		log_err(SHOW_DBGINFO, "Error has just been  signaled  on page %3d of vector %s (%d)\n", block, vect_name(vect), vect);
	}
	else
	{
		fprintf(stderr, "Unexpected type of signal caught in handler : %d (%s) with code %d\n", signum, strsignal(signum), info->si_code);
		crit_err_hdlr(signum, info, context);
	}
}

void silent_deallocating_sighandler(int signum, siginfo_t *info, void *context UNUSED)
{
	// handler to silently deallocate memory even where we removed authorizations
	if(signum == SIGSEGV && info->si_code == SEGV_ACCERR)
	{
		int block, vect = get_data_blockptr(info->si_addr, &block), r;
		void * page = (void*)((long)info->si_addr - ((long)info->si_addr % (sizeof(double) << get_log2_failblock_size())));
		r = mprotect(page, sizeof(double) << get_log2_failblock_size(), PROT_READ | PROT_WRITE);

		fprintf(stderr, "SILENT handler caught %d SIGSEGV (%d SEGV_ACCERR), pointing to %p [vect %d, block %2d], mprotect returned %d\n", signum, info->si_code, info->si_addr, vect, block, r);
	}
	else
	{
		fprintf(stderr, "Unexpected type of signal caught in SILENT handler : %d (%s) with code %d\n", signum, strsignal(signum), info->si_code);
		crit_err_hdlr(signum, info, context);
	}
}

void sleep_ns(long long ns)
{
	struct timespec next_sim_fault, remainder, remainder2;

	next_sim_fault.tv_sec = ns / (long long)(1e9); // secs to next fault
	next_sim_fault.tv_nsec = ns % (long long)(1e9); // nanosecs to next fault

	int r = nanosleep(&next_sim_fault, &remainder);
	if( r != 0 )
	{
		perror("nanosleep interrupted ");
		r = nanosleep(&remainder, &remainder2);

		if( r != 0 )
			fprintf(stderr, "Nanosleep skipped %d.%09d of %d.%09d sleeping time because of 2 successive interruptions\n",
					(int)remainder2.tv_sec, (int)remainder2.tv_nsec, (int)next_sim_fault.tv_sec, (int)next_sim_fault.tv_nsec);
	}
}

void* simulate_failures(void* ptr)
{
	error_sim_data *sim_err = (error_sim_data*)ptr;

	// default cancellability state + nanosleep is a cancellation point
	//pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
	//pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

	int i, nerr = sim_err->nerr;
	long long faults_nsec[nerr+1];

	if(nerr)
	{
		double total_time = 0, mtbe = sim_err->lambda / (double)nerr, faults_unscaled[nerr+1];

		log_err(SHOW_FAILINFO, "Error is going to be simulated with exponential distribution (e^(-x/lambda))/lambda microseconds, lambda [~mtbe] = %e"
				", and time scaled back for %d errors in duration %e\n", mtbe, nerr, sim_err->lambda);

		// at first, create unscaled intervals between evenst (start, {faults}, end)
		for(i=0; i<nerr+1; i++)
		{
			faults_unscaled[i] = exponential(mtbe, (double)rand()/(double)RAND_MAX);
			total_time += faults_unscaled[i];
		}

		// now scale back total time interval to time given as parameter (in ns for sleep function)
		const double factor = sim_err->lambda * 1e3 / total_time;
		for(i=0; i<=nerr; i++)
			faults_nsec[i] = (long long)(factor * faults_unscaled[i] + 0.5);

		#if VERBOSE >= SHOW_FAILINFO
		char str[12*nerr]; str[0] = '\0';
		for(i=0; i<nerr; i++)
			sprintf(str + strlen(str), ", %lld", faults_nsec[i]);
		log_err(SHOW_FAILINFO, "Intervals in ns between error injections are %s\n", str+2);
		#endif

		i=0;
	}
	else
		log_err(SHOW_FAILINFO, "Error is going to be simulated with exponential distribution (e^(-x/lambda))/lambda microseconds, lambda [~mtbe] = %e\n", sim_err->lambda);


	// Now wait for everything to be nicely started & first gradient to exist etc.
	sem_wait(&sim_err->start_sim);
	// Release immediately so thread can be cancelled without problems to destroy semaphore
	sem_post(&sim_err->start_sim);

	for(i=0; nerr == 0 || i<nerr; i++)
	{
		if(nerr)
			sleep_ns(faults_nsec[i]);
		else
			sleep_ns((long long)(exponential(sim_err->lambda, (double)rand()/(double)RAND_MAX) * 1e3));

		cause_mpr(sim_err);
		//flip_a_bit(sim_err->info);
	}

	if(nerr)
	{
		sleep_ns(faults_nsec[nerr]);
		#if CKPT
		sim_err->info->ckpt_freq = INT_MAX;
		#endif
	}

	return NULL;
}

void cause_mpr(error_sim_data *sim_err)
{
	int rand_page = (int)( ((double)rand() / (double)RAND_MAX) * sim_err->info->nb_failblocks ) ;
	int vect      = (int)( ((double)rand() / (double)RAND_MAX) * sim_err->info->nb_data ) ;

	double* addr = sim_err->info->data[ vect ] + rand_page * sim_err->info->failblock_size;

	log_err(SHOW_DBGINFO, "Error is going to be triggered on page %3d of vector %s (%d) : %p\n", rand_page, vect_name(vect+1), vect+1, (void*)addr);

	sim_err->inj++;

	mprotect((void*)addr, sizeof(double) << sim_err->info->log2fbs, PROT_NONE);
	//madvise((void*)addr, sysconf(_SC_PAGESIZE), MADV_HWPOISON);
}

void flip_a_bit(analyze_err *info)
{
	int flip_pos = (int)( ((double)rand() / (double)RAND_MAX) * info->nb_failblocks * info->failblock_size ) ;
	int vect     = (int)( ((double)rand() / (double)RAND_MAX) * info->nb_data ) ;
	int flip_bit = (int)( ((double)rand() / (double)RAND_MAX) * 8 * sizeof(double) ) ;

	long long *victim = ((long long*)info->data[ vect ]) + flip_pos;

	#if VERBOSE >= SHOW_DBGINFO
	double before = info->data[ vect ][ flip_pos ];
	#endif

	(*victim) ^= (long long)(1 << flip_bit);

	sim_err.inj++;

	log_err(SHOW_DBGINFO,"Flipped bit %2d of double %5d (page %2d) in vect %2s :\t% .14e -> % .14e\tdiff = %e\n",
			flip_bit, flip_pos, flip_pos/info->failblock_size, mask_names[vect+1],
			before, info->data[ vect ][ flip_pos ], fabs(before - info->data[ vect ][ flip_pos ]));
}

int get_data_blockptr(const void *vect, int *block)
{
	int i;
	intptr_t ptr = (intptr_t)vect, pos;
	const intptr_t block_size = sizeof(double) << get_log2_failblock_size(), max_vect_size = errinfo.nb_failblocks * block_size;


	for(i=0; i<errinfo.nb_data; i++)
	{
		pos = ptr - (intptr_t)errinfo.data[i];

		if( pos >= 0 && pos < max_vect_size )
		{
			*block = (int)(pos / block_size);
			return i+1;
		}
	}

	return -1;
}

int get_data_vectptr(const double *vect)
{
	int i;
	for(i=0; i<errinfo.nb_data; i++)
		if( errinfo.data[i] == vect )
			return i+1;

	return -1;
}

// get info about failures, for the recovery methods
int check_recovery_errors()
{
	int r = (int)errinfo.in_recovery_errors;
	errinfo.in_recovery_errors = 0;
	return r;
}

int check_block(const int block, int input_mask, int *is_shared)
{
	// if fault happened in this task/thread, it is already marked with out_mask : nothing more to do
	if( check_for_exceptions() )
	{
		return 1;
		log_err(SHOW_FAILINFO, "\tMask %s skips block %2d, failed in this task [after  computing]\n", single_mask(COMPLETE_WITH_FAIL(1 << out_vect)), block);
	}

	// this is called a posteriori, be sure to not mark just 'skipped'
	int b;
	const int out_mask = COMPLETE_WITH_FAIL(1 << out_vect);
	input_mask &= ~CONSTANT_MASKS;

	// atomically: if input_mask is marked, add output mask
	// if input_mask is not marked (or removed concurrently) don't add ouput mask and return 0
	do
	{
		b = errinfo.skipped_blocks[block];
	}
	while( (b & input_mask) && ! __sync_bool_compare_and_swap( &(errinfo.skipped_blocks[block]), b, b | out_mask ) );

	#if VERBOSE >= SHOW_FAILINFO
	if(b & input_mask)
	{
		char mask_str[ 30 ];
		log_err(SHOW_FAILINFO, "\tMask %s skips block %2d (was %s : %x) [after  computing]\n", single_mask(out_mask), block, str_mask(mask_str, b), b);
	}
	#endif

	if (is_shared)
		*is_shared = b & SHARED_BLOCK;

	return (b & input_mask);
}

int should_skip_block(const int block, int mask)
{
	const int out_mask = (1 << out_vect);
	mask &= ~CONSTANT_MASKS;
	int b;

	do
	{
		b = errinfo.skipped_blocks[block];
	}
	while( (b & mask) && ! __sync_bool_compare_and_swap( &(errinfo.skipped_blocks[block]), b, b | out_mask ) );

	#if VERBOSE >= SHOW_FAILINFO
	if( b & mask )
	{
		char mask_str[ 30 ];
		log_err(SHOW_FAILINFO, "\tMask %s skips block %2d (was %s : %x) [before computing]\n", single_mask(out_mask), block, str_mask(mask_str, b), b);
	}
	#endif

	return ( b & mask );
}

int count_neighbour_faults(const int block, int mask)
{
	mask &= ~CONSTANT_MASKS;
	int r = 0, i;

	for(i=errinfo.neighbours->r[block]; i<errinfo.neighbours->r[block+1]; i++)
		r += ((errinfo.skipped_blocks[ errinfo.neighbours->c[i] ] & mask) > 0);
	return r;
}

void mark_to_skip(const int block, int mask)
{
	mask &= ~CONSTANT_MASKS;
	int before = __sync_fetch_and_or( &(errinfo.skipped_blocks[block]), mask );

	if( (before & ~CONSTANT_MASKS) == 0 )
	{
		#pragma omp atomic
			errinfo.skips ++ ;
	}

	#if VERBOSE >= SHOW_FAILINFO
	char mask_str[ 30 ];
	log_err(SHOW_FAILINFO, "\tblock %2d marked as skipped/for skipping with mask %s : %x (was %x)\n", block, str_mask(mask_str, mask), mask, before);
	#endif
}

void mark_corrected(const int block, int mask)
{
	mask &= ~CONSTANT_MASKS;
	const int complete_mask = COMPLETE_WITH_FAIL(mask);
	int before = __sync_fetch_and_and( &(errinfo.skipped_blocks[block]), ~complete_mask );

	// if last thing skipped for this block was the one we just removed
	if( (before & ~CONSTANT_MASKS) > 0 && (before & complete_mask) == (before & ~CONSTANT_MASKS) )
	{
		#pragma omp atomic
			errinfo.skips -- ;
	}

	#if VERBOSE >= SHOW_FAILINFO
	char mask_str[ 30 ];
	log_err(SHOW_FAILINFO, "Before correction marked (with mask %s : %x), skipped block %2d is %s : %x\n", single_mask(mask), mask, block, str_mask(mask_str, before), before);
	#endif
}

int aggregate_skips()
{
	int i, r = 0;
	for(i=0; i<errinfo.nb_failblocks; i++)
		r |= errinfo.skipped_blocks[i];

	return r & (~CONSTANT_MASKS);
}

int has_skipped_blocks(int mask)
{
	mask &= ~CONSTANT_MASKS;
	int i, r = 0;
	for(i=0; i<errinfo.nb_failblocks; i++)
		if( errinfo.skipped_blocks[i] & mask )
		{
			r++;
			break;
		}

	return r;
}

int is_skipped_not_failed_block(const int block, int mask)
{
	mask &= ~CONSTANT_MASKS;
	int b = errinfo.skipped_blocks[block] & COMPLETE_WITH_FAIL(mask);

	return REMOVE_FAIL(mask) == b;
}

int is_failed_not_skipped_block(const int block, int mask)
{
	mask &= ~CONSTANT_MASKS;
	int b = errinfo.skipped_blocks[block] & COMPLETE_WITH_FAIL(mask);

	return COMPLETE_WITH_FAIL(mask) == b;
}

int overlapping_faults(int mask_v, int mask_w)
{
	mask_v &= ~CONSTANT_MASKS;
	mask_w &= ~CONSTANT_MASKS;
	int i, r = 0, block;

	for(i=0; i<errinfo.nb_failblocks; i++)
	{
		block = errinfo.skipped_blocks[i];
		if( block & mask_v && block & mask_w )
		{
			r++;
			break;
		}
	}

	return r;
}

void clear_failed_blocks(int mask, const int start, const int end)
{
	mask &= ~CONSTANT_MASKS;
	int i;
	for(i = (start >> errinfo.log2fbs); i < (end >> errinfo.log2fbs) ; i++)
		if( errinfo.skipped_blocks[i] & mask )
			__sync_fetch_and_and(&(errinfo.skipped_blocks[i]), ~mask);
}

void clear_failed(int mask)
{
	mask &= ~CONSTANT_MASKS;
	int i;
	for(i=0; i<errinfo.nb_failblocks; i++)
		if( errinfo.skipped_blocks[i] & mask )
			__sync_fetch_and_and(&(errinfo.skipped_blocks[i]), ~mask);
}

void clear_failed_vect(const double *vect)
{
	clear_failed( 1 << get_data_vectptr(vect) );
}

int get_all_failed_blocks(int mask, int **lost_blocks)
{
	mask &= ~CONSTANT_MASKS;
	int total_skips = errinfo.skips;
	if( ! total_skips )
		return 0;

	*lost_blocks = (int*) calloc( total_skips, sizeof(int) );
	if (*lost_blocks == NULL)
		err(1, "Failed to allocate *lost_blocks");

	int i, j = 0;
	for(i=0; i<errinfo.nb_failblocks && j < total_skips; i++)
		if( errinfo.skipped_blocks[i] & mask )
			(*lost_blocks)[ j++ ] = i;

	return j;
}

int get_all_failed_blocks_vect(const double *v, int **lost_blocks)
{
	int vect = get_data_vectptr(v);

	if( vect >= 0 )
		return get_all_failed_blocks(1 << vect, lost_blocks);
	else
		return 0;
}


void get_recovering_blocks_bounds(int *start, int *end, const int *lost, const int nb_lost)
{
	int min_block = errinfo.nb_failblocks + 1, max_block = -1, i, b;

	for(i=0; i<nb_lost; i++)
	{
		if( lost[i] < min_block )
			min_block = lost[i];
		if( lost[i] > max_block )
			max_block = lost[i];
	}

	int min_lost_item = min_block << get_log2_failblock_size(), max_lost_item = ((max_block + 1) << get_log2_failblock_size()) -1;

	// link to blocks
	for(b=0; b<nb_blocks; b++)
	{
		if( min_lost_item >= get_block_start(b) && min_lost_item < get_block_end(b) )
			*start = get_block_start(b);
		if( max_lost_item >= get_block_start(b) && max_lost_item < get_block_end(b) )
			*end = get_block_end(b);
	}
}

// function setting the number and set of lost blocks that are errinfo.neighbours with block id
void get_failed_neighbourset(const int *all_lost, const int nb_lost, const int start_block, int *set, int *num)
{
	int i, j, k = 0, added[errinfo.nb_failblocks];

	for(i=0; i<errinfo.nb_failblocks; i++)
		added[i] = 0;

	*num = 1;
	added[start_block] = 1;
	set[k] = start_block;

	do {
		// search the neighbours of set_k
		// if set_k and i are neighbours and i is found in the failed blocks, add i to set
		for(i=errinfo.neighbours->r[ set[k] ]; i < errinfo.neighbours->r[ set[k] + 1 ]; i++)
			if( added[ errinfo.neighbours->c[i] ] == 0 )
			{
				for(j=0; j<nb_lost; j++)
					if( all_lost[j] == errinfo.neighbours->c[i] )
					{
						added[ errinfo.neighbours->c[i] ] = 1;
						set[*num] = errinfo.neighbours->c[i];
						(*num)++;
					}
			}
	}
	// if a failed block in set has failed errinfo.neighbours, we should add them too
	while( ++k < *num );

	// okay now we should really sort the set...
	// should be mostly a small list, partly sorted already
	// so kiss and go for an insertion sort
	int insert;
	for (i = 1; i < *num; i++)
	{
		insert = set[i];

		for (j = i; j > 0 && set[j - 1] > insert ; j--)
			set[j] = set[j - 1];

		set[j] = insert;
	}
}

void compute_neighbourhoods(const Matrix *mat, const int bs, Matrix *neighbours)
{
	int i, ii, bi, k, bj, pos = 0, set_in_block[errinfo.nb_failblocks];

	// iterate all lines, i points to the start of the block, ii to the line and bi to the number of the block
	for(i=0, bi=0; i < mat->n; i += bs, bi++ )
	{
		neighbours->r[bi] = pos;

		for(bj=0; bj<errinfo.nb_failblocks; bj++)
			set_in_block[bj] = 0;

		for( ii = i; ii < i+bs && ii < mat->n ; ii ++ )

			// iterate all columns, k points to the position in mat, and bj to the number of the block
			for(k = mat->r[ii] ; k < mat->r[ii+1] ; k++ )
			{
				bj = mat->c[k] / bs;
				if( mat->v[k] != 0.0 && !set_in_block[bj])
					set_in_block[bj] = 1;
			}

		for(bj=0; bj<errinfo.nb_failblocks; bj++)
			if( set_in_block[bj] )
			{
				neighbours->c[pos] = bj;
				pos++;
			}
	}

	neighbours->r[neighbours->n] = neighbours->nnz = pos;
}


