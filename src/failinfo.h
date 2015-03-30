#ifndef FAILINFO_H_INCLUDED
#define FAILINFO_H_INCLUDED

#include <stdarg.h>    // vargs
#include <pthread.h>   // pthread_t
#include <semaphore.h> // sem_t
#include <signal.h>    // siginfo_t
#include <string.h>    // ffs
#include "global.h"
#include "cg.h"        // magic_pointers, ASSOC_CONST_MP , VECT_xxx and MASK_xxx
#include "matrix.h"    // Matrix

#define BASESIZE 16

#define KEEP_FAULTS   0
#define REMOVE_FAULTS 1

// bitmasks for the fault vectors
#define MASK_ITERATE    (1 << VECT_ITERATE)
#define MASK_A_ITERATE  (1 << VECT_A_ITERATE)
#define MASK_GRADIENT   (1 << VECT_GRADIENT)
#define MASK_P          (1 << VECT_P)
#define MASK_OLD_P      (1 << VECT_OLD_P)
#define MASK_A_P        (1 << VECT_A_P)

#define MASK_NORM_G     (1 << NORM_GRADIENT)
#define MASK_NORM_A_P   (1 << NORM_A_P)

#define MASK_RECOVERY   (1 << RECOVERY)
#define MASK_X_EXCHANGE (1 << MPI_X_EXCHANGE)
#define MASK_P_EXCHANGE (1 << MPI_P_EXCHANGE)

// the two next ones mean (in conjugation with the previous corresponding item) that 
// if g (resp. Ap) is marked failed with this extra bit, it has been skipped, but is not
// contaminated with errors : last possible bits in mask
#define FAIL_GRADIENT  ((1 << 30) | MASK_GRADIENT)
#define FAIL_A_P       ((1 << 31) | MASK_A_P)

#define COMPLETE_WITH_FAIL(mask) \
	(mask | (mask & MASK_GRADIENT ? FAIL_GRADIENT : 0) | (mask & MASK_A_P ? FAIL_A_P : 0))

#define REMOVE_FAIL(mask) (mask & ~((FAIL_GRADIENT ^ MASK_GRADIENT) | (FAIL_A_P ^ MASK_A_P)))

extern const char * const mask_names[];// = { "0 ", "X ", "Ax", "G ", "P4", "P5", "Ap", "7 ", "8 ", "Ng", "Np", "RC", "Fg", "Fp" };
static inline const char * vect_name(const int vect)
{
	return mask_names[vect];
}
#if VERBOSE 
static inline const char * single_mask(const int mask)
{
	switch( mask ) // with ffs ~= sizeof(mask) - clz should give this without branching
	{
		case FAIL_GRADIENT : return mask_names[    12     ];
		case FAIL_A_P      : return mask_names[    13     ];
		default            : 
			if( ffs(mask)-1 < 32 )
				return mask_names[ffs(mask)-1];
			else
				return "??";
	}
}
static inline char * str_mask(char * str, const int mask)
{
	int m = mask, b;
	char * pos = str;

	while(m)
	{
		b = ffs(m) - 1;
		m ^= 1 << b;

		if( b < 32 )
			strcpy(pos, mask_names[b]);
		else
			strcpy(pos, "??");
		pos += 2;
	}
	// null-terminate
	*pos = '\0';

	return str;
}
#else
#define single_mask(m) m
#define str_mask(s, m) m
#endif

typedef struct analyze_err
{
	int nb_failblocks, failblock_size, log2fbs, fault_strat;
	Matrix *neighbours;

	#if CKPT == CKPT_TO_DISK
	const char *ckpt;
	#endif

	// initialized last : data pointers info
	double **data;
	int nb_data;

	int in_recovery_errors, errors, skips;
	int *skipped_blocks;
} analyze_err;

typedef struct error_sim_data
{
	// simulation infos, parameters, etc.
	double lambda;
	int nerr;
	pthread_t th;
	sem_t start_sim;

	analyze_err *info;
} error_sim_data;

extern error_sim_data sim_err;
extern analyze_err errinfo;

// these are used to communicate between a thread and its tasks and vice versa, but not between threads
extern __thread sig_atomic_t out_vect, exception_happened;

// information about block-topology that does not inform on the errors
int get_data_vectptr(const double *vect);
int get_data_blockptr(const void *vect, int *block);

// get info about failures, for the recovery methods

static inline int get_nb_failblocks()
{
	return errinfo.nb_failblocks;
}

static inline int get_log2_failblock_size()
{
	return errinfo.log2fbs;
}

static inline int get_failblock_size()
{
	return errinfo.failblock_size;
}

static inline int get_strategy()
{
	return errinfo.fault_strat;
}

static inline int check_errors_signaled()
{
	return __sync_fetch_and_and( &errinfo.errors, 0 );
}

static inline int get_nb_failed_blocks()
{
	return errinfo.skips;
}

static inline int is_skipped_block(const int block, const int mask)
{
	return errinfo.skipped_blocks[block] & mask;
}

int aggregate_skips();
int has_skipped_blocks(const int mask);
int is_skipped_not_failed_block(const int block, const int mask);
int is_failed_not_skipped_block(const int block, const int mask);
int check_recovery_errors();

void mark_to_skip(const int block, const int mask);
void mark_corrected(const int block, const int mask);
int should_skip_block(const int block, const int mask);
int check_block(const int block, const int input_mask);
int count_neighbour_faults(const int block, const int mask);

int overlapping_faults(const int mask_v, const int mask_w);

int get_all_failed_blocks_vect(const double *v, int **lost_blocks);
int get_all_failed_blocks(const int mask, int **lost_blocks);
void clear_failed(const int mask);
void clear_failed_blocks(const int mask, const int start, const int end);
void clear_failed_vect(const double *vect);
#define clear_mvm() clear_failed( FAIL_A_P | MASK_A_ITERATE )

static inline void enter_task(const int vect)
{
	exception_happened = 0;
	out_vect = vect;
}

static inline void enter_task_vect(const double *output_vect)
{
	out_vect = get_data_vectptr(output_vect);
	exception_happened = 0;
}

static inline void exit_task()
{
	out_vect = 0;
}

static inline int check_for_exceptions()
{
	int r = exception_happened;
	exception_happened = 0;

	return r ;
}

// signal handlers
void resilience_sighandler(int signum, siginfo_t *info, void *context);
void silent_deallocating_sighandler(int signum, siginfo_t *info, void *context);

// cause an error
void flip_a_bit(analyze_err *info);
void cause_mpr(error_sim_data *sim_err);
void* simulate_failures(void *ptr);

// setup methods, called before anything happens
void populate_global(const int n, const int fail_size_bytes, const int fault_strat, const int max_err, const double lambda, const char *checkpoint_path);
void setup_resilience(const Matrix *A, const int nb, magic_pointers *mp);
void start_error_injection();
void unset_resilience();
void compute_neighbourhoods(const Matrix *mat, const int bs, Matrix *neighbours);

void get_failed_neighbourset(const int *all_lost, const int nb_lost, const int start_block, int *set, int *num);
void get_recovering_blocks_bounds(int *start, int *end, const int *lost, const int nb_lost);


#endif // FAILINFO_H_INCLUDED

