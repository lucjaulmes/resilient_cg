#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

// things that should be defined globally : constants, functions, etc.
// these are the possible fault strategies

// some values we pass around
#define MULTFAULTS_GLOBAL		1
#define MULTFAULTS_UNCORRELATED	2
#define MULTFAULTS_DECORRELATED	3

#define DO_NOTHING         0
#define SAVE_CHECKPOINT    1
#define RELOAD_CHECKPOINT  2
#define RESTART_CHECKPOINT 3

#ifndef RECOMPUTE_GRADIENT_FREQ
#define RECOMPUTE_GRADIENT_FREQ 50
#endif

#define SDC_NONE	 0
#define SDC_ALPHA    1
#define SDC_ORTHO    2
#define SDC_GRADIENT 3

#define CKPT_NONE		0
#define CKPT_TO_DISK	1
#define CKPT_IN_MEMORY	2

#define DUE_NONE		0
#define DUE_ASYNC		1
#define DUE_IN_PATH		2
#define DUE_ROLLBACK	3
#define DUE_LOSSY		4

#if SDC
	#if ! CHECK_SDC_FREQ
	#error you have to define a SDC check frequency
	#endif
	
	#if (SDC == SDC_GRADIENT) && (CHECK_SDC_FREQ % RECOMPUTE_GRADIENT_FREQ) != 0
	#error Using GRADIENT as SDC detection, CHECK_SDC_FREQ must be a multiple of RECOMPUTE_GRADIENT_FREQ
	#endif
#endif

#if SDC || DUE == DUE_ROLLBACK
	#if CKPT == CKPT_NONE
	#error you have to define a checkpoint strategy
	#endif
#endif

#if CKPT
	#if ! SDC && DUE != DUE_ROLLBACK
	#error checkpointing strategy defined but not used
	#endif

	#if ! CHECKPOINT_FREQ || (SDC && (CHECKPOINT_FREQ % CHECK_SDC_FREQ ) != 0) // multiple of check_sdc_freq if sdc is on
	#error you have to define a checkpoint frequency, multiple of CHECK_SDC_FREQ if you enable SDC checking
	#endif
#endif

#ifdef UNUSED
#elif defined(__GNUC__)
	#define UNUSED __attribute__((unused))
#elif defined(__LCLINT__)
	#define UNUSED /*@unused@*/
#elif defined(__cplusplus)
	#define UNUSED
#endif

#define STRINGIFY(a) #a

#if ! _ISOC11_SOURCE 
//#warning ISOC11_SOURCE not defined ! Replacing aligned_alloc from glibc >= 2.12 with memalign
#include <malloc.h>
#define aligned_alloc memalign
#endif

#include <string.h>
#include <stdlib.h>
#include <errno.h>

// a few global vars for parameters that everyone needs to know
extern int nb_blocks;
extern int MAXIT;
extern int *block_bounds;
extern int mpi_here, *mpi_zonestart, *mpi_zonesize;

static inline int get_block_start(const int b)
{
	return block_bounds[b];
}

static inline int get_block_end(const int b)
{
	return block_bounds[b+1];
}

static inline size_t round_up(size_t size, size_t alignment)
{
	// alignment has to be a power of 2. We round size to the closest multiple of alignment
	return ((size-1) | (alignment - 1)) + 1; // 4 ops
	//return (size + (alignment - 1)) & ~(alignment - 1); // 4 ops also ? a-1, s+a, ~a, &
}

static inline void* aligned_calloc(size_t alignment, size_t size)
{
	size_t aligned_size = round_up(size, alignment);
	void *ptr = aligned_alloc(alignment, aligned_size);
	if( ptr == NULL )
	{
		perror("aligned_alloc failed");
		exit(errno);
	}
	return memset(ptr, 0, aligned_size);
}

static inline char* alloc_deptoken()
{
	char *deptoken = aligned_alloc(64, 64);

	return deptoken;
}

#endif // GLOBAL_H_INCLUDED

