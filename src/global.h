#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

//from HUGEPAGES doc
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000 /* arch specific */
#endif


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

#define CKPT_NONE		0
#define CKPT_TO_DISK	1
#define CKPT_IN_MEMORY	2

#define DUE_NONE		0
#define DUE_ASYNC		1
#define DUE_IN_PATH		2
#define DUE_ROLLBACK	3
#define DUE_LOSSY		4

#if DUE == DUE_ROLLBACK
	#if CKPT == CKPT_NONE
	#error you have to define a checkpoint strategy to use rollback
	#endif
#endif

#if CKPT
	#if DUE != DUE_ROLLBACK
	#error checkpointing strategy defined but not used
	#endif
#endif

#if HUGEPAGES
	#define HUGEPAGE_FLAG MAP_HUGETLB
#else
	#define HUGEPAGE_FLAG 0x0
#endif

#ifdef UNUSED
#elif defined(__GNUC__)
	#define UNUSED __attribute__((unused))
#elif defined(__LCLINT__)
	#define UNUSED /*@unused@*/
#elif defined(__cplusplus)
	#define UNUSED
#endif

#include <string.h>
#include <stdlib.h>
#include <errno.h>

// a few global vars for parameters that everyone needs to know
extern int nb_blocks;
extern int MAXIT;

extern int *block_ends;
static inline void set_block_end(const int b, const int pos)
{
	block_ends[b] = pos;
}

static inline int get_block_start(const int b)
{
	if( b == 0 )
		return 0;
	else
		return block_ends[b-1];
}

static inline int get_block_end(const int b)
{
	return block_ends[b];
}

static inline size_t round_up(size_t size, size_t alignment)
{
	// alignment has to be a power of 2. We round size to the closest multiple of alignment
	return ((size-1) | (alignment - 1)) + 1; // 4 ops
	//return (size + (alignment - 1)) & ~(alignment - 1); // 4 ops also ? a-1, s+a, ~a, &
}

// for get_log2_failblock_size()
#include "failinfo.h"

static inline void* big_calloc(size_t size)
{

	size_t aligned_size = round_up(size, sizeof(double) << get_log2_failblock_size());
	void* ptr = mmap(NULL, aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | HUGEPAGE_FLAG, -1, 0);
	if( ptr == (void*)-1 )
	{
		perror("aligned_alloc failed");
		exit(errno);
	}
	return memset(ptr, 0, aligned_size);
}

static inline void big_free(void* ptr, size_t size)
{
	munmap(ptr, round_up(size, sizeof(double) << get_log2_failblock_size()));
}

#endif // GLOBAL_H_INCLUDED

