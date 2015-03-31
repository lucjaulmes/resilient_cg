#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

// things that should be defined globally : constants, functions, etc.
// these are the possible fault strategies
#define NOFAULT 0
#define SINGLEFAULT 1
#define MULTFAULTS_GLOBAL 2
#define MULTFAULTS_UNCORRELATED 3
#define MULTFAULTS_DECORRELATED 4

#define MAX_THREADS 64

#ifndef RECOMPUTE_GRADIENT_FREQ
#define RECOMPUTE_GRADIENT_FREQ 50
#endif

#ifdef UNUSED
#elif defined(__GNUC__)
	#define UNUSED __attribute__((unused))
#elif defined(__LCLINT__)
	#define UNUSED /*@unused@*/
#elif defined(__cplusplus)
	#define UNUSED
#endif

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


// alignment functions, blocksize (and typesize) has to be a power of 2
// suppose cache line size to be 64, we want to align to the smallest multiple
static inline int get_alignment() { return 64 ; }
static inline void align_ptr(void** ptr)
{
	const unsigned long align = get_alignment(), pcast = (const unsigned long)*ptr, mask = align-1;

	if( (pcast & mask) == 0 )
		return;

	*ptr = (void*) (( pcast & ~mask ) + align );
}

static inline void align_index(int *pos, int typesize)
{
	// here adding 1 to index increases by 'typesize' bytes
	const int align = get_alignment(), mask = align/typesize -1 ;
	
	if( (*pos & mask) == 0 )
		return;
	
	*pos = (*pos & ~mask) + align/typesize;
}

static inline char* alloc_deptoken()
{
	char *deptoken = NULL;
	posix_memalign( (void**)&deptoken, get_alignment(), 1);

	return deptoken;
}

#endif // GLOBAL_H_INCLUDED

