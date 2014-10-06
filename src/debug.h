#ifndef DEBUG_H_INCLUDED
#define DEBUG_H_INCLUDED

#include <stdarg.h>
#include "global.h"

// if we want to use several levels of verbosity
#define SHOW_DBGINFO  1
#define SHOW_FAILINFO 2
#define SHOW_TASKINFO 3
#define SHOW_TOOMUCH  4

// if we defined PERFORMANCE we are going to be very silent
// if we defined VERBOSE we are going to be selectively talkative

#ifndef PERFORMANCE

// this is all a bit manual, but necessary to remove side-effects of the VA_ARGS
// if we don't #define the unused log_err to {} (e.g. declaring to a function that does nothing)
// then the compiler will keep whatever's in the call for side-effects. BAD !
// i.e. log_err(SHOW_TOOMUCH, "%e\n", norm(vect)) will still spend time & FLOPS computing norm(vect) but never show it

#if VERBOSE >= 1
#define log_err_1(...) fprintf(stderr, __VA_ARGS__)
#endif

#if VERBOSE >= 2
#define log_err_2(...) fprintf(stderr, __VA_ARGS__)
#endif

#if VERBOSE >= 3
#define log_err_3(...) fprintf(stderr, __VA_ARGS__)
#endif

#if VERBOSE >= 4
#define log_err_4(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef log_err_1
#define log_err_1(...) {}
#endif

#ifndef log_err_2
#define log_err_2(...) {}
#endif

#ifndef log_err_3
#define log_err_3(...) {}
#endif

#ifndef log_err_4
#define log_err_4(...) {}
#endif


#define log_out(...) printf(__VA_ARGS__)

// and now refer to those nice log_err_X
#define PASTE(a,b) a ## b
#define log_err(level, ...) PASTE(log_err_, level)(__VA_ARGS__)

#else

#undef VERBOSE

#define log_out(...) {}
#define log_err(...) {}

#endif

#endif // DEBUG_H_INCLUDED

