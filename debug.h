#ifndef DEBUG_H_INCLUDED
#define DEBUG_H_INCLUDED

#include <stdarg.h>

// if we want to use several levels of verbosity
#define FULL_VERBOSE 2
#define LIGHT_VERBOSE 1

#ifdef PERFORMANCE
#undef VERBOSE
#endif

// if we defined PERFORMANCE we are going to be very silent
// if we defiend VERBOSE we are going to be very talkative

static inline void log_out(const char* fmt, ...)
{
	#ifndef PERFORMANCE
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
	#endif
}

static inline void log_err(const char* fmt, ...)
{
	#if defined VERBOSE
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
	#endif
}

#endif // DEBUG_H_INCLUDED

