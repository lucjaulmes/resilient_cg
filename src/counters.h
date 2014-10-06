#ifndef COUNTERS_H
#define COUNTERS_H

#include "global.h"
#include "debug.h"

void setup_measure();
void unset_measure();

void start_measure();
void stop_measure();

#if !defined PERFORMANCE || defined EXTRAE_EVENTS
void log_convergence(const int r, const double e, const int f);
void log_sdc(const double v, const int f);
#else
#define log_convergence(r, e, f) {}
#define log_sdc(v, f) {}
#endif

#endif // COUNTERS_H

