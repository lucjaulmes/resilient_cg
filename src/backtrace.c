#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef __USE_GNU
#define __USE_GNU
#endif

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>
#include <err.h>

#include "backtrace.h"

void crit_err_hdlr(int sig_num, siginfo_t * info, void * ucontext)
{
	void *             array[50];
	void *             caller_address;
	char **            messages;
	int                size, i;
	sig_ucontext_t *   uc;

	uc = (sig_ucontext_t *)ucontext;

	/* Get the address at the time the signal was raised */
	#if defined(__i386__) // gcc specific
	caller_address = (void *) uc->uc_mcontext.eip; // EIP: x86 specific
	#elif defined(__x86_64__) // gcc specific
	caller_address = (void *) uc->uc_mcontext.rip; // RIP: x86_64 specific
	#else
	#error Unsupported architecture.
	#endif

	fprintf(stderr, "signal %d (%s), code is %d [SEGV_ACCERR=%d] and address is %p from %p\n",
			sig_num, strsignal(sig_num), info->si_code, SEGV_ACCERR, info->si_addr,
			(void *)caller_address);

	size = backtrace(array, 50);

	/* overwrite sigaction with caller's address
	array[1] = caller_address; */

	messages = backtrace_symbols(array, size);

	/* skip first stack frame (points here) */
	for (i = 1; i < size && messages != NULL; ++i)
		fprintf(stderr, "[bt]: (%d) %s\n", i, messages[i]);

	free(messages);

	struct sigaction sigact = (struct sigaction){.sa_handler = SIG_DFL};
	if (sigaction(SIGSEGV, &sigact, NULL))
		err(1, "Failed restoring segfault handler for propagation");
	raise(SIGSEGV);
}

void register_sigsegv_handler()
{
	struct sigaction sigact;
	sigact.sa_sigaction = crit_err_hdlr;
	sigact.sa_flags = SA_RESTART | SA_SIGINFO;

	if (sigaction(SIGSEGV, &sigact, (struct sigaction*)NULL) != 0)
		fprintf(stderr, "error setting signal handler for %d (%s)\n", SIGSEGV, strsignal(SIGSEGV));

	if (sigaction(SIGBUS, &sigact, (struct sigaction*)NULL) != 0)
		fprintf(stderr, "error setting signal handler for %d (%s)\n", SIGBUS, strsignal(SIGBUS));
}

