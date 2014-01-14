CC = gcc

SRC = mmio.c csparse.c main.c matrix.c failinfo.c counters.c dense_solvers.c recover.c gmres.c cg.c #$(shell ls -c *.c | grep -v ^mcc_)
OBJECTS = $(SRC:.c=.o)
HEADERS = $(shell ls *.h)

#####################################################################################################
#defining several options  here that you can add to the options

# show backtrace on segfault (from inside the program) (traces are nicer with -g as well as -rdynamic)
USE_BT = -rdynamic -DBACKTRACE 

# to measure performance, we can use the gettimeofday (by default),
USE_TIMEOFDAY = -DGETTIMEOFDAY
# or alternatively :
# - the timers from the papi library (counts cycles, usecs)
USE_PAPI = -I/apps/PAPI/5.0.1/include -L/apps/PAPI/5.0.1/lib -lpapi -DPAPICOUNTERS
# - or use extrae to measure performance, by defining custom extrae events
USE_EXTRAE = -lseqtrace -DEXTRAE_EVENTS 

# use dense instead of sparse matrices
USE_DENSMAT = -DMATRIX_DENSE

#for verbosity : 
# -DPERFORMANCE is going to make the program shut up, 
# -DVERBOSE=x make it talk, with x from 0 (talk just a bit) to 6 (talk a LOT)

#this will be passed to debug builds, add any of the above options
DBG_OPTS = -Wall -g -DPERFORMANCE $(USE_BT) $(USE_TIMEOFDAY)
#this will be passed to release builds
RLS_OPTS = -O1 -DPERFORMANCE $(USE_TIMEOFDAY)
#####################################################################################################

OPTS = -lm
CC_OPTS = -Wno-unknown-pragmas -fdiagnostics-show-option #has to come after Wall

actual : debug
all : release debug

debug : $(SRC) $(HEADERS)
	$(CC) $(OPTS) $(DBG_OPTS) $(CC_OPTS) $(SRC) -o seq

release : $(SRC) $(HEADERS)
	$(CC) $(OPTS) $(RLS_OPTS) $(CC_OPTS) $(SRC) -o rls/seq

clean : 
	@rm -rfv seq *.o rls/*

print : 
	@echo $(SRC)

