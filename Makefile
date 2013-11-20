CC = gcc
MCC = mcc

SRC = mmio.c main.c matrix.c failinfo.c solvers.c recover.c gmres.c cg.c #$(shell ls -c *.c | grep -v ^mcc_)
OBJECTS = $(SRC:.c=.o)
HEADERS = $(shell ls *.h)

OPTS = -lm
DBG_OPTS = -Wall -g -DVERBOSE=2
RLS_OPTS = -O3 -DPERFORMANCE
CC_OPTS = -Wno-unknown-pragmas -fdiagnostics-show-option #has to come after Wall
MCC_OPTS = --keep-all-files --ompss --output-dir=.mercurium
M_D_OPTS = -L/usr/local/lib64/instrumentation  --instrumentation
M_R_OPTS = -L/usr/local/lib64/performance
DENSE_OPTS = -DMATRIX_DENSE

actual : debug
all : release debug
release : sparserelease denserelease
debug : sparsedebug densedebug
dense : densedebug denserelease
sparse : sparsedebug sparserelease

densedebug : $(SRC) $(HEADERS)
	$(CC) $(DENSE_OPTS) $(OPTS) $(DBG_OPTS) $(CC_OPTS) $(SRC) -o dense

denserelease : $(SRC) $(HEADERS)
	$(CC) $(DENSE_OPTS) $(OPTS) $(RLS_OPTS) $(CC_OPTS) $(SRC) -o rls/dense

sparsedebug : $(SRC) $(HEADERS)
	$(CC) $(OPTS) $(DBG_OPTS) $(CC_OPTS) $(SRC) -o sparse

sparserelease : $(SRC) $(HEADERS)
	$(CC) $(OPTS) $(RLS_OPTS) $(CC_OPTS) $(SRC) -o rls/sparse

clean : 
	@rm -r sparse dense *.o rls/* .mercurium/* graph.dot

print : 
	@echo $(SRC)

