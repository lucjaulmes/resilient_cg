CC = gcc
MCC = mcc

SRC = $(shell ls -c *.c | grep -v ^mcc_)
OBJECTS = $(SRC:.c=.o)
HEADERS = $(shell ls *.h)

OPTS = -lm
DBG_OPTS = -Wall -g
RLS_OPTS = -O3
CC_OPTS = -Wno-unknown-pragmas -fdiagnostics-show-option #has to come after Wall
MCC_OPTS = -L/usr/local/lib64/instrumentation --keep-all-files --ompss --output-dir=.mercurium --instrumentation

actual : plaindebug
all : release debug
release : plainrelease ompssrelease
debug : plaindebug ompssdebug
allplain : plaindebug plainrelease
allompss : ompssdebug ompssrelease

plaindebug : $(SRC) $(HEADERS)
	$(CC) $(OPTS) $(DBG_OPTS) $(CC_OPTS) $(SRC) -o plain

plainrelease : $(SRC) $(HEADERS)
	$(CC) $(OPTS) $(RLS_OPTS) $(CC_OPTS) $(SRC) -o rls/plain

ompssdebug : $(SRC) $(HEADERS)
	$(MCC) $(OPTS) $(DBG_OPTS) $(MCC_OPTS) $(SRC) -o ompss

ompssrelease : $(SRC) $(HEADERS)
	$(MCC) $(OPTS) $(RLS_OPTS) $(MCC_OPTS) $(SRC) -o rls/ompss

clean : 
	@rm -r plain ompss mcc_* *.o rls/* .mercurium/*

print : 
	@echo $(SRC)

