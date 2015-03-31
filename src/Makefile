CC=gcc
CFLAGS=-g -Wall -Wextra -Werror -Wno-unknown-pragmas -Wno-unused-parameter
LDFLAGS=-lm

pcg:$(patsubst %.c,%.o,$(wildcard *.c))
	$(CC) $(CFLAGS) $(LDFLAGS) -Wall -Wextra $^ -o $@

