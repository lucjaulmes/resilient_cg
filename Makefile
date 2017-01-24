####################################################################################################
#### this Makefile allows to build target by giving the flavour options in the top directory
#### e.g. make DUE=rollback CKPT=disk
####################################################################################################
# options to build differently behavioured same targets

# what our options can be, note that same order than values defiend in global.h
DUE_OPTS:=none async path rollback lossy
CKPT_OPTS:=none disk mem

# enough to number all our options
numbers:=$(shell seq 0 4)

# lousy defaults
DUE=none
CKPT=none

#check if variables in list of allowed values
ifeq ($(filter $(DUE),$(DUE_OPTS)),)
$(error Please choose DUE from "$(DUE_OPTS)")
else ifeq ($(filter $(CKPT),$(CKPT_OPTS)),)
$(error Please choose CKPT from $(CKPT_OPTS))
endif

# use function to find number of variable in list. 0-numbered.
pos=$(filter $(addprefix $(1), $(numbers)), $(join $(2), $(numbers)))
D:=$(subst $(DUE),,$(call pos, $(DUE), $(DUE_OPTS)))
C:=$(subst $(CKPT),,$(call pos, $(CKPT), $(CKPT_OPTS)))

FLAVOUR=-DDUE=$D -DCKPT=$C

DIR=$(DUE)_$(CKPT)

empty=
space=$(empty) $(empty)

#####################################################################################################

# Finally defining directories and targets
DESTDIR:=$(subst $(space),_,$(DIR))

.PHONY : default clean all ompssall plainall debug ompssdebug plaindebug conv ompssconv plainconv speedup ompssspeedup plainspeedup instr ompssinstr plaininstr
default:ompssall

all ompssall plainall debug ompssdebug plaindebug conv ompssconv plainconv speedup ompssspeedup plainspeedup instr ompssinstr plaininstr: $(DESTDIR)
	+$(MAKE) -C $(DESTDIR) -f ../Makefile.sub "FLAVOUR=$(FLAVOUR)" VPATH=../src $(@)

clean:
	rm -rf async_* lossy_* none_* path_* rollback_*

.SECONDEXPANSION:
$(DESTDIR):
	mkdir -p $(DESTDIR)

