####################################################################################################
#### this Makefile allows to build target by giving the flavour options in the top directory 
#### e.g. make DUE=none CKPT=disk CKPT_FREQ=40
####################################################################################################
# options to build differently behavioured same targets

# what our options can be, note that same order than values defiend in global.h
DUE_OPTS:=none async path rollback lossy
CKPT_OPTS:=none disk mem

# engough to number all our options
numbers:=$(shell seq 0 4)

# lousy defaults
DUE =none
CKPT=none

# numbers, play no role if associated var (without _FREQ) is 'none'
CKPT_FREQ=

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

FLAVOUR = -DDUE=$D -DCKPT=$C

DIR=$(DUE)

ifeq ($C,0)
DIR+=$(CKPT)
else 
DIR+=$(CKPT)$(CKPT_FREQ)
FLAVOUR+= -DCHECKPOINT_FREQ=$(CKPT_FREQ)
endif

empty=
space=$(empty) $(empty)

#####################################################################################################

# Finally defining directories and targets
DESTDIR:=$(subst $(space),_,$(DIR))

.PHONY : default clean all ompssall plainall debug ompssdebug plaindebug conv ompssconv plainconv speedup ompssspeedup plainspeedup
default:ompssall

all ompssall plainall debug ompssdebug plaindebug conv ompssconv plainconv speedup ompssspeedup plainspeedup: $(DESTDIR)
	cd $(DESTDIR) && make -f ../Makefile.sub "FLAVOUR=$(FLAVOUR)" VPATH=.:..:../src $(@)

clean:
	rm -rf async_* lossy_* none_* path_* rollback_*

.SECONDEXPANSION:
$(DESTDIR):
	mkdir -p $(DESTDIR)

