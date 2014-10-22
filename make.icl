#
# special make rules for internal ICL use only
#

CFLAGS   += -DUSE_FLOCK

ifneq ($(CXXFLAGS),)
    CXXFLAGS += -DUSE_FLOCK
endif
