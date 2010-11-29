
include ../make.inc

all: 	libquark.a

lib: 	libquark.a

clean:
	rm -f *.o *~

cleanall: clean
	rm -f *.a

.c.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

libquark.a: icl_hash.o icl_list.o quarkos.o quark.o
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@

.PHONY: clean

