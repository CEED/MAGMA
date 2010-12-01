include ../make.inc

all: 	lib/libquark.a

lib: 	lib/libquark.a

clean:
	rm -f *.o *~

cleanall: clean
	rm -f lib/libquark.a

.c.o:
	$(CC) $(OPTS) -Iinclude $(INC) -c $< -o $@

lib/libquark.a: icl_hash.o icl_list.o quarkos.o quark.o
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@

.PHONY: clean

