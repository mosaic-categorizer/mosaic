darshan-logutils-make-shared-object:
	mkdir darshan-install
	cd darshan-install && wget https://github.com/darshan-hpc/darshan/archive/refs/tags/darshan-2.3.1.tar.gz && tar -xvzf darshan-2.3.1.tar.gz
	cd darshan-install/darshan-darshan-2.3.1/darshan-util && ./prepare && ./configure
	cd darshan-install/darshan-darshan-2.3.1/darshan-util && echo -e 'darshan-logutils.so: darshan-logutils.c\n\t$$(CC) $$(CFLAGS_SHARED)  $$< -o $$@ $$(LIBS)' >> Makefile && make darshan-logutils.so
	mv darshan-install/darshan-darshan-2.3.1/darshan-util/darshan-logutils.so .
	rm -rf darshan-install
