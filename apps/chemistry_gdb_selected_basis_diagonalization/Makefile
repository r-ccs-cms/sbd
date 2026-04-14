include Configuration
SBD_INCLUDE_DIR=$(SBD_PATH)/include
LIBFLAGS= $(SYSLIB)
# LDFLAGS can be set in Configuration to override link-time flags (e.g. strip -x cu for nvcc)
LDFLAGS ?= $(CCFLAGS)
MAKEFILES= Makefile Configuration
# header file
HEADER=
# source
SOURCES= main.cc
#objects
OBJECTS=
# compilation
.SUFFIXES:
.SUFFIXES: .h .cc .o
.cc.o:  $*.cc
	$(CCCOM) -c $(CCFLAGS) -I$(SBD_INCLUDE_DIR) $<
###############################################################
diag: clean main.o $(OBJECTS)
	$(CCCOM) $(LDFLAGS) -o diag main.o $(OBJECTS) $(LIBFLAGS)

clean:
	rm -f main.o

###############################################################
