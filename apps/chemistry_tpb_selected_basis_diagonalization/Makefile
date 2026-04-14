include Configuration
SBD_INCLUDE_DIR=$(SBD_PATH)/include
LIBFLAGS= $(SYSLIB)
CUFLAGS ?=
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
	$(CCCOM) -c $(CCFLAGS) $(CUFLAGS) -I$(SBD_INCLUDE_DIR) $<
###############################################################
diag: clean main.o $(OBJECTS)
	$(CCCOM) $(CCFLAGS) -o diag main.o $(OBJECTS) $(LIBFLAGS)

clean:
	rm -f main.o

###############################################################
