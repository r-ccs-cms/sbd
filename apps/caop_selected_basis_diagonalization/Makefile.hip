include Configuration.hip
SBD_INCLUDE_DIR=$(SBD_PATH)/include
HIPSHIM_DIR=$(SBD_PATH)/hipshim
LIBFLAGS= $(SYSLIB)
MAKEFILES= Makefile.hip Configuration.hip
# header file
HEADER=
# source
SOURCES= main.cc
#objects
OBJECTS=
# compilation: the source is HIP (-x hip); hipshim maps SBD's CUDA/Thrust/CUB/NCCL names onto ROCm
.SUFFIXES:
.SUFFIXES: .h .cc .o
.cc.o:  $*.cc
	$(CCCOM) -c -x hip $(CCFLAGS) -I$(HIPSHIM_DIR) -include $(HIPSHIM_DIR)/thrust_cuda.h -I$(SBD_INCLUDE_DIR) $<
###############################################################
# linking: no -x hip here, or the object file would be read as HIP source
diag: clean main.o $(OBJECTS)
	$(CCCOM) $(LDFLAGS) -o diag main.o $(OBJECTS) $(LIBFLAGS)

clean:
	rm -f main.o

###############################################################
