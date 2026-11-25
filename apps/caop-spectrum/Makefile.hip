include Configuration.hip
HIPSHIM_DIR=$(SBD_PATH)/hipshim
INCLUDES := -I$(HIPSHIM_DIR) -include $(HIPSHIM_DIR)/thrust_cuda.h -I$(SBD_PATH)/include
.PHONY: all clean
all: caop-spectrum
# compile as HIP (-x hip), then -x none so that the libraries are not read as HIP source
caop-spectrum: main.cc cli_options.h
	$(CCCOM) -x hip $(CCFLAGS) $(INCLUDES) $< -x none -o $@ $(LDFLAGS) $(SYSLIB)
clean:
	rm -f caop-spectrum
