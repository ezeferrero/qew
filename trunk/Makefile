# Binary file
BIN = main-QEW

# Pathts & Includes
CUDA_INSTALL_PATH	= /usr/local/cuda
#CUDA_INSTALL_PATH	?= /usr
CUDASDK_INSTALL_PATH	= /opt/cudasdk
CUDA_LIB_PATH		= $(CUDA_INSTALL_PATH)/lib
CUDASDK_LIB_PATH	= $(CUDASDK_INSTALL_PATH)/C/lib
INCLUDES += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDASDK_INSTALL_PATH)/C/common/inc -I$(CUDASDK_INSTALL_PATH)/shared/inc 
# -I/home/ferrero/cuda_libraries/

# Flags
LIBS			= -L$(CUDA_LIB_PATH) -L$(CUDASDK_LIB_PATH)
CFLAGS			= -O3 
#CFLAGS			= -O -g
LDFLAGS			= -lrt -lm -lcudart -lcufft 
NVCC_FLAGS		= -O3 -arch=sm_20 --use_fast_math #-ftz=true --ptxas-options=-v 

# Default Values
SAMPLES			= 1000
L			= 1024
M			= 2048
#T_RUN=2^T_RUN_EXP
T_RUN_EXP		= 16
T_RUN			= 65536
T_DATA			= 1024
SPACING			= LOGARITHMIC
#SPACING			= LINEAR
TEMP			= 0.f
C_BETA			= 1.f
C_EPSILON		= 1.f
H_START			= 1.294f
DT			= 0.025f
CUDA_DEVICE		= 0
OFILE			= [EW,$(L),$(M),$(SAMPLES),$(T_RUN),$(T_DATA),$(TEMP),$(C_BETA),$(C_EPSILON),$(H_START),$(DT),GPU$(CUDA_DEVICE)].dat

# Simulation Parameters & Compilation Options
PARAMETERS		= -DLL=$(L) -DMM=$(M) -DSAMPLES=$(SAMPLES) -DT_RUN_EXP=$(T_RUN_EXP) -DT_RUN=$(T_RUN) -DT_DATA=$(T_DATA) \
			-DTEMP=$(TEMP) -DC_BETA=$(C_BETA) -DC_EPSILON=$(C_EPSILON) \
			-DH_START=$(H_START) -DDT=$(DT) -D$(SPACING) -DCUDA_DEVICE=$(CUDA_DEVICE) \
			-DZEROTEMP #-DLINEAR_SPLINE #-DRF #-DUNIFORM_DISORDER #-DIGNORE_SF #-DANHARMONIC

# Compilers
NVCC                    = $(CUDA_INSTALL_PATH)/bin/nvcc
GCC                     = g++
LINKER                  = g++

# Files
C_SOURCES               =
CU_SOURCES              = $(BIN).cu
HEADERS                 =
C_OBJS                  = $(patsubst %.c, %.o, $(C_SOURCES))
CU_OBJS                 = $(patsubst %.cu, %.o, $(CU_SOURCES))

# Rules
$(BIN): clean $(C_OBJS) $(CU_OBJS) $(HEADERS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(C_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

$(C_OBJS): $(C_SOURCES) $(HEADERS)
	$(CC) -c $(C_SOURCES) $(CFLAGS) $(INCLUDES)

$(CU_OBJS): $(CU_SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $(CU_SOURCES) $(PARAMETERS) $(INCLUDES)

run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ./$(BIN) > $(OFILE) &
#	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) valgrind --tool=memcheck --leak-check=full --track-origins=yes ./$(BIN)  

clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log
                                                     
