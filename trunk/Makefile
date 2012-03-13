# Binary file
BIN = main-EW

# Pathts & Includes
CUDA_INSTALL_PATH       ?= /usr
CUDASDK_INSTALL_PATH    = /home/ferrero/NVIDIA_GPU_Computing_SDK
CUDA_LIB_PATH           = $(CUDA_INSTALL_PATH)/lib
CUDASDK_LIB_PATH        = $(CUDASDK_INSTALL_PATH)/C/lib64
INCLUDES += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDASDK_INSTALL_PATH)/C/common/inc -I$(CUDASDK_INSTALL_PATH)/shared/inc \
-I/home/ferrero/cuda_libraries/

# Flags
LIBS                    = -L$(CUDA_LIB_PATH) -L$(CUDASDK_LIB_PATH)
CFLAGS                  = -O3 
#CFLAGS			= -O -g
LDFLAGS                 = -lrt -lm -lcudart -lcufft
NVCC_FLAGS              = -O3 -arch=sm_20 --ptxas-options=-v --use_fast_math -ftz=true 

# Default Values
SAMPLES			= 100
L			= 4096
M			= 4096
T_RUN			= 100000
T_DATA			= 10000
TEMP			= 0.f
C_BETA			= 1.0f
C_EPSILON		= 0.35f
H_START			= 0.05f
DHDT			= 0.f
T_START_FIELD		= 100000
DT			= 0.05f

# Simulation Parameters
PARAMETERS              = -DL=$(L) -DM=$(M) -DSAMPLES=$(SAMPLES) -DT_RUN=$(T_RUN) -DT_DATA=$(T_DATA) -DTEMP=$(TEMP) \
			-DC_BETA=$(C_BETA) -DC_EPSILON=$(C_EPSILON) \
			-DH_START=$(H_START) -DDHDT=$(DHDT) -DT_START_FIELD=$(T_START_FIELD) -DDT=$(DT)

# Compilers
NVCC                    = $(CUDA_INSTALL_PATH)/bin/nvcc
GCC                     = g++-4.6
LINKER                  = g++-4.6

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
#	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) valgrind --tool=memcheck --leak-check=full --track-origins=yes ./$(BIN)  
	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ./$(BIN)  

clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log
                                                         
