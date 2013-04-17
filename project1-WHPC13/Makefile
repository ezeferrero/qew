# Paths & Includes
CUDA_INSTALL_PATH       = /usr/local/cuda-5.0
CUDA_LIB_PATH           = $(CUDA_INSTALL_PATH)/lib64
INCLUDES += -I$(CUDA_INSTALL_PATH)/include -I./include/ 

# Flags
LIBS                    = -L$(CUDA_LIB_PATH) 
LDFLAGS                 = -lrt -lm -lcudart -lcufft 
NVCC_FLAGS              = -O2 --use_fast_math #-DCUFFT 
OMP_FLAGS              = -O2 -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP -lgomp


NVCC                    = $(CUDA_INSTALL_PATH)/bin/nvcc -arch=sm_21
GCC			= g++
LINKER			= $(NVCC)

# Rules
all: qew_CUDA qew_OMP

qew_cuda.o: qew_minimal.cu qew_cufft.h

qew_omp.o: qew_minimal.cpp

qew_CUDA: qew_cuda.o qew_cufft.h
	$(LINKER) $(NVCC_FLAGS) -o qew_CUDA qew_minimal.cu $(LDFLAGS) $(INCLUDES) $(LIBS)
	  
qew_OMP: qew_omp.o
	$(GCC) -o qew_OMP qew_minimal.cpp $(OMP_FLAGS) $(INCLUDES)
				 
clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log qew_CUDA qew_OMP
                                                     
