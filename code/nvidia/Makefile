
NVCC=nvcc
ARCH=sm_60

CUDA_PATH?=/usr/local/cuda
CUDA_LIB64=$(CUDA_PATH)/lib64
CUDNN_PATH?=/usr/local/cudnn
NCCL_PATH?=/usr/local/nccl
MPI_PATH?=/usr/local/openmpi
BIN_DIR?=bin
MKDIR=mkdir -p
#BLAS
BLAS_LIBRARY?=cublas
BLAS_PATH?=$(CUDA_LIB64)
#CONV
CONV_LIBRARY?=cudnn
CONV_PATH?=$
.PHONY=all gemm conv rnn all_reduce nccl_single nccl_mpi clean

all: gemm conv rnn all_reduce

gemm:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) gemm_bench.cu -o $(BIN_DIR)/gemm_bench -I $(CUDA_PATH)/include -L $(BLAS_PATH) -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand -arch=$(ARCH) -std=c++11

conv:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) conv_bench.cu -o $(BIN_DIR)/conv_bench -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcurand -lcudnn -arch=$(ARCH) -std=c++11

rnn:
	$(MKDIR) $(BIN_DIR)
	$(CUDA_PATH)/bin/$(NVCC) rnn_bench.cu -o $(BIN_DIR)/rnn_bench -I $(CUDA_PATH)/include -I $(CUDNN_PATH)/include/ -L $(CUDNN_PATH)/lib64/ -L $(CUDA_LIB64) -lcurand -lcudnn -arch=$(ARCH) -std=c++11

#all_reduce: nccl_single nccl_mpi

#nccl_single:
#	$(MKDIR) $(BIN_DIR)
#	$(CUDA_PATH)/bin/$(NVCC) nccl_single_all_reduce.cu -o $(BIN_DIR)/nccl_single_all_reduce -I $(NCCL_PATH)/build/include/ -I $(CUDNN_PATH)/include/ -L $(NCCL_PATH)/build/lib/ -L $(CUDNN_PATH)/lib64 -lnccl -lcurand -arch=$(ARCH) -std=c++11 

#nccl_mpi:
#	$(CUDA_PATH)/bin/$(NVCC) nccl_mpi_all_reduce.cu -o $(BIN_DIR)/nccl_mpi_all_reduce -I $(NCCL_PATH)/build/include/ -I $(CUDNN_PATH)/include/ -I $(MPI_PATH)/include -L $(NCCL_PATH)/build/lib/ -L $(CUDNN_PATH)/lib64 -L $(MPI_PATH)/lib -lnccl -lcurand -lmpi -arch=$(ARCH) -std=c++11 

clean:
	rm -rf $(BIN_DIR)

rebuild: clean all
