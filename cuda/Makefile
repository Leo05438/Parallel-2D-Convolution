NVCC = nvcc
CXX = g++
CXXFLAGS=-I./common -Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm
CXXINCS = -I ../common

# CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
# CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

EXE = cuda_basic cuda_pitch cuda_tiling
CPP = pthread.cpp

OBJS = cuda.o kernel_basic.o kernel_pitch.o kernel_tiling.o

all: $(EXE)
	rm $(OBJS)

cuda_basic: kernel_basic.o cuda.o
	$(NVCC) -o $@ $^

cuda_pitch: kernel_pitch.o cuda.o
	$(NVCC) -o $@ $^

cuda_tiling: kernel_tiling.o cuda.o
	$(NVCC) -o $@ $^

cuda.o: cuda.cpp
	$(CXX) -c $< $(CXXFLAGS) $(CXXINCS) -o $@

kernel_basic.o: kernel_basic.cu
	$(NVCC) -c $< -o $@

kernel_pitch.o: kernel_pitch.cu
	$(NVCC) -c $< -o $@

kernel_tiling.o: kernel_tiling.cu
	$(NVCC) -c $< -o $@

.PHONY: clean

clean:                             
	rm -rf $(EXE) $(OBJS)