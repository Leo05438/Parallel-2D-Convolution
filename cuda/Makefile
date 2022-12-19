NVCC = nvcc
CXX = g++
CXXFLAGS=-I./common -Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm
CXXINCS = -I ../common

# CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
# CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_86,code=sm_86 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_86,code=sm_86 -Xcompiler '-fPIC' -g -O3

EXE = conv_basic.out conv_pitch.out conv_tiling.out conv_tiling_modified.out conv_tiling+pitch.out conv_sk_basic.out

OBJS = cuda.o cuda_sk.o kernel_basic.o kernel_pitch.o kernel_tiling.o kernel_tiling_modified.o kernel_tiling+pitch.o kernel_sk_basic.o

all: $(EXE)
	rm $(OBJS)

# for general kernel
conv_basic.out: kernel_basic.o cuda.o
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $^

conv_pitch.out: kernel_pitch.o cuda.o
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $^

conv_tiling.out: kernel_tiling.o cuda.o
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $^

conv_tiling_modified.out: kernel_tiling_modified.o cuda.o
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $^

conv_tiling+pitch.out: kernel_tiling+pitch.o cuda.o
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $^

kernel_basic.o: kernel_basic.cu
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c $< -o $@

kernel_pitch.o: kernel_pitch.cu
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c $< -o $@

kernel_tiling.o: kernel_tiling.cu
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c $< -o $@

kernel_tiling_modified.o: kernel_tiling_modified.cu
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c $< -o $@

kernel_tiling+pitch.o: kernel_tiling+pitch.cu
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c $< -o $@

# for sk kernel
conv_sk_basic.out: kernel_sk_basic.o cuda_sk.o
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ $^

kernel_sk_basic.o: kernel_sk_basic.cu
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c $< -o $@

# for cpp files
cuda.o: cuda.cpp
	$(CXX) -c $< $(CXXFLAGS) $(CXXINCS) -o $@

cuda_sk.o: cuda_sk.cpp
	$(CXX) -c $< $(CXXFLAGS) $(CXXINCS) -o $@

.PHONY: clean

clean:                             
	rm -rf $(EXE) $(OBJS)