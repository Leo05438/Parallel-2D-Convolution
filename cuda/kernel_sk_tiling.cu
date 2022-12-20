#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_CACHE_WIDTH 64
#define KERNEL_CACHE_WIDTH 16
#define BLOCK_SIZE 32

float *d_img, *d_ans, *d_kernel;

__global__ void convCol(float *d_ans, float *d_img, float *d_kernel_0,
                    int width, int height, int k_size, int pad) {
    
    int r = blockIdx.y * (blockDim.y - 2 * pad) + threadIdx.y; // for padded image
    int c = blockIdx.x * blockDim.x + threadIdx.x; // for padded image

    if (r >= height + 2 * pad || c >= width)
        return;

    __shared__ float shared_patch[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_kernel[BLOCK_SIZE];

    // tiling : load patch and kernel to shared memory in a SM
    if (r < pad || r >= height + pad)
        shared_patch[threadIdx.y][threadIdx.x] = 0.0;
    else 
        shared_patch[threadIdx.y][threadIdx.x] = d_img[(r - pad) * width + c];

    shared_kernel[threadIdx.y % k_size] = d_kernel_0[(threadIdx.y % k_size)];
    __syncthreads();

    if (threadIdx.y < pad || threadIdx.y >= BLOCK_SIZE - pad || r >= height + pad)
        return;
    
    float res = 0.0;
    int ans_r = r - pad; 
    int ans_c = c; 
    for (int ki = -pad; ki <= pad; ki++) {
        res += shared_patch[threadIdx.y + ki][threadIdx.x] * shared_kernel[ki + pad];
    }
    d_ans[ans_r * width + ans_c] = res;
}

__global__ void convRow(float *d_ans, float *d_img, float *d_kernel_1,
                    int width, int height, int k_size, int pad) {
    
    int r = blockIdx.y * blockDim.y + threadIdx.y; // for padded image
    int c = blockIdx.x * (blockDim.x - 2 * pad) + threadIdx.x; // for padded image

    if (r >= height || c >= width + 2 * pad)
        return;

    __shared__ float shared_patch[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_kernel[BLOCK_SIZE];

    // tiling : load patch and kernel to shared memory in a SM
    if (c < pad || c >= width + pad)
        shared_patch[threadIdx.y][threadIdx.x] = 0.0;
    else 
        shared_patch[threadIdx.y][threadIdx.x] = d_img[r * width + (c - pad)];
    
    shared_kernel[threadIdx.y % k_size] = d_kernel_1[(threadIdx.y % k_size)];
    __syncthreads();

    if (threadIdx.x < pad || threadIdx.x >= BLOCK_SIZE - pad || c >= width + pad)
        return;
    
    float res = 0.0;
    int ans_r = r; 
    int ans_c = c - pad; 
    for (int ki = -pad; ki <= pad; ki++) {
        res += shared_patch[threadIdx.y][threadIdx.x + ki] * shared_kernel[ki + pad];
    }
    d_ans[ans_r * width + ans_c] = res;
}

void mallocKernelAndAns(float *kernel_arr, int width, int height, int k_size, int pad) {

    cudaMalloc((void **)&d_img, width * height * sizeof(float));
    cudaMalloc((void **)&d_ans, width * height * sizeof(float));
    cudaMalloc((void **)&d_kernel, k_size * k_size * sizeof(float));
    cudaMemcpy(d_kernel, kernel_arr, k_size * k_size * sizeof(float), cudaMemcpyHostToDevice);

}

void convolution(float *img_arr, 
                 float *ans_arr,
                 int width, 
                 int height, 
                 int k_size, 
                 int pad) {

    // init cuda arr
    cudaMemcpy(d_img, img_arr, width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlockRow(width / (BLOCK_SIZE - 2 * pad) + 1, height / BLOCK_SIZE + 1);
    dim3 numBlockCol(width / BLOCK_SIZE + 1, height / (BLOCK_SIZE - 2 * pad) + 1);
    convCol<<<numBlockCol, blockSize>>>(d_ans, d_img, (d_kernel), width, height, k_size, pad);
    convRow<<<numBlockRow, blockSize>>>(d_img, d_ans, (d_kernel + k_size), width, height, k_size, pad);
    
    cudaMemcpy(ans_arr, d_img, width * height * sizeof(float), cudaMemcpyDeviceToHost);
}

void freeKernelAndAns() {
    cudaFree(d_img);
    cudaFree(d_ans);
    cudaFree(d_kernel);
}