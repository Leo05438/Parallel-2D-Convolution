#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_CACHE_WIDTH 64
#define KERNEL_CACHE_WIDTH 16
#define BLOCK_SIZE 32

float *d_img, *d_ans, *d_kernel;

__global__ void convCol(float *d_ans, float *d_img, float *d_kernel_0,
                    int width, int height, int k_size, int pad) {
    
    int r = blockIdx.y * blockDim.y + threadIdx.y; // for padded image
    int c = blockIdx.x * blockDim.x + threadIdx.x; // for padded image

    if (r >= height || c >= width)
        return;

    __shared__ float shared_patch[IMAGE_CACHE_WIDTH][IMAGE_CACHE_WIDTH];
    __shared__ float shared_kernel[KERNEL_CACHE_WIDTH];

    // tiling : load patch and kernel to shared memory in a SM
    shared_patch[threadIdx.y][threadIdx.x]             = (r - pad >= 0) ? d_img[(r - pad) * width + c] : 0.0;
    shared_patch[threadIdx.y + pad][threadIdx.x]       = d_img[r * width + c];
    shared_patch[threadIdx.y + pad + pad][threadIdx.x] = (r + pad < height) ? d_img[(r + pad) * width + c] : 0.0;

    shared_kernel[threadIdx.y % k_size] = d_kernel_0[(threadIdx.y % k_size)];
    __syncthreads();
    
    float res = 0.0;
    for (int ki = -pad; ki <= pad; ki++) {
        res += shared_patch[threadIdx.y + pad + ki][threadIdx.x] * shared_kernel[ki + pad];
    }
    d_ans[r * width + c] = res;
}

__global__ void convRow(float *d_ans, float *d_img, float *d_kernel_1,
                    int width, int height, int k_size, int pad) {
    
    int r = blockIdx.y * blockDim.y + threadIdx.y; // for padded image
    int c = blockIdx.x * blockDim.x + threadIdx.x; // for padded image

    if (r >= height || c >= width)
        return;

    __shared__ float shared_patch[IMAGE_CACHE_WIDTH][IMAGE_CACHE_WIDTH];
    __shared__ float shared_kernel[KERNEL_CACHE_WIDTH];

    // tiling : load patch and kernel to shared memory in a SM
    shared_patch[threadIdx.y][threadIdx.x]             = (c - pad >= 0) ? d_img[r * width + c - pad] : 0.0;
    shared_patch[threadIdx.y][threadIdx.x + pad]       = d_img[r * width + c];
    shared_patch[threadIdx.y][threadIdx.x + pad + pad] = (c + pad < width) ? d_img[r * width + c + pad] : 0.0;

    shared_kernel[threadIdx.y % k_size] = d_kernel_1[(threadIdx.y % k_size)];
    __syncthreads();
    
    float res = 0.0;
    for (int ki = -pad; ki <= pad; ki++) {
        res += shared_patch[threadIdx.y][threadIdx.x + pad + ki] * shared_kernel[ki + pad];
    }
    d_ans[r * width + c] = res;
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
    dim3 numBlock(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
    convCol<<<numBlock, blockSize>>>(d_ans, d_img, (d_kernel), width, height, k_size, pad);
    convRow<<<numBlock, blockSize>>>(d_img, d_ans, (d_kernel + k_size), width, height, k_size, pad);
    
    cudaMemcpy(ans_arr, d_img, width * height * sizeof(float), cudaMemcpyDeviceToHost);
}

void freeKernelAndAns() {
    cudaFree(d_img);
    cudaFree(d_ans);
    cudaFree(d_kernel);
}