#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

float *d_img, *d_ans, *d_kernel;

__global__ void conv0(float *d_ans, float *d_img, float *d_kernel_0,
                    int width, int height, int k_size, int pad) {
    
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= height || c >= width)
        return;

    float res = 0.0;
    for (int ki = -pad; ki <= pad; ki++) {
        if (r + ki >= 0 && r + ki < height)
            res += d_img[(r + ki) * width + c] * d_kernel_0[(ki + pad)];
    }
    if (r == height - 1 && c == 0) printf("res 0 : %f\n", res);
    d_ans[r * width + c] = res;
}

__global__ void conv1(float *d_ans, float *d_img, float *d_kernel_1,
                    int width, int height, int k_size, int pad) {
    
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= height || c >= width)
        return;
    
    float res = 0.0;
    for (int ki = -pad; ki <= pad; ki++) {
        if (c + ki >= 0 && c + ki < width)
            res += d_img[r * width + (c + ki)] * d_kernel_1[(ki + pad)];
    }
    if (r == height - 1 && c == 0) printf("res 1 : %f\n", res);
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
    conv0<<<numBlock, blockSize>>>(d_ans, d_img, (d_kernel), width, height, k_size, pad);
    conv1<<<numBlock, blockSize>>>(d_img, d_ans, (d_kernel + k_size), width, height, k_size, pad);
    
    cudaMemcpy(ans_arr, d_img, width * height * sizeof(float), cudaMemcpyDeviceToHost);
}

void freeKernelAndAns() {
    cudaFree(d_img);
    cudaFree(d_ans);
    cudaFree(d_kernel);
}