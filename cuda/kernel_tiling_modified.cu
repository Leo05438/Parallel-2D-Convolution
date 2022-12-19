#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_CACHE_WIDTH 64
#define KERNEL_CACHE_WIDTH 16
#define BLOCK_SIZE 32

float *d_img, *d_ans, *d_kernel;

__global__ void conv(float *d_ans, float *d_img, float *d_kernel, 
                     int width, int height, int k_size, int pad) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int r_pad = r + pad;
    const int r_2pad = r + pad + pad;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int c_pad = c + pad;
    const int c_2pad = c + pad + pad;
    const int x = threadIdx.x;
    const int x_pad = x + pad;
    const int x_2pad = x + pad + pad;
    const int y = threadIdx.y;
    const int y_pad = y + pad;
    const int y_2pad = y + pad + pad;

    int padded_img_width = width + 2 * pad;

    if (r >= height || c >= width)
        return;
    
    __shared__ float shared_patch[IMAGE_CACHE_WIDTH][IMAGE_CACHE_WIDTH];
    __shared__ float shared_kernel[KERNEL_CACHE_WIDTH][KERNEL_CACHE_WIDTH];

    // tiling : load patch and kernel to shared memory in a SM
    shared_patch[y_pad][x_pad] = d_img[(r_pad) * padded_img_width + c_pad];

    // left part
    // if (x < pad)
    shared_patch[y_pad][x] = d_img[(r_pad) * padded_img_width + c];
    // right part (edge case)
    // if ((x >= BLOCK_SIZE - pad) || (c_pad >= width))
    shared_patch[y_pad][x_2pad] = d_img[(r_pad) * padded_img_width + c_2pad];
    // top part
    // if (y < pad)
    shared_patch[y][x_pad] = d_img[r * padded_img_width + c_pad];
    // bottom part (edge case)
    // if ((y >= BLOCK_SIZE - pad) || (r_pad >= height))
    shared_patch[y_2pad][x_pad] = d_img[(r_2pad) * padded_img_width + c_pad];
    // top-left part
    // if (y < pad && x < pad)
    shared_patch[y][x] = d_img[r * padded_img_width + c];
    // top-right part (edge case)
    // if ((y < pad && x >= BLOCK_SIZE - pad) || (c_pad >= width))
    shared_patch[y][x_2pad] = d_img[r * padded_img_width + c_2pad];
    // bottom-left part (edge case)
    // if ((y >= BLOCK_SIZE - pad && x < pad) || (r_pad >= height))
    shared_patch[y_2pad][x] = d_img[(r_2pad) * padded_img_width + c];
    // bottom-right part (edge case)
    // if ((y >= BLOCK_SIZE - pad && x >= BLOCK_SIZE - pad) || (r_pad >= height || c_pad >= width))
    shared_patch[y_2pad][x_2pad] = d_img[(r_2pad) * padded_img_width + c_2pad];

    shared_kernel[y % k_size][x % k_size] = d_kernel[(y % k_size) * k_size + x % k_size];
    __syncthreads();
    
    float res = 0.0;
    for (int kr = -pad; kr <= pad; kr++) {
        for (int kc = -pad; kc <= pad; kc++) {
            res += shared_patch[y + kr + pad][x + kc + pad] * shared_kernel[kr + pad][kc + pad];
        }
    }

    d_ans[r * width + c] = res;
}

void mallocKernelAndAns(float *kernel_arr, int width, int height, int k_size, int pad) {

    cudaMalloc((void **)&d_img, (width + 2 * pad) * (height + 2 * pad) * sizeof(float));
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
    cudaMemcpy(d_img, img_arr, (width + 2 * pad) * (height + 2 * pad) * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
    conv<<<numBlock, blockSize>>>(d_ans, d_img, d_kernel, width, height, k_size, pad);

    cudaMemcpy(ans_arr, d_ans, width * height * sizeof(float), cudaMemcpyDeviceToHost);

}

void freeKernelAndAns() {
    cudaFree(d_img);
    cudaFree(d_ans);
    cudaFree(d_kernel);
}