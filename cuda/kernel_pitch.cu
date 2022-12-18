#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

float *d_img, *d_ans, *d_kernel;
size_t img_pitch, ans_pitch, kernel_pitch;

__global__ void con(float *d_img, float *d_ans, float *d_kernel,
                    size_t img_pitch, size_t ans_pitch, size_t kernel_pitch,
                    int width, int height, int k_size, int pad) {
    
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r > height || c > width)
        return;

    float res = 0.0;
    for (int kr = -pad; kr <= pad; kr++) {
        for (int kc = -pad; kc <= pad; kc++) {
            float img_rc = *((float*)((char*)d_img + (r + pad + kr) * img_pitch) + (c + pad + kc));
            float kernel_rc = *((float*)((char*)d_kernel + (kr + pad) * kernel_pitch) + (kc + pad));
            res += img_rc * kernel_rc;
        }
    }
    *((float*)((char*)d_ans + r * ans_pitch) + c) = res;
}

void mallocKernelAndAns(float *kernel_arr, int width, int height, int k_size) {

    cudaMallocPitch((void **)&d_ans, &ans_pitch, width * sizeof(float), height);
    cudaMallocPitch((void **)&d_kernel, &kernel_pitch, k_size * sizeof(float), k_size);
    cudaMemcpy2D(d_kernel, kernel_pitch, 
                    kernel_arr, k_size * sizeof(float), 
                    k_size * sizeof(float), k_size, 
                    cudaMemcpyHostToDevice);

}

void convolution(float *img_arr, 
                 float *ans_arr,
                 int width, 
                 int height, 
                 int k_size, 
                 int pad) {

    // init cuda arr
    cudaMallocPitch((void **)&d_img, &img_pitch, (width + 2 * pad) * sizeof(float), height + 2 * pad);
    cudaMemcpy2D(d_img, img_pitch, 
                img_arr, (width + 2 * pad) * sizeof(float), 
                (width + 2 * pad) * sizeof(float), (height + 2 * pad), 
                cudaMemcpyHostToDevice);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
    con<<<numBlock, blockSize>>>(d_img, d_ans, d_kernel, 
                                 img_pitch, ans_pitch, kernel_pitch,
                                 width, height, k_size, pad);
    
    cudaMemcpy2D(ans_arr, width * sizeof(float), 
                 d_ans, ans_pitch, 
                 width * sizeof(float), height, 
                 cudaMemcpyDeviceToHost);

    cudaFree(d_img);
}

void freeKernelAndAns() {
    cudaFree(d_kernel);
    cudaFree(d_ans);
}