#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include "kernel.h"

using namespace std;

vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width,height,bpp,pad;
#include "../common/tools.hpp"

float* vec2arr(const vector<vector<float> >& vec) {
    int rows = vec.size();
    int cols = vec[0].size();
    float *ret = new float[rows * cols];
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            ret[r * cols + c] = vec[r][c];
        }
    }
    return ret;
}

void arr2vec(vector<vector<float> >& vec, float *arr) {
    int rows = vec.size();
    int cols = vec[0].size();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            vec[r][c] = arr[r * cols + c];
        }
    }
}

void display(float* arr, int width, int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            printf("%f ", arr[r * width + c]);
        }
        printf("\n");
    }
}

void display(const vector<vector<float> >& vec) {
    int rows = vec.size();
    int cols = vec[0].size();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%f ", vec[r][c]);
        }
        printf("\n");
    }
}

int main(){

    init();

    float *img_arr, *ans_arr, *kernel_arr;
    int k_size = kernel.size();

    img_arr = vec2arr(img);
    ans_arr = vec2arr(ans);
    kernel_arr = vec2arr(kernel);

    mallocKernelAndAns(kernel_arr, width, height, k_size);

    struct timeval start, end;
    gettimeofday(&start, 0);

    // init cuda mem
    for(int T = 0; T < 5000; T++) {
        convolution(img_arr, ans_arr, width, height, k_size, pad);
    }

    gettimeofday(&end, 0);
    freeKernelAndAns();

    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Elapsed time: %f sec\n", (sec + (usec / 1000000.0))); 

    arr2vec(ans, ans_arr);
    writeImage();
    writeAns();
    checkAns();

    free(img_arr);
    free(ans_arr);
    free(kernel_arr);
    return 0;
}