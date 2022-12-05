#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <vector>
using namespace std;

// image variables
vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width, height, bpp, pad;
#include "../common/tools.hpp"

// shared variables
long thread_cnt;

void compute_conv(long row_id, vector<vector<float> > &ans){
    for(int col_id = pad; col_id < width + pad; col_id++){
        float out = 0;

        for(int i = 0; i < kernel.size(); i++){
            for(int j = 0; j < kernel[0].size(); j++){
                out += (kernel[i][j] * img[row_id + i - pad][col_id + j - pad]);
            }
        }

        ans[row_id - pad][col_id - pad] = out;
    }
}

void* thread_conv_func(void *rank){
    long my_rank = (long) rank;

    for(long row_id = pad + my_rank; row_id < height + pad; row_id += thread_cnt){
        compute_conv(row_id, ans);
    }

    return NULL;
}

int main(int argc, char *argv[]){
    clock_t start, end;
    pthread_t* thread_handles;

    init();

    start = clock();

    thread_cnt = strtol(argv[1], NULL, 10);
    thread_handles = (pthread_t*) malloc (thread_cnt * sizeof(pthread_t));

    // create threads
    for(long thread = 0; thread < thread_cnt; thread++)
        pthread_create(&thread_handles[thread], NULL, thread_conv_func, (void*) thread);
    
    // wait for threads join
    for(long thread = 0; thread < thread_cnt; thread++)
        pthread_join(thread_handles[thread], NULL);

    end = clock();
    printf("Time = %f\n",((double)(end-start))/CLOCKS_PER_SEC);

    checkAns();

    return 0;
}