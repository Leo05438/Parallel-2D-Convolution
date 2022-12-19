#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <vector>
#include <sys/time.h>
using namespace std;

// image variables
vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width, height, bpp, pad;
#include "../common/tools.hpp"

// shared variables
long thread_cnt;

void* thread_conv_func(void *rank){
    long my_rank = (long) rank;
    int step = height / thread_cnt;
    long long int my_first_i = pad + step * my_rank;
    long long int my_last_i = (my_rank == thread_cnt - 1) ? height + pad : my_first_i + step;

    for(long row_id = my_first_i; row_id < my_last_i; row_id++){
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

    return NULL;
}

int main(int argc, char *argv[]){
    pthread_t* thread_handles;
    struct timeval start, end;

    if(argc != 2){
        printf("Usage: ./pthread <# threads>\n");
        exit(0);
    }

    init();

    gettimeofday(&start, 0);

    thread_cnt = strtol(argv[1], NULL, 10);
    thread_handles = (pthread_t*) malloc (thread_cnt * sizeof(pthread_t));

    for(int T = 0; T < 500; T++){
        // create threads
        for(long thread = 0; thread < thread_cnt; thread++)
            pthread_create(&thread_handles[thread], NULL, thread_conv_func, (void*) thread);
        
        // wait for threads join
        for(long thread = 0; thread < thread_cnt; thread++)
            pthread_join(thread_handles[thread], NULL);
    }

    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Elapsed time: %f sec\n", (sec+(usec/1000000.0))); 

    checkAns();

    return 0;
}