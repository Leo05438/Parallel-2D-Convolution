#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <vector>
#include <sys/time.h>
using namespace std;

// image variables
vector<vector<float> > img;
vector<vector<float> > tmp_img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width, height, bpp, pad;
#include "../common/tools.hpp"

// shared variables
long thread_cnt;
pthread_barrier_t mybarrier;

void compute_conv(long row_id, vector<vector<float> > &out_img, int turn){
    for(int col_id = pad; col_id < width + pad; col_id++){
        float out = 0;

        if(turn == 0){
            for(int i = 0; i < kernel[turn].size(); i++){
                out += (kernel[turn][i] * img[row_id + i - pad][col_id]);
            }

            out_img[row_id][col_id] = out;
        }
        else{
            for(int i = 0; i < kernel[turn].size(); i++){
                out += (kernel[turn][i] * tmp_img[row_id][col_id + i - pad]);
            }

            out_img[row_id - pad][col_id - pad] = out;
        }
    }
}

void* thread_conv_func(void *rank){
    long my_rank = (long) rank;

    for(long row_id = pad + my_rank; row_id < height + pad; row_id += thread_cnt){
        compute_conv(row_id, tmp_img, 0);
    }

    pthread_barrier_wait(&mybarrier);

    for(long row_id = pad + my_rank; row_id < height + pad; row_id += thread_cnt){
        compute_conv(row_id, ans, 1);
    }

    return NULL;
}

void inittmp_img(){
    vector<float> tmp;
    for(int i = pad; i < img.size(); i++){
        tmp.clear();
        for(int j = pad; j < img[0].size(); j++){
            tmp.push_back(0);
        }
        tmp_img.push_back(tmp);
    }
}

int main(int argc, char *argv[]){
    pthread_t* thread_handles;
    struct timeval start, end;

    if(argc != 2){
        printf("Usage: ./pthread <# threads>\n");
        exit(0);
    }

    init_sk();
    inittmp_img();

    gettimeofday(&start, 0);

    thread_cnt = strtol(argv[1], NULL, 10);
    thread_handles = (pthread_t*) malloc (thread_cnt * sizeof(pthread_t));
    pthread_barrier_init(&mybarrier, NULL, thread_cnt);

    for(int T = 0; T < 500; T++){
        // create threads
        for(long thread = 0; thread < thread_cnt; thread++)
            pthread_create(&thread_handles[thread], NULL, thread_conv_func, (void*) thread);
        
        // wait for threads join
        for(long thread = 0; thread < thread_cnt; thread++)
            pthread_join(thread_handles[thread], NULL);
    }

    pthread_barrier_destroy(&mybarrier);

    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Elapsed time: %f sec\n", (sec+(usec/1000000.0))); 

    checkAns();

    return 0;
}