#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <vector>
#include <queue>
#include <sys/time.h>
#include <semaphore.h>
using namespace std;

struct Jobs{
    int row_id, col_id;

    Jobs(): row_id(-1), col_id(-1) {}
    Jobs(const int &row, const int &col): row_id(row), col_id(col) {}

    Jobs& operator =(const Jobs& a){
        row_id = a.row_id;
        col_id = a.col_id;
        return *this;
    }
};

// image variables
vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width, height, bpp, pad;
#include "../common/tools.hpp"

// shared variables
long thread_cnt;
queue<Jobs> job_queue;
sem_t event_new_job, mutex_job_queue;

inline void compute_conv(int row_id, int col_id, vector<vector<float> > &ans){
    float out = 0;

    for(int i = 0; i < kernel.size(); i++){
        for(int j = 0; j < kernel[0].size(); j++){
            out += (kernel[i][j] * img[row_id + i - pad][col_id + j - pad]);
        }
    }

    ans[row_id - pad][col_id - pad] = out;
}

void* worker_thread_func(void*){
    Jobs new_job;

    while(true){
        sem_wait(&mutex_job_queue);

        if(!job_queue.empty()){   
            new_job = job_queue.front();
            job_queue.pop();  

            sem_post(&mutex_job_queue);

            compute_conv(new_job.row_id, new_job.col_id, ans);
        }
        else{
            sem_post(&mutex_job_queue);
            break;
        }
    }

    return NULL;
}

void assign_jobs(){
    for(int row_id = pad; row_id < height + pad; row_id++){
        for(int col_id = pad; col_id < width + pad; col_id++){
            job_queue.push(Jobs(row_id, col_id));
        }
    }
}

int main(int argc, char *argv[]){
    pthread_t* thread_handles;
    struct timeval start, end;

    if(argc < 2){
        printf("Usage: ./pthread <# threads>\n");
        exit(0);
    }

    char input_kernel_name[32] = {0}, input_img_name[32] = {0};
    char input_kernel_fullname[256] = {0}, input_img_fullname[256] = {0};
    
    if (argc > 3) {
        sprintf(input_img_name, "%s", argv[2]);
        sprintf(input_kernel_name, "%s", argv[3]);
    } else if (argc == 3) {
        sprintf(input_img_name, "%s", argv[2]);
        sprintf(input_kernel_name, "%s", "kernel3x3.txt");
    } else {
        sprintf(input_img_name, "%s", "image.jpeg");
        sprintf(input_kernel_name, "%s", "kernel3x3.txt");
    } 

    sprintf(input_img_fullname, "../common/image/%s", input_img_name);
    sprintf(input_kernel_fullname, "../common/kernel/%s", input_kernel_name);
    init(input_img_fullname, input_kernel_fullname);

    sem_init(&event_new_job, 0, 0);
    sem_init(&mutex_job_queue, 0, 1);

    gettimeofday(&start, 0);

    thread_cnt = strtol(argv[1], NULL, 10);
    thread_handles = (pthread_t*) malloc (thread_cnt * sizeof(pthread_t));

    for(int T = 0; T < RUN_NUM; T++){
        // assign jobs
        assign_jobs();

        // create worker threads
        for(long thread = 0; thread < thread_cnt; thread++)
            pthread_create(&thread_handles[thread], NULL, worker_thread_func, NULL);
        
        // wait for threads join
        for(long thread = 0; thread < thread_cnt; thread++)
            pthread_join(thread_handles[thread], NULL);
    }

    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Elapsed time: %f sec\n", (sec+(usec/1000000.0))); 

    sem_destroy(&event_new_job);
    sem_destroy(&mutex_job_queue);

    char ans_txt_name[256], out_txt_name[256], out_img_name[256];
    char *strip_input_img_name = strip_dot(input_img_name);
    char *strip_input_kernel_name = strip_dot(input_kernel_name);
    sprintf(ans_txt_name, "../serial/output/%s_%s.txt", strip_input_img_name, strip_input_kernel_name);
    sprintf(out_txt_name, "./output/%s_%s.txt", strip_input_img_name, strip_input_kernel_name);
    sprintf(out_img_name, "./output/%s_%s.jpeg", strip_input_img_name, strip_input_kernel_name);
    
    writeAns(out_txt_name);
    writeImage(out_img_name);
    checkAns(ans_txt_name, out_img_name);

    return 0;
}