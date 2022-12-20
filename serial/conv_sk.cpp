#include <stdio.h>
#include <vector>
#include <sys/time.h>
using namespace std;

vector<vector<float> > img;
vector<vector<float> > tmp_img;
vector<vector<float> > kernel;//row 0: vertical kernel, row 1: horizontal kernel
vector<vector<float> > ans;
int width,height,bpp,pad;
#include "../common/tools.hpp"

float cov_sk(int row,int col,int turn){
    float output=0;
    if(!turn){//turn==0
        for(int i=0;i<kernel[turn].size();i++){
            output+=kernel[turn][i]*img[row+i-pad][col];
        }
    }else{//turn==1
        for(int i=0;i<kernel[turn].size();i++){
            output+=kernel[turn][i]*tmp_img[row][col+i-pad];
        }
    }
    
    return output;
}

void inittmp_img(){
    vector<float> tmp;
    for(int i=0;i<img.size();i++){
        tmp.clear();
        for(int j=0;j<img[0].size();j++){
            tmp.push_back(0);
        }
        tmp_img.push_back(tmp);
    }
}

int main(int argc, char *argv[]){

    char input_kernel_name[32], input_img_name[32];
    char input_kernel_fullname[256], input_img_fullname[256];
    
    if (argc > 2) {
        sprintf(input_img_name, "%s", argv[1]);
        sprintf(input_kernel_name, "%s", argv[2]);
    } else if (argc == 2) {
        sprintf(input_img_name, "%s", argv[1]);
        sprintf(input_kernel_name, "%s", "kernel3x3_sk.txt");
    } else {
        sprintf(input_img_name, "%s", "image.jpeg");
        sprintf(input_kernel_name, "%s", "kernel3x3_sk.txt");
    } 

    sprintf(input_img_fullname, "../common/image/%s", input_img_name);
    sprintf(input_kernel_fullname, "../common/kernel/%s", input_kernel_name);
    init_sk(input_img_fullname, input_kernel_fullname);

    inittmp_img();
    struct timeval start, end;
    gettimeofday(&start, 0);

    for(int T = 0; T < RUN_NUM; T++){
        for(int i=pad;i<img.size()-pad;i++){
            for(int j=pad;j<img[0].size()-pad;j++){
                tmp_img[i][j]=cov_sk(i,j,0);
            }
        }
        for(int i=pad;i<img.size()-pad;i++){
            for(int j=pad;j<img[0].size()-pad;j++){
                ans[i-pad][j-pad]=cov_sk(i,j,1);
            }
        }
    }
    
    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Elapsed time: %f sec\n", (sec+(usec/1000000.0))); 

    char ans_txt_name[256], out_txt_name[256], out_img_name[256];
    char *strip_input_img_name = strip_dot(input_img_name);
    char *strip_input_kernel_name = strip_dot(input_kernel_name);
    sprintf(ans_txt_name, "../serial/output/%s_%s.txt", strip_input_img_name, strip_input_kernel_name);
    sprintf(out_txt_name, "./output/%s_%s.txt", strip_input_img_name, strip_input_kernel_name);
    sprintf(out_img_name, "./output/%s_%s.jpeg", strip_input_img_name, strip_input_kernel_name);
    
    writeAns(out_txt_name);
    writeImage(out_img_name);
    checkAns(ans_txt_name, out_img_name);
    
    writeAns(out_txt_name);
    writeImage(out_img_name);
    return 0;
}