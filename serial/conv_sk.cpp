#include <stdio.h>
#include <vector>
#include <sys/time.h>
using namespace std;

vector<vector<float> > img;
vector<vector<float> > kernel;//row 0: vertical kernel, row 1: herizontal kernel
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
            output+=kernel[turn][i]*img[row][col+i-pad];
        }
    }
    
    return output;
}

int main(){
    init_sk();
    struct timeval start, end;
    gettimeofday(&start, 0);

    for(int T = 0; T < 500; T++){
        for(int i=pad;i<img.size()-pad;i++){
            for(int j=pad;j<img[0].size()-pad;j++){
                ans[i-pad][j-pad]=cov_sk(i,j,0);
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

    writeAns();
    writeImage();
    return 0;
}