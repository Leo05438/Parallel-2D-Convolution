#include <stdio.h>
#include <time.h>
#include <vector>
using namespace std;

vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width,height,bpp,pad;
#include "../common/tools.hpp"

float cov(int row,int col){
    float output=0;
    for(int i=0;i<kernel.size();i++){
        for(int j=0;j<kernel[0].size();j++){
            output+=kernel[i][j]*img[row+i-pad][col+j-pad];
        }
    }
    return output;
}

int main(){
    init();

    clock_t start, end;
    start=clock();
    for(int i=pad;i<img.size()-pad;i++){
        for(int j=pad;j<img[0].size()-pad;j++){
            ans[i-pad][j-pad]=cov(i,j);
        }
    }
    end=clock();
    printf("Time = %f\n",((double)(end-start))/CLOCKS_PER_SEC);

    writeAns();
    return 0;
}