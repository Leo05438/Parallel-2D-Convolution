#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <vector>
#include "../common/tools.hpp"
using namespace std;

float cov(int row,int col){
    float output=0;
    for(int i=0;i<kernel.size();i++){
        for(int j=0;j<kernel[0].size();j++){
            output+=kernel[i][j]*img[row+i-1][col+j-1];
        }
    }
    return output;
}

int main(){
    clock_t start, end;
    init();

    start=clock();
    for(int i=1;i<img.size()-1;i++){
        for(int j=1;j<img[0].size()-1;j++){
            ans[i-1][j-1]=cov(i,j);
        }
    }
    end=clock();
    printf("Time = %f\n",((double)(end-start))/CLOCKS_PER_SEC);

    writeAns();
    
    
    return 0;
}