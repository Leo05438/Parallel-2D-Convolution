#include <fstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;

void readImg(){
    int row,col,entry;
    vector<float> tmp;
    ifstream f;
    f.open("img.txt");
    f>>row>>col;
    tmp.clear();
    for(int i=0;i<col+2;i++){
        tmp.push_back(0);
    }
    img.push_back(tmp);
    for(int i=0;i<row;i++){
        tmp.clear();
        tmp.push_back(0);
        for(int j=0;j<col;j++){
            f>>entry;
            tmp.push_back(entry);
        }
        tmp.push_back(0);
        img.push_back(tmp);
    }
    tmp.clear();
    for(int i=0;i<col+2;i++){
        tmp.push_back(0);
    }
    img.push_back(tmp);
    f.close();
}
void readKernel(){
    int row,col,entry,sum=0;
    ifstream f;
    f.open("kernel.txt");
    f>>row>>col;
    for(int i=0;i<row;i++){
        vector<float> tmp;
        for(int j=0;j<col;j++){
            f>>entry;
            tmp.push_back(entry);
            sum+=entry;
        }
        kernel.push_back(tmp);
    }
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            kernel[i][j]/=sum;
        }
    }
    f.close();
}
void initAns(){
    vector<float> tmp;
    for(int i=1;i<img.size()-1;i++){
        tmp.clear();
        for(int j=1;j<img[0].size()-1;j++){
            tmp.push_back(0);
        }
        ans.push_back(tmp);
    }
}
void showImgAndKernel(){
    printf("img:\n");
    // system("tiv img.txt");
    for(int i=0;i<img.size();i++){
        for(int j=0;j<img[0].size();j++){
            printf("%3.f ",img[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("kernel:\n");
    for(int i=0;i<kernel.size();i++){
        for(int j=0;j<kernel[0].size();j++){
            printf("%3.2f ",kernel[i][j]);
        }
        printf("\n");
    }
}
void showAns(){
    printf("ans:\n");
    for(int i=0;i<ans.size();i++){
        for(int j=0;j<ans[0].size();j++){
            printf("%3.f ",ans[i][j]);
        }
        printf("\n");
    }
}
void init(bool vis){
    readImg();
    readKernel();
    initAns();
    if(vis){
        showImgAndKernel();
    }
}
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
    init(true);
    for(int i=1;i<img.size()-1;i++){
        for(int j=1;j<img[0].size()-1;j++){
            ans[i-1][j-1]=cov(i,j);
        }
    }
    showAns();
    
    return 0;
}