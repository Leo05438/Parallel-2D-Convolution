#include <fstream>
#include <vector>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace std;

vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width,height,bpp,pad;

void readImg(){
    vector<float> tmp;
    uint8_t *rgb_image=stbi_load("../common/image.jpeg",&width,&height,&bpp,3);

    tmp.clear();
    for(int i=0;i<pad;i++){
        for(int j=0;j<width+pad*2;j++){
            tmp.push_back(0);
        }
        img.push_back(tmp);
    }
    for(int i=0;i<bpp;i++){
        for(int j=0;j<height;j++){
            if(i==0){
                tmp.clear();
                for(int k=0;k<pad;k++){
                    tmp.push_back(0);
                }
            }
            for(int k=0;k<width;k++){
                if(i==0){
                    tmp.push_back(((float)rgb_image[i+k*bpp+j*width*bpp]/bpp));
                }else{
                    img[j][k]+=rgb_image[i+k*bpp+j*width*bpp]/bpp;
                }
            }
            if(i==0){
                for(int k=0;k<pad;k++){
                    tmp.push_back(0);
                }
                img.push_back(tmp);
            }
        }
    }
    for(int i=0;i<pad;i++){
        for(int j=0;j<width+pad*2;j++){
            tmp.push_back(0);
        }
        img.push_back(tmp);
    }
    img.push_back(tmp);
    stbi_image_free(rgb_image);
}
void readKernel(){
    int size,entry,sum=0;
    ifstream f;
    f.open("kernel.txt");
    f>>size;
    pad=size/2;

    for(int i=0;i<size;i++){
        vector<float> tmp;
        for(int j=0;j<size;j++){
            f>>entry;
            tmp.push_back(entry);
            sum+=entry;
        }
        kernel.push_back(tmp);
    }
    if(sum==0)sum=1;
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
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
void init(){
    readKernel();
    readImg();
    initAns();
}
void writeAns(){
    uint8_t *rgb_image=new uint8_t[width*height];
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            rgb_image[width*i+j]=ans[i][j];
        }
    }
    stbi_write_png("./image.jpeg", width, height, 1, rgb_image, width * 1);
}