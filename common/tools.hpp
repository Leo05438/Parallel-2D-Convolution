#include <fstream>
#include <vector>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace std;

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
                    img[j+pad][k+pad]+=rgb_image[i+k*bpp+j*width*bpp]/bpp;
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
    // img.push_back(tmp);
    stbi_image_free(rgb_image);
}

void readKernel(){
    int size,entry,sum=0;
    ifstream f;
    f.open("../common/kernel.txt");
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
    for(int i=pad;i<img.size()-pad;i++){
        tmp.clear();
        for(int j=pad;j<img[0].size()-pad;j++){
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
    FILE *fptr = fopen("./ans.txt", "w");
    
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            fprintf(fptr, "%f\n", (float) ans[i][j]);
        }
    }

    fclose(fptr);
}

void writeImage(){
    uint8_t* ans_image = (uint8_t*) malloc(width * height * 1);

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            ans_image[width * i + j] = ans[i][j];
        }
    }

    stbi_write_png("./image.jpeg", width, height, 1, ans_image, width * 1);
    stbi_image_free(ans_image);
}

void checkAns(){
    FILE *fptr = fopen("../serial/ans.txt", "r");
    vector<float> ans_arr;
    float tmp;
    bool flag;

    for(long long int i = 0; i < height * width; i++){
        flag = fscanf(fptr, "%f", &tmp);
        ans_arr.push_back(tmp);
    }

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if((ans[i][j] - ans_arr[i * width + j]) > 10e-5){
                printf("Wrong Answer in ans[%d][%d]:\n", i, j);
                printf("Serial Ans = %f, Your Ans = %f\n", ans_arr[i * width + j], ans[i][j]);
                return;
            }
        }
    }

    printf("Correct Anwser\n");
    writeImage();
}