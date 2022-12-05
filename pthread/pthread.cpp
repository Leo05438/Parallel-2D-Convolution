#include <stdint.h>
#include <pthread.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNEL_NUM 3

int main() {
    int width, height, bpp;

    uint8_t *rgb_image = stbi_load("../common/image.jpeg", &width, &height, &bpp, 3);

    // Write your code to populate rgb_image here
    // stbi_write_png("./image.jpeg", width, height, CHANNEL_NUM, rgb_image, width * CHANNEL_NUM);

    stbi_image_free(rgb_image);

    return 0;
}