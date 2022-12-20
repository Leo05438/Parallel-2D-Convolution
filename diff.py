import cv2
import numpy as np

img_serial = cv2.imread('serial/output/128x128_kernel11x11_sk.jpeg', cv2.IMREAD_GRAYSCALE)
img_cuda = cv2.imread('cuda/output/128x128_kernel11x11_sk.jpeg', cv2.IMREAD_GRAYSCALE)

diff = img_cuda - img_serial

f_out = open('diff.txt', 'w')

for line in diff:
    for ele in line:
        f_out.write(f'{ele} ')
    f_out.write('\n')

f_out.close()
print('done!')