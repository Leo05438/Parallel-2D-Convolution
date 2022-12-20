# bin/sh

THREADS=(2 3 4 5 6 7 8)
KERNELS=('3x3' '5x5' '7x7' '9x9' '11x11')
RESOLUTIONS=('128x128' '512x512' '2048x2048')

cd cuda/
make
for RESOLUTION in "${RESOLUTIONS[@]}"
do
    for KERNEL in "${KERNELS[@]}"
    do

        img_fname="${RESOLUTION}.jpeg"
        kernel_fname="kernel${KERNEL}.txt"
        kernel_fname_sk="kernel${KERNEL}_sk.txt"

        echo -e "\nmethod=cuda|exec=conv_basic.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv_basic.out $img_fname $kernel_fname
        echo -e "\nmethod=cuda|exec=conv_pitch.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv_pitch.out $img_fname $kernel_fname
        echo -e "\nmethod=cuda|exec=conv_tiling.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv_tiling.out $img_fname $kernel_fname
        echo -e "\nmethod=cuda|exec=conv_tiling_modified.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv_tiling_modified.out $img_fname $kernel_fname
        echo -e "\nmethod=cuda|exec=conv_tiling+pitch.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv_tiling+pitch.out $img_fname $kernel_fname
        echo -e "\nmethod=cuda|exec=conv_sk_basic.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv_sk_basic.out $img_fname $kernel_fname_sk
        echo -e "\nmethod=cuda|exec=conv_sk_tiling.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv_sk_tiling.out $img_fname $kernel_fname_sk
        echo -e "\nmethod=cuda|exec=conv_sk_tiling_modified.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv_sk_tiling_modified.out $img_fname $kernel_fname_sk
        
    done
done
cd ..