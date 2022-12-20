# bin/sh

THREADS=(2 3 4 5 6 7 8)
KERNELS=('3x3' '5x5' '7x7' '9x9' '11x11')
RESOLUTIONS=('128x128' '512x512' '2048x2048')

cd serial/
make
for RESOLUTION in "${RESOLUTIONS[@]}"
do
    for KERNEL in "${KERNELS[@]}"
    do

        img_fname="${RESOLUTION}.jpeg"
        kernel_fname="kernel${KERNEL}.txt"
        kernel_fname_sk="kernel${KERNEL}_sk.txt"

        echo -e "\nmethod=serial|exec=conv.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv.out $img_fname $kernel_fname
        echo -e "\nmethod=serial|exec=conv_sk.out|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
        ./conv_sk.out $img_fname $kernel_fname_sk
        
    done
done
cd ..

