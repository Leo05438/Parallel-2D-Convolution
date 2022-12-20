# bin/sh

THREADS=(2 3 4 5 6 7 8)
KERNELS=('3x3' '5x5' '7x7' '9x9' '11x11')
RESOLUTIONS=('128x128' '512x512' '2048x2048')

cd pthread/
make
for RESOLUTION in "${RESOLUTIONS[@]}"
do
    for KERNEL in "${KERNELS[@]}"
    do
        for THREAD in "${THREADS[@]}"
        do

            img_fname="${RESOLUTION}.jpeg"
            kernel_fname="kernel${KERNEL}.txt"
            kernel_fname_sk="kernel${KERNEL}_sk.txt"

            echo -e "\nmethod=pthread|exec=conv.out|thread_num=$THREAD|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
            ./conv.out $THREAD $img_fname $kernel_fname
            echo -e "\nmethod=pthread|exec=conv_sk.out|thread_num=$THREAD|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
            ./conv_sk.out $THREAD $img_fname $kernel_fname_sk
            echo -e "\nmethod=pthread|exec=conv_tp.out|thread_num=$THREAD|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
            ./conv_tp.out $THREAD $img_fname $kernel_fname
            echo -e "\nmethod=pthread|exec=conv_tprow.out|thread_num=$THREAD|resolution=${RESOLUTION}|kernel_size=${KERNEL}"
            ./conv_tprow.out $THREAD $img_fname $kernel_fname
            
        done
    done
done
cd ..

