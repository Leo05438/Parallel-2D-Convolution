# bin/sh

# ./run_serial.sh  > log_serial.txt
./run_pthread.sh > log_pthread.txt
./run_openMP.sh  > log_openMP.txt
./run_cuda.sh    > log_cuda.txt
./clean.sh