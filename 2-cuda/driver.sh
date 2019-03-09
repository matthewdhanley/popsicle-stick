#!/bin/bash
echo Compiling device_info.cu
nvcc device_info.cu -o device_info
echo Compiling 0-naive.cu
nvcc 0-naive.cu -o 0-naive
echo Compiling 1-curand.cu
nvcc 1-curand.cu -o 1-curand
echo Compiling 2-local-var.cu
nvcc 2-local-var.cu -o 2-local-var
echo Compiling 3-shared-mem.cu
nvcc 3-shared-mem.cu -o 3-shared-mem
echo Compiling 3-shared-mem.cu with fast math
nvcc 3-shared-mem.cu -o 4-shared-mem-fast-math --use_fast_math
echo Compiling 5-grid-size.cu
nvcc 5-grid-size.cu -o 5-grid-size --use_fast_math
echo Compiling 6-occupancy.cu with fast math
nvcc 6-occupancy.cu -o 6-occupancy --use_fast_math
echo Compiling 7-cublas.cu with fast math
nvcc 7-cublas.cu -lcublas -o 7-cublas --use_fast_math
echo Compiling coalesced.cu
nvcc coalesced.cu -o coalesced
echo Compiling data_tx_size.cu
nvcc data_tx_size.cu -o data_tx_size
echo Compiling local_vars.cu
nvcc local_vars.cu -o local_vars
echo Compiling shared.cu
nvcc shared.cu -o shared
echo Compiling vary_data_tx_size.cu
nvcc vary_data_tx_size.cu -o vary_data_tx_size
echo ''
echo Running device_info
./device_info
echo Running 0-naive.cu
for i in `seq 1 3`;
do
    ./0-naive
done
echo ''
echo Running 1-curand.cu
for i in `seq 1 3`;
do
	./1-curand
done
echo ''
echo Running 2-local-var.cu
for i in `seq 1 3`;
do
	./2-local-var
done
echo ''
echo Running 3-shared-mem.cu
for i in `seq 1 3`;
do
    ./3-shared-mem
done
echo ''
echo Running 4-shared-mem-fast-math
for i in `seq 1 3`;
do
    ./4-shared-mem-fast-math
done
echo ''
echo Running 5-grid-size.cu
for i in `seq 1 3`;
do
    ./5-grid-size
done
echo ''
echo Running 6-occupancy.cu
for i in `seq 1 3`;
do
    ./6-occupancy
done
echo ''
echo Running 7-cublas.cu
for i in `seq 1 3`;
do
    ./7-cublas
done
echo Running coalesced.cu
./coalesced
echo Running data_tx_size.cu
./data_tx_size
echo Running local_vars.cu
./local_vars
echo Running shared.cu
./shared
echo Running vary_data_tx_size.cu
./vary_data_tx_size
