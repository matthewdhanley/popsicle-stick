#!/bin/bash
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
echo ''
echo Running 0-naive.cu
for i in `seq 1 10`;
do
    ./0-naive
done
echo ''
echo Running 1-curand.cu
for i in `seq 1 10`;
do
	./1-curand
done
echo ''
echo Running 2-local-var.cu
for i in `seq 1 10`;
do
	./2-local-var
done
echo ''
echo Running 3-shared-mem.cu
for i in `seq 1 10`;
do
    ./3-shared-mem
done
echo ''
echo Running 4-shared-mem-fast-math
for i in `seq 1 10`;
do
    ./4-shared-mem-fast-math
done
echo ''
echo Running 5-grid-size.cu
for i in `seq 1 10`;
do
    ./5-grid-size
done
echo ''
echo Running 6-occupancy.cu
for i in `seq 1 10`;
do
    ./6-occupancy
done