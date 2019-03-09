#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include "gpu_err_check.h"
#include <float.h>

const unsigned long int MAX_OPS = 10000000000;
const int threads_per_block = 512;
const int n_blocks = 1000;

__global__ void naive(float* a, float* res) {
    int starting_point = blockIdx.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < threads_per_block; i++) {
        res[tid] += a[starting_point + i];
    }
}


__global__ void share(float* a, float* res) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_a[threads_per_block];
    __shared__ float s_result[threads_per_block];
    s_a[threadIdx.x] = a[tid];
    for (int i = 0; i < threads_per_block; i++) {
        s_result[threadIdx.x] += s_a[threadIdx.x + i];
    }
    res[tid] = s_result[threadIdx.x];
}


int main(){
    cudaFree(0); // avoid spoofing profiler.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* a = (float*) malloc(n_blocks * threads_per_block * sizeof(float));

    for (int i = 0; i < n_blocks * threads_per_block; i++){
        a[i] = 3.0;
    }

    float* d_a;
    float* d_res;


    printf("n_ops, naive, shared\n");
    float milliseconds;
    for (int k=0; k<100; k++) {
        cudaMalloc((void**) &d_a, n_blocks * threads_per_block * sizeof(float));
        cudaMalloc((void**) &d_res, n_blocks * threads_per_block * sizeof(float));
        cudaMemcpy(d_a, a, n_blocks * threads_per_block * sizeof(float), cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        naive<<< n_blocks, threads_per_block >>> (d_a, d_res);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("%.7g,", milliseconds / 1000.0);

        cudaEventRecord(start);
        share<<< n_blocks, threads_per_block >>> (d_a, d_res);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaDeviceSynchronize();
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("%.7g\n", milliseconds / 1000.0);


        cudaFree(&d_a);
    }

    return 0;
}