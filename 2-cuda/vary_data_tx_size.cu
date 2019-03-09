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

__global__ void strictly_ops(unsigned long int n_ops) {
    unsigned long int num_iterations = n_ops / threads_per_block / n_blocks / 2;
    float res;
    for (unsigned long int i = 0; i < num_iterations; i++) {
        res = 3.0 * 4.0 + 6.0;
    }
}


void cpu_strictly_ops(unsigned long int n_ops){
    float res;
    for (unsigned long int i = 0; i < n_ops/2; i++){
        res = 3.0 * 4.0 + 6.0;
    }
}


int main(){
    cudaFree(0); // avoid spoofing profiler.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("n_ops, s_size, gpu, cpu\n");
    float milliseconds;
    for (unsigned long int i = 1; i <MAX_OPS; i*=2 ) {
        for (int j=2000; j<2000000; j*=10) {
            for (int k=0; k<10; k++) {
                printf("%lu,", i);
                printf("%d,",j);
                float* data = (float*) malloc(j);
                float* d_data;
                cudaEventRecord(start);
                cudaMalloc((void**) &d_data, j);
                cudaMemcpy(d_data, data, j, cudaMemcpyHostToDevice);
                strictly_ops << < n_blocks, threads_per_block >> > (i);
                cudaDeviceSynchronize();
                cudaMemcpy(data, d_data, j, cudaMemcpyDeviceToHost);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                printf("%.7g,", milliseconds / 1000.0);

                cudaEventRecord(start);
                cpu_strictly_ops(i);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                printf("%.7g\n", milliseconds / 1000.0);
            }
        }
    }

    return 0;
}