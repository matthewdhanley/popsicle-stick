//============================================================================
// Name        : popsicle-stick.cpp
// Author      : Matthew Hanley
// Version     :
// Copyright   :
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "gpu_err_check.h"


// Number of iterations to run
const int N = 5000000;

// size of grid dim
const int m = 10;


__host__ __device__ double rollingAverage(double cur_avg, double new_sample, int cur_n){
    // n is the number of data points prior to new_sample being added
    cur_avg = (new_sample + ((double) cur_n) * cur_avg)/(cur_n+1);
    return cur_avg;
}


__global__ void setup_kernel(curandState *state, unsigned long long seed){
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    int stride = tid;
    int stride_step = gridDim.x * blockDim.x;
    while(stride < N){
        curand_init(seed, tid, 0, &state[stride]);
        stride += stride_step;
    }
}


__global__ void monte_carlo_kernel(int* draws, curandState* state, double* average){

    // adding shared memory
    extern __shared__ int grid[];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;  // thread index

    // set up striding
    int stride = tid;  // init stride to current tid
    int stride_step = gridDim.x * blockDim.x; // step size is equal to grid size (in unit of threads)


// =====================================================================================================================
// Randomizing the Grid
// =====================================================================================================================
    int random_number;  // variable to hold random number
    int tmp;  // variable to hold element for swapping
    int j;  // variable for finding index to swap with
    int start_index;// where the loop should start
    // The arrays were generated all at once,
    // thus each thread has its own special location.


    while(stride < N){ // make sure that computation is needed
        curandState localState = state[stride];
        start_index = stride * m * m;

        // populate sequential numbers
        for(int i = 0; i < m * m; i++){
            draws[start_index + i] = i;
        }

        for(int i = m*m-1; i>=0; i--){
            random_number = (int) truncf( curand_uniform(&localState)*100000 );  // generate random number and make it big
            j = (random_number) % (i+1);  // get the index of a number to swap with that is less than i
            // perform the swap
            tmp = draws[start_index + j];
            draws[start_index + j] = draws[start_index + i];
            draws[start_index + i] = tmp;
        }
        stride += stride_step;
    }
    __syncthreads();

    // reset stride to tid
    stride = tid;
// =====================================================================================================================

    int n = 0;
    double local_average = 0;

    __syncthreads(); // wait for all threads to catch up.

    while (stride < N){
        int win = 0;
        int draw;
        start_index = stride * m * m;
        int n_draws = 0;

        for (int i = 0; i < m*m; i++){
            grid[threadIdx.x * m * m + i] = 0;
        }

        // simulating stick drawing, a sequential process in this case.
        while (win == 0){
            draw = draws[start_index + n_draws];
            grid[threadIdx.x * m * m + draw] = 1;

            int col = draw % m;
            int row = draw / m;
            int row_count = 0;
            int col_count = 0;

            for(j=0; j<m; j++){
                col_count += grid[threadIdx.x * m * m + j * m + col];
            }

            for(j=row*m; j<row * m + m; j++){
                row_count += grid[threadIdx.x * m * m + j];
            }

            if (col_count >= m || row_count >= m){
                win = 1;
            }
            n_draws++;
        }
        n++;
        local_average = rollingAverage(local_average, (double) n_draws, n);
        stride += stride_step;
    }
    average[tid] = local_average;
    __syncthreads();
}


int main() {
    cudaFree(0); // avoid spoofing profiler.
    clock_t begin = clock();
//    srand(time(0));

    // Init variables
    double* average;
    int* d_draws;
    double* d_average;

    // create and allocate for the random state
    curandState *d_state;
    cudaSafeCall( cudaMalloc(&d_state, N * sizeof(curandState)) );

    int threads_per_block = 64;
//    int n_blocks = (N + threads_per_block - 1) / threads_per_block;
    int n_blocks = 1;

    // allocate space on host
    average = (double*) malloc(threads_per_block * n_blocks * sizeof(double));

    // allocate space on the device for lots of popsicle sticks
    cudaSafeCall( cudaMalloc((void**) &d_draws, m * m * sizeof(int) * N) );
    cudaSafeCall( cudaMalloc((void**) &d_average, sizeof(double) * threads_per_block * n_blocks) );
    cudaCheckError();

    setup_kernel<<<n_blocks, threads_per_block>>>(d_state, (unsigned long long) time(NULL));
    cudaCheckError();
    cudaSafeCall( cudaDeviceSynchronize() );

    monte_carlo_kernel<<<n_blocks, threads_per_block, m * m * threads_per_block * sizeof(int)>>>(d_draws, d_state, d_average);
    cudaCheckError();
    cudaSafeCall( cudaDeviceSynchronize() );

    // copy data to host for analysis.
    cudaSafeCall( cudaMemcpy(average, d_average, threads_per_block * n_blocks * sizeof(double), cudaMemcpyDeviceToHost) );

    double big_avg = 0;
    for (int i = 0; i < threads_per_block * n_blocks; i++){
        big_avg = rollingAverage(big_avg, average[i], i);
    }

    printf("Average: %f\n", big_avg);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("TIME: %f\n", time_spent);

    cudaFree(d_draws);

    return 0;
}
