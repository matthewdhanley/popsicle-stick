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


__host__ __device__ double rollingAverage(double cur_avg, double new_sample, int cur_n){
    // n is the number of data points prior to new_sample being added
    cur_avg = (new_sample + ((double) cur_n) * cur_avg)/(cur_n+1);
    return cur_avg;
}


__global__ void monte_carlo_kernel(int* grid, int* draws, int m, double* average){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;  // thread index

	// set up striding
	int stride = tid;  // init stride to current tid
	int stride_step = gridDim.x * blockDim.x; // step size is equal to grid size (in unit of threads)

	average[tid] = 0;

    int n = 0;
	__syncthreads(); // wait for all threads to catch up.

	while (stride < N){
		int win = 0;
		int draw;
		int start_index = stride * m * m;
        int n_draws = 0;

        for (int i = 0; i < m*m; i++){
            grid[tid * m * m + i] = 0;
        }

		// simulating stick drawing, a sequential process in this case.
		while (win == 0){
			draw = draws[start_index + n_draws];
			grid[tid * m * m + draw] = 1;

			int col = draw % m;
			int row = draw / m;
			int row_count = 0;
			int col_count = 0;

			for(int j=0; j<m; j++){
				col_count += grid[tid * m * m + j * m + col];
			}

			for(int j=row*m; j<row * m + m; j++){
				row_count += grid[tid * m * m + j];
			}

			if (col_count >= m || row_count >= m){
				win = 1;
			}
			n_draws++;
		}
		n++;
		average[tid] = rollingAverage(average[tid], (double) n_draws, n);
		stride += stride_step;
	}
	__syncthreads();
}


void randperm_out(int a, int* out){
    for (int i=0; i<a; i++){
        out[i] = i;
    }
    int temp;
    int j;
    for (int i=a-1; i >= 0; --i){
        j = rand() % (i+1);
        temp = out[i];
        out[i] = out[j];
        out[j] = temp;
    }
}


int main() {
    cudaSetDevice(1);
    cudaFree(0); // avoid spoofing profiler.
    clock_t begin = clock();
    srand(time(0));

	// Grid Size
	int m = 10;

	// Init variables
	int* draws;
	double* average;
	int* d_draws;
	int* d_grid;
	double* d_average;

    int threads_per_block = 64;
    int n_blocks = 1;

	// allocate space on host
	draws = (int*) malloc(N*m*m*sizeof(int));
	average = (double*) malloc(threads_per_block * n_blocks * sizeof(double));

	clock_t begin_rand = clock();
    // Randomize numbers on the CPU
	for (int i = 0; i<m*m*N; i += m*m){
        randperm_out(m*m, &draws[i]);
    }
	clock_t end_rand = clock();
	double rand_time = (double)(end_rand - begin_rand) / CLOCKS_PER_SEC;
    printf("RAND TIME: %f\n", rand_time);

	// allocate space on the device for lots of popsicle sticks
	cudaSafeCall( cudaMalloc((void**) &d_draws, m * m * sizeof(int) * N) );
	cudaSafeCall( cudaMalloc((void**) &d_grid, m * m * sizeof(int) * threads_per_block * n_blocks) );
	cudaSafeCall( cudaMalloc((void**) &d_average, sizeof(double) * threads_per_block * n_blocks) );
	cudaCheckError();

	cudaSafeCall( cudaMemcpy(d_draws, draws, N*m*m*sizeof(int), cudaMemcpyHostToDevice) );
	cudaCheckError();

	monte_carlo_kernel<<<n_blocks, threads_per_block>>>(d_grid, d_draws, m, d_average);
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
	cudaFree(d_grid);

	return 0;
}
