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
const int N = 1000000;


__global__ void setup_kernel(curandState *state, unsigned long long seed){
	int tid = threadIdx.x+blockDim.x*blockIdx.x;
	int stride = tid;
	int stride_step = gridDim.x * blockDim.x;
	while(stride < N){
		curand_init(seed, tid, 0, &state[tid]);
		stride += stride_step;
	}
}


__global__ void monte_carlo_kernel(int* number_grid, int* popsicle_sticks,
									   int grid_dim, curandState *state, int* good_behavior){

	int tid = blockDim.x * blockIdx.x + threadIdx.x;  // thread index

	// set up striding
	int stride = tid;  // init stride to current tid
	int stride_step = gridDim.x * blockDim.x; // step size is equal to grid size (in unit of threads)

	// init good_behavior to zero. Might be able to eliminate this loop.
	while(stride < N){
		good_behavior[stride] = 0;
		stride += stride_step;
	}

	__syncthreads();

	// reset stride to the current tid
	stride = tid;

	// have all the threads work together to populate the sticks, in order first.
	while(stride < grid_dim * grid_dim * N){
		popsicle_sticks[stride] = stride % (grid_dim * grid_dim);
		number_grid[stride] = 0;  // the number grid is where the popsicle sticks drawn are stored
		stride += stride_step;
	}

	__syncthreads(); // wait for all threads to catch up.

	int random_number;  // variable to hold random number
	int tmp;  // variable to hold element for swapping
	int j;  // variable for finding index to swap with
	int start_index;// where the loop should start
	  // The arrays were generated all at once,
    // thus each thread has its own special location.

	// now going to randomize without replacement
	stride = tid;  // reset the stride to current tid

	while(stride < N){ // make sure that computation is needed
		curandState localState = state[stride];
		start_index = stride * grid_dim * grid_dim;
		for(int i = grid_dim*grid_dim-1; i>=0; i--){
			random_number = (int) truncf( curand_uniform(&localState)*100000 );  // generate random number and make it big
			j = (random_number) % (i+1);  // get the index of a number to swap with that is less than i
			if (blockIdx.x == 200 && threadIdx.x == 0){
				printf("%d\n", start_index + j);
			}
			// perform the swap
			tmp = popsicle_sticks[start_index + j];
//			printf("%d ",popsicle_sticks[grid_dim*grid_dim*N-1]);
			popsicle_sticks[start_index + j] = popsicle_sticks[start_index + i];
			popsicle_sticks[start_index + i] = tmp;
//			printf("%d ",popsicle_sticks[grid_dim*grid_dim*N-1]);
		}
		stride += stride_step;
	}
	__syncthreads();


	// reset stride to tid
	stride = tid;

	while (stride < N){
		int pizza_party = 0;
		int good_behavior_count = 0;
		int stick_drawn;
		start_index = stride * grid_dim * grid_dim;

		// simulating stick drawing, a sequential process in this case.
		while (pizza_party == 0){
			stick_drawn = popsicle_sticks[start_index + good_behavior_count];
			number_grid[start_index + stick_drawn] += 1;
//			if (number_grid[start_index + stick_drawn] > 1){
//				printf("%d ",stick_drawn);
//			}


			int col = stick_drawn % grid_dim;
			int row = stick_drawn / grid_dim;
			int row_count = 0;
			int col_count = 0;
			for(int j=0; j<grid_dim; j++){
				col_count += number_grid[start_index + j * grid_dim + col];
			}
			for(int j=row*grid_dim; j<row * grid_dim + grid_dim; j++){
				row_count += number_grid[start_index + j];
			}
			if (col_count >= grid_dim || row_count >= grid_dim){
				pizza_party = 1;
			}
			good_behavior_count++;
		}
		good_behavior[stride] = good_behavior_count;
//		if (good_behavior_count >= 100){
//			printf("blockIdx.x: %3d\tthreadIdx.x %3d\ttid: %7d\tstride: %7d\tgood_behavior_count: %5d\n",blockIdx.x, threadIdx.x, tid, stride, good_behavior_count);
//		}
		stride += stride_step;
	}
	__syncthreads();
}


int main() {
	cudaFree(0); // avoid spoofing profiler.

	// Grid Size
	int grid_dim = 10;

	// Init variables
	int* d_popsicle_sticks;
	int* d_good_behavior;
	int* good_behavior;
	int* d_number_grid;
	int* popsicles;

	// create and allocate for the random state
	curandState *d_state;
    cudaSafeCall( cudaMalloc(&d_state, N * sizeof(curandState)) );

	// allocate space on host
	good_behavior = (int*) malloc(N*sizeof(int));

	// allocate space on the device for lots of popsicle sticks
	cudaSafeCall( cudaMalloc((void**) &d_popsicle_sticks, grid_dim * grid_dim * sizeof(int) * N) );
	cudaSafeCall( cudaMalloc((void**) &d_number_grid, grid_dim * grid_dim * sizeof(int) * N) );
	cudaSafeCall( cudaMalloc((void**) &d_good_behavior, sizeof(int) * N) );

	cudaCheckError();

	int threads_per_block = 128;
	int n_blocks = (N + threads_per_block - 1) / threads_per_block;

    printf("Running setup . . .\n");
	setup_kernel<<<n_blocks, threads_per_block>>>(d_state, (unsigned long long) time(NULL));
	cudaCheckError();
	cudaSafeCall( cudaDeviceSynchronize() );

	printf("Running monte_carlo_kernel . . .\n");
	monte_carlo_kernel<<<1, threads_per_block>>>(d_number_grid, d_popsicle_sticks, grid_dim, d_state, d_good_behavior);
	cudaCheckError();
	cudaSafeCall( cudaDeviceSynchronize() );

	// copy data to host for analysis.
	printf("Getting data back . . .\n");
	cudaSafeCall( cudaMemcpy(good_behavior, d_good_behavior, N*sizeof(int), cudaMemcpyDeviceToHost) );

	// calculate average
	float good_behavior_big;
	for (int i = 0; i < N; i++){
//		printf("%d ", good_behavior[i]);
		good_behavior_big += (float) good_behavior[i];
	}

	printf("%f\n", good_behavior_big / (float)N);

	free(good_behavior);
	cudaFree(d_popsicle_sticks);
	cudaFree(d_good_behavior);
	cudaFree(d_number_grid);

	return 0;
}
