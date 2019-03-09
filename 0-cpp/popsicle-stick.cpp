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


int* randperm(int n){
	int* out;
	out = (int*)malloc(n * sizeof(int));
	for (int i=0; i<n; i++){
		out[i] = i;
	}
	int temp;
	int j;
	for (int i=n-1; i >= 0; --i){
		j = rand() % (i+1);
		temp = out[i];
		out[i] = out[j];
		out[j] = temp;
	}
	return out;
}


int main() {
	srand(time(NULL));
	clock_t begin = clock();
	// Number of iterations to run
	int N = 5000000;

	// Grid Size
	int grid_dim = 10;

	// Init variables
	int* popsicle_sticks;
	int good_behavior_count;
	int* number_grid;
	unsigned int big_counter = 0;

	// allocate space
	popsicle_sticks = (int*) malloc(grid_dim * grid_dim * sizeof(int));

	for (int i=0;i<N;i++){
		number_grid = (int*) calloc(grid_dim * grid_dim, sizeof(int));

		// no pizza party yet
		int pizza_party = 0;

		// reset number of sticks pulled
		good_behavior_count = 0;

		// generate pop sticks in random order
		popsicle_sticks = randperm(grid_dim*grid_dim);

		while (pizza_party == 0){
			int stick_drawn = popsicle_sticks[good_behavior_count];
			number_grid[stick_drawn] = 1;

			if (good_behavior_count > grid_dim){
				int col = stick_drawn % grid_dim;
				int row = stick_drawn / grid_dim;
				int row_count = 0;
				int col_count = 0;
				for(int j=0; j<grid_dim; j++){
					col_count += number_grid[j*grid_dim+col];
				}
				for(int j=row*grid_dim; j<row*grid_dim+grid_dim; j++){
					row_count += number_grid[j];
				}
				if (col_count == grid_dim || row_count == grid_dim){
					pizza_party = 1;
				}
			}
			good_behavior_count++;
		}
		big_counter += (unsigned int) good_behavior_count;
//		printf("Big Counter %d\n",big_counter);
	}
	printf("Average: %d\n",big_counter/N);
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("TIME: %f\n", time_spent);

	free(number_grid);
	free(popsicle_sticks);
	return 0;
}
