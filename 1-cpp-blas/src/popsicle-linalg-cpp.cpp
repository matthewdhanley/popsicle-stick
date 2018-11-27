//============================================================================
// Name        : popsicle-linalg-cpp.cpp
// Author      : Matthew Hanley
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>


// Globals
int N = 100000;  // number of iterations
int n = 100;  // number of elements in the grid
int m = (int) sqrt(n);  // length of one row


void make_D_array(double* D){
	// makes the first m rows
	for (int i=0; i<m; i++){
		for (int j=i*m; j<i*m+10; j++){
			D[i*n+j] = 1.0;
		}
	}

	// makes the second m rows
	for (int i=m; i<2*m; i++){
		for (int j=0; j<n; j+=m){
			D[i*n+j+i-m] = 1.0;
		}
	}
}


void print_D(double* D){
	for (int i=0; i<m*2; i++){
		printf("Row: %d\n",i);
		for (int j=0; j<n; j++){
			printf("%f ", D[i*n+j]);
		}
		printf("\n");
	}
}


int* randperm(int a){
	int* out;
	out = (int*)malloc(a * sizeof(int));
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
	return out;
}


int mult (double *mat1, double *mat2, double *mat3,
          double alpha, double beta,
          int mm, int nn, int kk)
{

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mm, nn, kk, alpha, mat1, kk, mat2, nn, beta,
                mat3, nn);

    return 0;
}


double rollingAverage(double cur_avg, double new_sample, int cur_n){
	// n is the number of data points prior to new_sample being added
	cur_avg = (new_sample + ((double) cur_n) * cur_avg)/(cur_n+1);
	return cur_avg;
}


int main() {
	// seed the random number generator
	srand(time(0));

	// allocate space for d
	double* D = (double*) calloc(n*m*2, sizeof(double));
	double* result = (double*) calloc(m*2, sizeof(double));
	make_D_array(D);
	double avg = 0;


	for (int k=0; k<N; k++){
		double* x = (double*) calloc(n, sizeof(double));
		int* picks = randperm(n);
		int break_flag = 0;
		for (int i=0; i<n; i++){
			x[picks[i]] = 1.0;
			mult(D, x, result, 1.0, 0.0, m*2, 1, n);
			for (int j=0; j<m*2; j++){
				if (result[j] == (double) m){
					break_flag = 1;
				}
			}
			if (break_flag){
//				printf("%d ",i);
				avg = rollingAverage(avg, (double) i, k);
				break;
			}
		}
	}
	printf("Average: %f\n",avg);


	return 0;
}
