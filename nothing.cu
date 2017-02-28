#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
int main()
{
    double *matrix, *d_A;
    
    matrix = (double *)calloc(1000, sizeof(double));
    cudaMalloc( &d_A, 1000*sizeof(double));
    cudaMemcpy(d_A, matrix, 1000*sizeof(double), cudaMemcpyHostToDevice);
    
    printf("\nthe first element of matrix is %f\n", matrix[0]);
    
    cudaFree(d_A);
    free(matrix);
    return 0;
}