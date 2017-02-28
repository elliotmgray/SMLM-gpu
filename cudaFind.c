#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mex.h"
#include "cublas_v2.h"

#define BLOCK_SIZE 16

/* Forward declaration*/
__global__ void getN(double *d_A, int l, int *d_n);
__global__ void getI(double *d_A, int l, double *d_I, int *d_n);
void printError(cudaError_t cudaStat);
/* mex interface function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *A, *I;
	int l;
    int n[2];
    for (int i=0; i<2; i++)
        n[i] = 0;
        
    cudaError_t cudaStat;
    l = mxGetM(prhs[0]); A = mxGetPr(prhs[0]);
	double *d_A, *d_I;
	cudaStat = cudaMalloc( &d_A, l*sizeof(double));
        printError(cudaStat);
	cudaStat = cudaMemcpy( d_A, A, l*sizeof(double), cudaMemcpyHostToDevice);
        printError(cudaStat);
	int *d_n;
	cudaStat = cudaMalloc(&d_n, 2*sizeof(int));
        printError(cudaStat);
	cudaStat = cudaMemcpy(d_n, n, 2*sizeof(int), cudaMemcpyHostToDevice);
        printError(cudaStat);
        
    /* Throw errors if input does not have correct format */
	if(nrhs != 1)
        mexPrintf("One input column vector is required.");
	if(nlhs != 1)
        mexErrMsgTxt("Two outputs required: n and I. n will be scalar, and size(I) = (n,1).");

    int dimGrid;
    int r = l % BLOCK_SIZE;
    if (r != 0)
        dimGrid = ((l - r)/BLOCK_SIZE) +1;
    else
        dimGrid = l/BLOCK_SIZE;



    getN<<<dimGrid,BLOCK_SIZE>>>(d_A, l, d_n);
        cudaStat = cudaGetLastError();
        printError(cudaStat);
    cudaStat = cudaMemcpy( n, d_n, 2*sizeof(int), cudaMemcpyDeviceToHost);
        printError(cudaStat);
    
	plhs[0] = mxCreateDoubleMatrix(n[0],1,mxREAL);
	cudaStat = cudaMalloc( &d_I, n[0]*sizeof(double));
        printError(cudaStat);
	I = mxGetPr(plhs[0]);

	getI<<<dimGrid,BLOCK_SIZE>>>(d_A, l, d_I, d_n);
        cudaStat = cudaGetLastError();
        printError(cudaStat);
    cudaStat = cudaMemcpy( I, d_I, n[0]*sizeof(double), cudaMemcpyDeviceToHost);
        printError(cudaStat);

    cudaFree(d_I);
    cudaFree(d_A);
    cudaFree(d_n);
}

__global__ void getN( double *d_A, int l, int *d_n)
{
    int tid = threadIdx.x + BLOCK_SIZE*blockIdx.x;
    if (tid < l)
        if (d_A[tid] == 0)
            atomicAdd(&d_n[0],1);
}

__global__ void getI( double *d_A, int l, double *d_I, int *d_n)
{
    int tid = threadIdx.x + BLOCK_SIZE*blockIdx.x;
    if (tid < l)
    {
        if (d_A[tid] == 0)
        {
            d_I[atomicAdd(&d_n[1],1)] = tid;
        }
    }
}
/* ================================================================================================*/
/* Auxiliary routine: print error message after cuda API function call*/
/* ================================================================================================*/
void printError(cudaError_t cudaStat)
{
    if (cudaStat != cudaSuccess)
        mexPrintf("\nCUDA error: %s", cudaGetErrorString(cudaStat));
}