#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mex.h"
#include "cublas_v2.h"

/*This function will take 2 A's and 2 B's, it will compute A1\B1 and A2\B2 on device 1,
and it will compute A1\B2 and A2\B1 on device 2.  It will return the outputs X1, X2, Y1, and Y2
such that:
 A1\B1 = X1, 
 A2\B2 = X2, 
 
 A1\B2 = Y1, and 
 A2\B1 = Y2, approximately.
 
 Equivalently, it finds X1, X2, Y1, and Y2 such that these norms are minimized:
 ||A1*X1 - B1||,
 ||A2*X2 - B2||,
 
 ||A1*Y1 - B2||, and
 ||A2*Y2 - B1||.
 */
void printError(cudaError_t cudaStat);

/* mex interface function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Input should be (A1,B1,A2,B2)*/
	/* Output should be (X1,X2,Y1,Y2)*/
	mwSize A1_M = mxGetM(prhs[0]);
    mwSize A1_n = mxGetN(prhs[0]);
    double *A1 = mxGetPr(prhs[0]);
    
    mwSize B1_M = mxGetM(prhs[1]);
    mwSize B1_n = mxGetN(prhs[1]);
    double *B1 = mxGetPr(prhs[1]);

	mwSize A2_M = mxGetM(prhs[2]);
    mwSize A2_n = mxGetN(prhs[2]);
    double *A2 = mxGetPr(prhs[2]);
    
    mwSize B2_M = mxGetM(prhs[3]);
    mwSize B2_n = mxGetN(prhs[3]);
    double *B2 = mxGetPr(prhs[3]);
     
    mwSize X1_M = A1_n;
    mwSize X1_n = B1_n;
	plhs[0] = mxCreateDoubleMatrix(X1_M,X1_n,mxREAL);
    double *X1 = mxGetPr(plhs[0]);
    
    mwSize X2_M = A2_n;
    mwSize X2_n = B2_n;
	plhs[1] = mxCreateDoubleMatrix(X2_M,X2_n,mxREAL);
    double *X2 = mxGetPr(plhs[1]);
	
    mwSize mY1 = A1_n;
    mwSize nY1 = B2_n;
	plhs[2] = mxCreateDoubleMatrix(mY1,nY1,mxREAL);
    double *Y1 = mxGetPr(plhs[2]);
    
    mwSize mY2 = A2_n;  mwSize nY2 = B1_n;
	plhs[3] = mxCreateDoubleMatrix(mY2,nY2,mxREAL);
    double *Y2 = mxGetPr(plhs[3]);
    mexPrintf("size(X1) = %d by %d",X1_M,X1_n);
    mexPrintf("size(X2) = %d by %d",X2_M,X2_n);
    mexPrintf("size(Y1) = %d by %d",mY1,nY1);
    mexPrintf("size(Y2) = %d by %d",mY2,nY2);
    /* Throw errors if input does not have correct format */
	if (nrhs != 4)
        mexPrintf("\nFour inputs (A and B) are required.");
    else if (nlhs != 4)
        mexPrintf("\nWrong number of output arguments.");
    
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexPrintf("\nInput must be noncomplex double.");
        
    if (A1_M != B1_M || A2_M != B2_M)
        mexPrintf("\nNumber of rows in A must equal number of rows in B.");
        
    if (A1_M < A1_n || A2_M < A2_n)
        mexPrintf("\nThis is an underdetermined system.  The routine will not work.");

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    cudaError_t cudaStat;
/* Allocate and copy resources for device 0*/
    cudaStat = cudaSetDevice(0);
    printError(cudaStat);
    double *d0_A1, *d0_B1, *d0_A2, *d0_B2;
    cudaStat = cudaMalloc(&d0_A1, A1_M*A1_n*sizeof(double));
    printError(cudaStat);
    cudaStat = cudaMalloc(&d0_B1, B1_M*B1_n*sizeof(double));
    printError(cudaStat);
    cudaStat = cudaMalloc(&d0_A2, A2_M*A2_n*sizeof(double));
    printError(cudaStat);
    cudaStat = cudaMalloc(&d0_B2, B2_M*B2_n*sizeof(double));
    printError(cudaStat);
    cudaStat = cudaMemcpyAsync(d0_A1, A1, A1_M*A1_n*sizeof(double), cudaMemcpyHostToDevice);
    printError(cudaStat);
	cudaStat = cudaMemcpyAsync(d0_B1, B1, B1_M*B1_n*sizeof(double), cudaMemcpyHostToDevice);
    printError(cudaStat);
    cudaStat = cudaMemcpyAsync(d0_A2, A2, A2_M*A2_n*sizeof(double), cudaMemcpyHostToDevice);
    printError(cudaStat);
	cudaStat = cudaMemcpyAsync(d0_B2, B2, B2_M*B2_n*sizeof(double), cudaMemcpyHostToDevice);
    printError(cudaStat);
/* Allocate and copy resources for device 1*/
    cudaStat =  cudaSetDevice(1);
    printError(cudaStat);
    double *d1_A1, *d1_B1, *d1_A2, *d1_B2;
    cudaStat = cudaMalloc(&d1_A1, A1_M*A1_n*sizeof(double));
    printError(cudaStat);
    cudaStat = cudaMalloc(&d1_B1, B1_M*B1_n*sizeof(double));
    printError(cudaStat);
    cudaStat = cudaMalloc(&d1_A2, A2_M*A2_n*sizeof(double));
    printError(cudaStat);
    cudaStat = cudaMalloc(&d1_B2, B2_M*B2_n*sizeof(double));
    printError(cudaStat);
    cudaStat = cudaMemcpyAsync(d1_A1, A1, A1_M*A1_n*sizeof(double), cudaMemcpyHostToDevice);
    printError(cudaStat);
	cudaStat = cudaMemcpyAsync(d1_B1, B1, B1_M*B1_n*sizeof(double), cudaMemcpyHostToDevice);
    printError(cudaStat);
    cudaStat = cudaMemcpyAsync(d1_A2, A2, A2_M*A2_n*sizeof(double), cudaMemcpyHostToDevice);
    printError(cudaStat);
	cudaStat = cudaMemcpyAsync(d1_B2, B2, B2_M*B2_n*sizeof(double), cudaMemcpyHostToDevice);
    printError(cudaStat);
/* Allocate and copy more resources for device 0*/
    cudaStat = cudaSetDevice(0);
    printError(cudaStat);
    double **Aarray0, **d0_Aarray, **Barray0, **d0_Barray;
    Aarray0 = (double **)mxCalloc(2, sizeof(double *));
    Aarray0[0] = d0_A1;
    Aarray0[1] = d0_A2;
    Barray0 = (double **)mxCalloc(2, sizeof(double *));
    Barray0[0] = d0_B1;
    Barray0[1] = d0_B2;
    cudaStat = cudaMalloc(&d0_Aarray, 2*sizeof(double *));
    printError(cudaStat);
	cublasSetVector(2, sizeof( double *), Aarray0, 1, d0_Aarray, 1);
    cudaStat = cudaMalloc(&d0_Barray, 2*sizeof(double *));
    printError(cudaStat);
	cublasSetVector(2, sizeof( double *), Barray0, 1, d0_Barray, 1);
/* Allocate and copy more resources for device 1*/
    cudaStat = cudaSetDevice(1);
    printError(cudaStat);
    double **Aarray1, **d1_Aarray, **Barray1, **d1_Barray;
    Aarray1 = (double **)mxCalloc(2, sizeof(double *));
    Aarray1[0] = d1_A1;
    Aarray1[1] = d1_A2;
    Barray1 = (double **)mxCalloc(2, sizeof(double *));
    Barray1[0] = d1_B2;
    Barray1[1] = d1_B1;
    cudaStat = cudaMalloc(&d1_Aarray, 2*sizeof(double *));
    printError(cudaStat);
	cublasSetVector(2, sizeof( double *), Aarray1, 1, d1_Aarray, 1);
    cudaStat = cudaMalloc(&d1_Barray, 2*sizeof(double *));
    printError(cudaStat);
	cublasSetVector(2, sizeof( double *), Barray1, 1, d1_Barray, 1);
    
    
    int info0[2];
    int info1[2];
    cublasStatus_t status0, status1;
    cublasHandle_t handle0;
    cublasHandle_t handle1;
    
    
    

	

    cudaStat = cudaSetDevice(0);
    printError(cudaStat);
    cudaStat = cudaDeviceSynchronize();
    printError(cudaStat);
    cublasCreate(&handle0);
    status0 = cublasDgelsBatched( handle0,
                                 CUBLAS_OP_N,
                                 A1_M,
                                 A1_n,
                                 B1_n,
                                 d0_Aarray,
                                 A1_M,
                                 d0_Barray,
                                 B1_M,
                                 info0,
                                 NULL,
                                 2);  
    if (status0 == CUBLAS_STATUS_SUCCESS)
        mexPrintf("\n0 successful operation");
    else if (status0 == CUBLAS_STATUS_NOT_INITIALIZED)
        mexPrintf("\n0 the library was not initialized.");
    if (status0 == CUBLAS_STATUS_INVALID_VALUE)
        mexPrintf("\n0 the parameters m,n,batchsize<0, lda<imax(1,m), or ldc<imax(1,m)");
    if (status0 == CUBLAS_STATUS_NOT_SUPPORTED)
        mexPrintf("\n0 the parameters m<n or trans is different from non-transpose");
    if (status0 == CUBLAS_STATUS_ARCH_MISMATCH)
        mexPrintf("\n0 the device has a compute capability < 200");
    if (status0 == CUBLAS_STATUS_EXECUTION_FAILED)
        mexPrintf("\n0 the function failed to launch on the gpu");
    cublasDestroy(handle0);
    
    cudaStat = cudaSetDevice(1);
    printError(cudaStat);
    cudaStat = cudaDeviceSynchronize();
    printError(cudaStat);
    cublasCreate(&handle1);
    status1 = cublasDgelsBatched( handle1,
                                 CUBLAS_OP_N,
                                 A1_M,
                                 A1_n,
                                 B1_n,
                                 d1_Aarray,
                                 A1_M,
                                 d1_Barray,
                                 B1_M,
                                 info1,
                                 NULL,
                                 2);
    if (status1 == CUBLAS_STATUS_SUCCESS)
        mexPrintf("\n1 successful operation");
    else if (status1 == CUBLAS_STATUS_NOT_INITIALIZED)
        mexPrintf("\n1 the library was not initialized.");
    if (status1 == CUBLAS_STATUS_INVALID_VALUE)
        mexPrintf("\n1 the parameters m,n,batchsize<0, lda<imax(1,m), or ldc<imax(1,m)");
    if (status1 == CUBLAS_STATUS_NOT_SUPPORTED)
        mexPrintf("\n1 the parameters m<n or trans is different from non-transpose");
    if (status1 == CUBLAS_STATUS_ARCH_MISMATCH)
        mexPrintf("\n1 the device has a compute capability < 200");
    if (status1 == CUBLAS_STATUS_EXECUTION_FAILED)
        mexPrintf("\n1 the function failed to launch on the gpu");
    cublasDestroy(handle1);
        
    cudaStat = cudaSetDevice(0);
    printError(cudaStat);
    cudaStat = cudaMemcpy(X1, d0_B1, X1_M*X1_n*sizeof(double), cudaMemcpyDeviceToHost);
    printError(cudaStat);
    cudaStat = cudaMemcpy(X2, d0_B2, X2_M*X2_n*sizeof(double), cudaMemcpyDeviceToHost);
    printError(cudaStat);
                                 
    cudaStat = cudaSetDevice(1);
    printError(cudaStat);
    cudaStat = cudaMemcpy(Y1, d1_B2, mY1*nY1*sizeof(double), cudaMemcpyDeviceToHost);
    printError(cudaStat);
    cudaStat = cudaMemcpy(Y2, d1_B1, mY2*nY2*sizeof(double), cudaMemcpyDeviceToHost);
    printError(cudaStat);
    
    /* Free device memory*/
    cudaFree(d0_A1);
    cudaFree(d0_B1);
    cudaFree(d0_A2);
    cudaFree(d0_B2);
    cudaFree(d0_Aarray);
    cudaFree(d0_Barray);
    cudaFree(d1_A1);
    cudaFree(d1_B1);
    cudaFree(d1_A2);
    cudaFree(d1_B2);
    cudaFree(d1_Aarray);
    cudaFree(d1_Barray);
}

/* ================================================================================================*/
/* Auxiliary routine: print error message after cuda API function call*/
/* ================================================================================================*/
void printError(cudaError_t cudaStat)
{
    if (cudaStat != cudaSuccess)
        mexPrintf("\nCUDA error: %s", cudaGetErrorString(cudaStat));
}