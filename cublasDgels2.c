#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mex.h"
#include "cublas_v2.h"

/* Matrices are stored in row-major order:*/

/* mex interface function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Input should be (A1,B1,A2,B2)*/
	/* Output should be (X1,X2)*/
	mwSize mA1 = mxGetM(prhs[0]);
    mwSize mB1 = mxGetM(prhs[1]);
    mwSize nA1 = mxGetN(prhs[0]);
    mwSize nB1 = mxGetN(prhs[1]);
    double *A1 = mxGetPr(prhs[0]);
    double *B1 = mxGetPr(prhs[1]);
    mwSize mX1 = nA1;  mwSize nX1 = nB1;
	plhs[0] = mxCreateDoubleMatrix(mX1,nX1,mxREAL);
    double *X1 = mxGetPr(plhs[0]);
	
	mwSize mA2 = mxGetM(prhs[2]);
    mwSize mB2 = mxGetM(prhs[3]);
    mwSize nA2 = mxGetN(prhs[2]);
    mwSize nB2 = mxGetN(prhs[3]);
    double *A2 = mxGetPr(prhs[2]);
    double *B2 = mxGetPr(prhs[3]);
    mwSize mX2 = nA2;  mwSize nX2 = nB2;
	plhs[1] = mxCreateDoubleMatrix(mX2,nX2,mxREAL);
    double *X2 = mxGetPr(plhs[1]);
	
    /* Throw errors if input does not have correct format */
	if (nrhs != 4)
        mexPrintf("\nFour inputs (A and B) are required.");
    else if (nlhs > 2)
        mexPrintf("\nToo many output arguments.");
    
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexPrintf("\nInput must be noncomplex double.");
        
    if (mA1 != mB1 || mA2 != mB2)
        mexPrintf("\nNumber of rows in A must equal number of rows in B.");
        
    if (mA1 < nA1 || mA2 < nA2)
        mexPrintf("\nThis is an underdetermined system.  The routine will not work.");

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    double *d_A1;
    cudaMalloc(&d_A1, mA1*nA1*sizeof(double));
    cudaMemcpy(d_A1, A1, mA1*nA1*sizeof(double), cudaMemcpyHostToDevice);
                  
    double *d_B1;
    cudaMalloc(&d_B1, mB1*nB1*sizeof(double));
	cudaMemcpy(d_B1, B1, mB1*nB1*sizeof(double), cudaMemcpyHostToDevice);
        
    double *d_A2;
    cudaMalloc(&d_A2, mA2*nA2*sizeof(double));
    cudaMemcpy(d_A2, A2, mA2*nA2*sizeof(double), cudaMemcpyHostToDevice);
                  
    double *d_B2;
    cudaMalloc(&d_B2, mB2*nB2*sizeof(double));
	cudaMemcpy(d_B2, B2, mB2*nB2*sizeof(double), cudaMemcpyHostToDevice);
    
    int info[2];
    cublasStatus_t status;
    cublasHandle_t handle;
    cublasCreate(&handle);
	
    double **Aarray;
    Aarray = (double **)mxCalloc(2, sizeof(double *));
    Aarray[0] = d_A1;
    Aarray[1] = d_A2;
    
    double **d_Aarray;
    cudaMalloc(&d_Aarray, 2*sizeof(double *));
	cublasSetVector(2, sizeof( double *), Aarray, 1, d_Aarray, 1);
    
    double **Barray;
    Barray = (double **)mxCalloc(2, sizeof(double *));
    Barray[0] = d_B1;
    Barray[1] = d_B2;
    
    double **d_Barray;
    cudaMalloc(&d_Barray, 2*sizeof(double *));
	cublasSetVector(2, sizeof( double *), Barray, 1, d_Barray, 1);
	/* library call*/
    status = cublasDgelsBatched( handle,
                                 CUBLAS_OP_N,
                                 mA1,
                                 nA1,
                                 nB1,
                                 d_Aarray,
                                 mA1,
                                 d_Barray,
                                 mB1,
                                 info,
                                 NULL,
                                 2);  
    /* Read X from device memory*/
    cudaMemcpy(X1, d_B1, mX1*nX1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(X2, d_B2, mX2*nX2*sizeof(double), cudaMemcpyDeviceToHost);
    /* Destroy handle to cublas API*/
    cublasDestroy(handle);
    /* Free device memory*/
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_Aarray);
    cudaFree(d_Barray);
}