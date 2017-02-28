#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mex.h"
#include "cublas_v2.h"

/* Matrices are stored in row-major order:*/

/* mex interface function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mwSize mA = mxGetM(prhs[0]);
    mwSize mB = mxGetM(prhs[1]);
    mwSize nA = mxGetN(prhs[0]);
    mwSize nB = mxGetN(prhs[1]);
    double *A = mxGetPr(prhs[0]);
    double *B = mxGetPr(prhs[1]);
    mwSize mX = nA;  mwSize nX = nB;
	plhs[0] = mxCreateDoubleMatrix(mX,nX,mxREAL);
    double *X = mxGetPr(plhs[0]);
	
    /* Throw errors if input does not have correct format */
	if (nrhs != 2)
        mexErrMsgTxt("\nTwo inputs (A and B) are required.");
    else if (nlhs > 1)
        mexErrMsgTxt("\nToo many output arguments.");
    
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("\nInput must be noncomplex double.");
        
    if (mA != mB)
        mexErrMsgTxt("\nNumber of rows in A must equal number of rows in B.");
        
    if (mA < nA)
        mexErrMsgTxt("\nThis is an underdetermined system.  The routine will not work.");

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    double *d_A;
    cudaMalloc(&d_A, mA*nA*sizeof(double));
    cudaMemcpy(d_A, A, mA*nA*sizeof(double), cudaMemcpyHostToDevice);
                  
    double *d_B;
    cudaMalloc(&d_B, mB*nB*sizeof(double));
	cudaMemcpy(d_B, B, mB*nB*sizeof(double), cudaMemcpyHostToDevice);
        
    int info;
    cublasStatus_t status;
    cublasHandle_t handle;
    cublasCreate(&handle);
	
    double **Aarray;
    Aarray = (double **)mxCalloc(1, sizeof(double *));
    Aarray[0] = d_A;
    
    double **d_Aarray;
    cudaMalloc(&d_Aarray, sizeof(double *));
	cublasSetVector(1, sizeof( double *), Aarray, 1, d_Aarray, 1);
    
    double **Barray;
    Barray = (double **)mxCalloc(1, sizeof(double *));
    Barray[0] = d_B;
    
    double **d_Barray;
    cudaMalloc(&d_Barray, sizeof(double *));
	cublasSetVector(1, sizeof( double *), Barray, 1, d_Barray, 1);
	/* library call*/
    status = cublasDgelsBatched( handle,
                                 CUBLAS_OP_N,
                                 mA,
                                 nA,
                                 nB,
                                 d_Aarray,
                                 mA,
                                 d_Barray,
                                 mB,
                                 &info,
                                 NULL,
                                 1);  
    /* Read X from device memory*/
    cudaMemcpy(X, d_B, mX*nX*sizeof(double), cudaMemcpyDeviceToHost);
    /* Destroy handle to cublas API*/
    cublasDestroy(handle);
    /* Free device memory*/
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Aarray);
    cudaFree(d_Barray);
}