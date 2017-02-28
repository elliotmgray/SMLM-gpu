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

    
    plhs[0] = mxCreateDoubleMatrix(nA,nB,mxREAL);
    
    double *A = mxGetPr(prhs[0]);
    double *B = mxGetPr(prhs[1]);
    mwSize mX = nA;  mwSize nX = nB;  double *X = mxGetPr(plhs[0]);
    
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

    cudaError_t cudaStat;

    double *d_A;
    
    cudaStat = cudaMalloc(&d_A, mA*nA*sizeof(double));
    if (cudaStat == cudaSuccess)
        mexPrintf("\nDevice memory was allocated for d_A");
    else if (cudaStat == cudaErrorMemoryAllocation)
        mexPrintf("\nDevice memory allocation failed for d_A");
        
    cudaStat = cudaMemcpy(d_A, A, mA*nA*sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStat == cudaSuccess)
        mexPrintf("\nA was succesfully copied into d_A");
    else
        mexPrintf("\nA failed to copy into d_A");
                  
    double *d_B;
    cudaStat = cudaMalloc(&d_B, mB*nB*sizeof(double));
    if (cudaStat == cudaSuccess)
        mexPrintf("\nDevice memory was allocated for d_B");
    else if (cudaStat == cudaErrorMemoryAllocation)
        mexPrintf("\nDevice memory allocation failed for d_B");
        
    cudaStat = cudaMemcpy(d_B, B, mB*nB*sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStat == cudaSuccess)
        mexPrintf("\nA was succesfully copied into d_B");
    else
        mexPrintf("\nA failed to copy into d_B");
        
    int info;
    /*int devInfoArray[1] = {-1};*/
    
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
        mexPrintf("\nUnsuccessful in initializing cublas handle!");
    
    double *Aarray[] = {d_A};
    double **d_Aarray;
    cudaStat = cudaMalloc(&d_Aarray, sizeof(double *));
    if (cudaStat == cudaSuccess)
        mexPrintf("\nDevice memory was allocated for d_Aarray");
    else if (cudaStat == cudaErrorMemoryAllocation)
        mexPrintf("\nDevice memory allocation failed for d_Aarray");
    status = cublasSetVector(1, sizeof( double *), Aarray, 1, d_Aarray, 1);
    if (status == CUBLAS_STATUS_SUCCESS)
        mexPrintf("\nSuccessful in copying Aarray to device!");
    else if (status == CUBLAS_STATUS_NOT_INITIALIZED)
        mexPrintf("\nThe library was not initialized when setting Aarray on the device.");
    else if (status == CUBLAS_STATUS_INVALID_VALUE)
        mexPrintf("\nThe parameters incx, incy, elemsize <=0 when setting Aarray on the device.");
    else if (status == CUBLAS_STATUS_MAPPING_ERROR)
        mexPrintf("\nThere was an error accessing GPU memory in the setting of Aarray on the device.");
    else
        mexPrintf("\nsomething weird is going on");
    
    
    double *Barray[] = {d_B};
    double **d_Barray;
    cudaStat = cudaMalloc(&d_Barray, sizeof(double *));
    if (cudaStat == cudaSuccess)
        mexPrintf("\nDevice memory was allocated for d_Barray");
    else if (cudaStat == cudaErrorMemoryAllocation)
        mexPrintf("\nDevice memory allocation failed for d_Barray");
    status = cublasSetVector(1, sizeof( double *), Barray, 1, d_Barray, 1);
    if (status == CUBLAS_STATUS_SUCCESS)
        mexPrintf("\nSuccessful in copying Barray to device!");
    else if (status == CUBLAS_STATUS_NOT_INITIALIZED)
        mexPrintf("\nThe library was not initialized when setting Barray on the device.");
    else if (status == CUBLAS_STATUS_INVALID_VALUE)
        mexPrintf("\nThe parameters incx, incy, elemsize <=0 when setting Barray on the device.");
    else if (status == CUBLAS_STATUS_MAPPING_ERROR)
        mexPrintf("\nThere was an error accessing GPU memory in the setting of Barray on the device.");
    else
        mexPrintf("\nsomething weird is going on");


    
    mexPrintf("\nm = %d - the number of columns in both A and B", mA);
    mexPrintf("\nn = %d - the number of rows in A", nA);
    mexPrintf("\nnrhs = %d - the number of rows in B", nB);
    mexPrintf("\nThe dimensions of solution X should be %d by %d", nA, nB);

    /* call the cublas mldivide function from the cublas library*/
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
    
    if (status == CUBLAS_STATUS_SUCCESS)
        mexPrintf("\nsuccessful operation");
    else if (status == CUBLAS_STATUS_NOT_INITIALIZED)
        mexPrintf("\nthe library was not initialized.");
    if (status == CUBLAS_STATUS_INVALID_VALUE)
        mexPrintf("\nthe parameters m,n,batchsize<0, lda<imax(1,m), or ldc<imax(1,m)");
    if (status == CUBLAS_STATUS_NOT_SUPPORTED)
        mexPrintf("\nthe parameters m<n or trans is different from non-transpose");
    if (status == CUBLAS_STATUS_ARCH_MISMATCH)
        mexPrintf("\nthe device has a compute capability < 200");
    if (status == CUBLAS_STATUS_EXECUTION_FAILED)
        mexPrintf("\nthe function failed to launch on the gpu");
    mexPrintf("\nInfo = %d", info);
    /*mexPrintf("\ndevInfoArray[0] = %d", devInfoArray[0]);*/

    /* Read X from device memory*/
    cudaStat = cudaMemcpy(X, d_B, mX*nX*sizeof(double), cudaMemcpyDeviceToHost);

    /* Destroy handle to cublas API*/
    cublasDestroy(handle);
    
    /* Free device memory*/
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Aarray);
    cudaFree(d_Barray);
    /*cudaFree(d_devInfoArray);*/
}
