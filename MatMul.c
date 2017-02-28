#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mex.h"
#include "cublas_v2.h"

/* Thread block size*/
#define BLOCK_SIZE 16

/* Matrices are stored in row-major order:*/
/* M(row, col) = *(M.elements + row * M.stride + col)*/
typedef struct {
    int width;
    int height;
    int stride; 
    double *elements;
} Matrix;

/* Forward declaration*/
void MatMul(const Matrix A, const Matrix B, Matrix C);
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

/* mex interface function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mwSize mA = mxGetM(prhs[0]);
    mwSize mB = mxGetM(prhs[1]);
    mwSize nA = mxGetN(prhs[0]);
    mwSize nB = mxGetN(prhs[1]);
    int Stride = nA/BLOCK_SIZE;
    Matrix A, B, C;
    
    plhs[0] = mxCreateDoubleMatrix(mA,nB,mxREAL);
    
    A.width = nA;  A.height = mA;  A.stride = Stride;  A.elements = mxGetPr(prhs[0]);
    B.width = nB;  B.height = mB;  B.stride = Stride;  B.elements = mxGetPr(prhs[1]);
    C.width = nB;  C.height = mA;  C.stride = Stride;  C.elements = mxGetPr(plhs[0]);
    
    /* Throw errors if input does not have correct format */
	if(nrhs != 2)
        mexErrMsgTxt("Two inputs (A and B) are required.");
    else if(nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");
    
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgTxt("Input must be noncomplex double.");
        
    if (nA != mB)
        mexErrMsgTxt("Number of rows in A must equal number of columns in B.");

    MatMul(A, B, C);
}
/* Matrix multiplication - Host code*/
/* Matrix dimensions are assumed to be multiples of BLOCK_SIZE*/
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    /* Load A and B to device memory*/
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(double);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
                cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(double);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
                cudaMemcpyHostToDevice);

    /* Allocate C in device memory*/
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(double);
    cudaMalloc(&d_C.elements, size);

    /* Invoke kernel*/
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    /* Read C from device memory*/
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    /* Free device memory*/
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
/* Get a matrix element*/
__device__ double GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
/* Set a matrix element*/
__device__ void SetElement(Matrix A, int row, int col,
                           double value)
{
    A.elements[row * A.stride + col] = value;
}
/* Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is*/
/* located col sub-matrices to the right and row sub-matrices down*/
/* from the upper-left corner of A*/
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}
/* Matrix multiplication kernel called by MatMul()*/
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    /* Block row and column*/
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    /* Each thread block computes one sub-matrix Csub of C*/
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    /* Each thread computes one element of Csub*/
    /* by accumulating results into Cvalue*/
    double Cvalue = 0;

    /* Thread row and column within Csub*/
    int row = threadIdx.y;
    int col = threadIdx.x;

    /* Loop over all the sub-matrices of A and B that are*/
    /* required to compute Csub*/
    /* Multiply each pair of sub-matrices together*/
    /* and accumulate the results*/
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        /* Get sub-matrix Asub of A*/
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        /* Get sub-matrix Bsub of B*/
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        /* Shared memory used to store Asub and Bsub respectively*/
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        /* Load Asub and Bsub from device memory to shared memory*/
        /* Each thread loads one element of each sub-matrix*/
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        /* Synchronize to make sure the sub-matrices are loaded*/
        /* before starting the computation*/
        __syncthreads();
        /* Multiply Asub and Bsub together*/
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        /* Synchronize to make sure that the preceding*/
        /* computation is done before loading two new*/
        /* sub-matrices of A and B in the next iteration*/
        __syncthreads();
    }

    /* Write Csub to device memory*/
    /* Each thread writes one element*/
    SetElement(Csub, row, col, Cvalue);
}


