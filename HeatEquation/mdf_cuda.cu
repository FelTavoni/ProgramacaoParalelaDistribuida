/*
 * Paralelized marathon code - CUDA mapped.
 * 
 * Strategy: Each core will calculate one position of the heating cube!
 *
 * Universidade Federal de São Carlos,
 * Felipe Tavoni.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#define STABILITY 1.0f/sqrt(3.0f)


__global__ void mdf_heat_once(double*  __restrict__ u0, 
                            double*  __restrict__ u1, 
                            const unsigned int* npX, 
                            const unsigned int* npY, 
                            const unsigned int* npZ,
                            const double* deltaH,
                            const double* deltaT,
                            const double* alpha,
                            const double* inErr,
                            const double* boundaries){

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // z + ((*npZ) * y) + ((*npZ) * (*npY) * x)

    // For each position on the grid, spread the heat, adjusting the neighboorhood.
    double left   = *boundaries;
    double right  = *boundaries;
    double up     = *boundaries;
    double down   = *boundaries;
    double top    = *boundaries;
    double bottom = *boundaries;

    if ((x <= (*npX)) && (y <= (*npY)) && (z <= (*npZ))) {
    
        if ((x > 0) && (x < ((*npX) - 1))){
          left  = u0[z + ((*npZ) * y) + ((*npZ) * (*npY) * x-1)];
          right = u0[z + ((*npZ) * y) + ((*npZ) * (*npY) * x+1)];
        }else if (x == 0) right = u0[z + ((*npZ) * y) + ((*npZ) * (*npY) * x+1)];
        else left = u0[z + ((*npZ) * y) + ((*npZ) * (*npY) * x-1)];
        
        if ((y > 0) && (y < ((*npY) - 1))){
          up  = u0[z + ((*npZ) * y-1) + ((*npZ) * (*npY) * x)];
          down = u0[z + ((*npZ) * y+1) + ((*npZ) * (*npY) * x)];
        }else if (y == 0) down = u0[z + ((*npZ) * y+1) + ((*npZ) * (*npY) * x)];
        else up = u0[z + ((*npZ) * y-1) + ((*npZ) * (*npY) * x)];
        
        if ((z > 0) && (z < ((*npZ) - 1))){
          top  = u0[z-1 + ((*npZ) * y) + ((*npZ) * (*npY) * x)];
          bottom = u0[z+1 + ((*npZ) * y) + ((*npZ) * (*npY) * x)];
        }else if (z == 0) bottom = u0[z+1 + ((*npZ) * y) + ((*npZ) * (*npY) * x)];
        else top = u0[z-1 + ((*npZ) * y) + ((*npZ) * (*npY) * x)];
        
        // Simply applying the formula and stores the value on a new spot.
        u1[z + ((*npZ) * y) + ((*npZ) * (*npY) * x)] =  (*alpha) * ( top + bottom + up + down + left + right  - (6.0f * u0[z + ((*npZ) * y) + ((*npZ) * (*npY) * x)] )) + u0[z + ((*npZ) * y) + ((*npZ) * (*npY) * x)];

    }
                
}

__global__ void mdf_heat_check(double*  __restrict__ u0, 
                            double*  __restrict__ u1, 
                            const unsigned int* npX, 
                            const unsigned int* npY, 
                            const unsigned int* npZ,
                            const double* inErr,
                            const double* boundaries,
                            volatile int* heated){

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // If all the positions are heated more than 100, finish the iteration.
    double err = 0.0f;
    err = fabs(u0[z + ((*npZ) * y) + ((*npZ) * (*npY) * x)] - (*boundaries));
    if (*heated && (err < (*inErr)))
        *heated = 0;
}

int onDevice(unsigned int h_npX, unsigned int h_npY, unsigned int h_npZ, double h_deltaH, double h_deltaT, double h_alpha, double h_boundaries, double h_inErr) {

    // For debbugging purposes
    cudaError_t err = cudaGetLastError();

    // Allocate variables in the GPU and copy they content from host.
    double *d_deltaT; //0.01;
    double *d_deltaH;  //0.25f;
    unsigned int *d_npX;  //1.0f;
    unsigned int *d_npY;  //1.0f;
    unsigned int *d_npZ;  //1.0f;
    // Constant variables.
    double *d_boundaries;
    double *d_inErr;
    double *d_alpha;

    printf("I'M HERE - NUMBER ONE!!!\n");
    fflush(stdout);
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    cudaMalloc((void**)&d_deltaT, sizeof(double));
    cudaMemcpy(&d_deltaT, &h_deltaT, sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_deltaH, sizeof(double));
    cudaMemcpy(&d_deltaH, &h_deltaH, sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_npX, sizeof(double));
    cudaMemcpy(&d_npX, &h_npX, sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_npY, sizeof(double));
    cudaMemcpy(&d_npY, &h_npY, sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_npZ, sizeof(double));
    cudaMemcpy(&d_npZ, &h_npZ, sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_boundaries, sizeof(double));
    cudaMemcpy(&d_boundaries, &h_boundaries, sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_inErr, sizeof(double));
    cudaMemcpy(&d_inErr, &h_inErr, sizeof(double), cudaMemcpyHostToDevice);

    // Allocate memory inside the GPU for the grid. Here the matrix is flattened.
    double *d_u0;
    double *d_u1;
    cudaMalloc((void**)&d_u0, h_npZ * h_npY * h_npX * sizeof(double));
    cudaMemset((void**)&d_u0, 0x00, h_npZ * h_npY * h_npX * sizeof(double));
    cudaMalloc((void**)&d_u1, h_npZ * h_npY * h_npX * sizeof(double));
    cudaMemset((void**)&d_u1, 0x00, h_npZ * h_npY * h_npX * sizeof(double));
    printf("I'M HERE - NUMBER TWO!!!\n");
    fflush(stdout);
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

  double steps = 0;
  volatile int *heated;
  *heated = 0;
  volatile int *d_heated;
  cudaMalloc((void**)&d_heated, sizeof(int));
  cudaMemcpy(&d_heated, &heated, sizeof(int), cudaMemcpyHostToDevice);

  // Defining the grid.
  dim3 threadsPerBlock(4, 4, 4); // 4 * 4 * 4 = 64 threads = 2 warps!
  dim3 blocksPerGrid(ceil( (double)h_npX/64), ceil( (double)h_npY/64), ceil( (double)h_npZ/64)); // Dividing the cube into (x*y*z)/64 minicubes.

    while (!(*heated)) {
        
        steps++;

        *heated = 1;
        cudaMemcpy(&d_heated, &heated, sizeof(int), cudaMemcpyHostToDevice);

        // Calling the kernel for heat function.
        mdf_heat_once<<<blocksPerGrid, threadsPerBlock>>>(d_u0, d_u1, d_npX, d_npY, d_npZ, d_deltaH, d_deltaT, d_alpha, d_inErr, d_boundaries);
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

        // Switch the cubes, since the previous won't be reused, so we don't need to allocate more memory.
        double *ptr = d_u0;
        d_u0 = d_u1;
        d_u1 = ptr;

        mdf_heat_check<<<blocksPerGrid, threadsPerBlock>>>(d_u0, d_u1, d_npX, d_npY, d_npZ, d_inErr, d_boundaries, d_heated);
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

        cudaMemcpy(&d_heated, &heated, sizeof(int), cudaMemcpyDeviceToHost);
    }

    printf("Steps: %.1lf\n", steps);

  // Free the space in GPU.
  return EXIT_SUCCESS;
}

int onHost() {
  // Define variables to be used in the process
  double h_deltaT = 0.0f; //0.01;
  double h_deltaH =0.0f;  //0.25f;
  double h_sizeX = 0.0f;  //1.0f;
  double h_sizeY = 0.0f;  //1.0f;
  double h_sizeZ = 0.0f;  //1.0f;

  // Some constants defined in the description.
  double boundaries = 100.0f;
  double inErr = 1e-15;

  // Alpha constant in formula.
  double alpha;

  // Read input
  fscanf(stdin, "%lf", &h_deltaT);
  fscanf(stdin, "%lf", &h_deltaH);
  fscanf(stdin, "%lf", &h_sizeZ);
  fscanf(stdin, "%lf", &h_sizeY);
  fscanf(stdin, "%lf", &h_sizeX);

  // Calculate ne number of elements in x, y and z axis.
  unsigned int h_npX = (unsigned int) (h_sizeX / h_deltaH);
  unsigned int h_npY = (unsigned int) (h_sizeY / h_deltaH);
  unsigned int h_npZ = (unsigned int) (h_sizeZ / h_deltaH);

  alpha = h_deltaT / (h_deltaH * h_deltaH);

  // Call the device to calculate the heating.
  onDevice(h_npX, h_npY, h_npZ, h_deltaH, h_deltaT, alpha, boundaries, inErr);
  // mdf_heat(h_u0, h_u1, h_npX, h_npY, h_npZ, h_deltaH, h_deltaT, 1e-15, 100.0f);
  //mdf_print(u1,  npX, npY, npZ);

  return EXIT_SUCCESS;
}

int main (int argc, char *argv[]){
    return onHost();
}
