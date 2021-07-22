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


__global__ void mdf_heat_once(double ***  __restrict__ u0, 
                            double ***  __restrict__ u1, 
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

    // For each position on the grid, spread the heat, adjusting the neighboorhood.
    double left   = *boundaries;
    double right  = *boundaries;
    double up     = *boundaries;
    double down   = *boundaries;
    double top    = *boundaries;
    double bottom = *boundaries;

    if ((x <= (*npX)) && (y <= (*npY)) && (z <= (*npZ))) {
    
        if ((x > 0) && (x < ((*npX) - 1))){
          left  = u0[z][y][x-1];
          right = u0[z][y][x+1];
        }else if (x == 0) right = u0[z][y][x+1];
        else left = u0[z][y][x-1];
        
        if ((y > 0) && (y < ((*npY) - 1))){
          up  = u0[z][y-1][x];
          down = u0[z][y+1][x];
        }else if (y == 0) down = u0[z][y+1][x];
        else up = u0[z][y-1][x];
        
        if ((z > 0) && (z < ((*npZ) - 1))){
          top  = u0[z-1][y][x];
          bottom = u0[z+1][y][x];
        }else if (z == 0) bottom = u0[z+1][y][x];
        else top = u0[z-1][y][x];
        
        // Simply applying the formula and stores the value on a new spot.
        u1[z][y][x] =  (*alpha) * ( top + bottom + up + down + left + right  - (6.0f * u0[z][y][x] )) + u0[z][y][x];

    }
                
}

__global__ void mdf_heat_check(double ***  __restrict__ u0, 
                            double ***  __restrict__ u1, 
                            const unsigned int* npX, 
                            const unsigned int* npY, 
                            const unsigned int* npZ,
                            const double* inErr,
                            const double* boundaries,
                            double* heated){

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // If all the positions are heated more than 100, finish the iteration.
    double err = 0.0f;
    double maxErr = 0.0f;
    err = fabs(u0[z][y][x] - (*boundaries));
    if (err > (*inErr))
      (*heated)++;
}

int onDevice(double*** h_u0, double*** h_u1, double h_npX, double h_npY, double h_npZ, double h_deltaH, double h_deltaT) {

  // Define variables to be used in GPU.
  double ***d_u0;
  double ***d_u1;
  double *d_deltaT; //0.01;
  double *d_deltaH;  //0.25f;
  double *d_npX;  //1.0f;
  double *d_npY;  //1.0f;
  double *d_npZ;  //1.0f;

  // double *inErr;
  // double *boundaries;
  double d_alpha = h_deltaT / (h_deltaH * h_deltaH);

  cudaMalloc((void**)&d_deltaT, sizeof(double));
  cudaMalloc((void**)&d_deltaH, sizeof(double));
  cudaMalloc((void**)&d_npX, sizeof(double));
  cudaMemcpy(d_npX, &h_npX, sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_npY, sizeof(double));
  cudaMemcpy(d_npY, &h_npY, sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_npZ, sizeof(double));
  cudaMemcpy(d_npZ, &h_npZ, sizeof(double), cudaMemcpyHostToDevice);

  // Allocate memory inside the GPU.
  cudaMalloc((void**)d_u0, h_npZ * sizeof(double**));
  cudaMalloc((void**)d_u1, h_npZ * sizeof(double**));

  for (unsigned int i = 0; i < h_npZ; i++){
      cudaMalloc((void**)d_u0[i], h_npY * sizeof(double*));
      cudaMalloc((void**)d_u1[i], h_npY * sizeof(double*));
  }

  for (unsigned int i = 0; i < h_npZ; i++){
      for (unsigned int j = 0; j < h_npY; j++){
          double *d_aux0;
          double *d_aux1;
          cudaMalloc((void**)d_aux0, h_npX * sizeof(double));
          cudaMalloc((void**)d_aux1, h_npX * sizeof(double));
          // Initial condition - zero in all points
          cudaMemset((void*)d_aux0, 0, h_npX * sizeof(double));
          cudaMemset((void*)d_aux1, 0, h_npX * sizeof(double));
          d_u0[i][j] = d_aux0;
          d_u1[i][j] = d_aux1;
      }
  }

  // Defining the grid.
  dim3 threadsPerBlock(4, 4, 4); // 4 * 4 * 4 = 64 threads = 2 warps!
  dim3 blocksPerGrid(ceil( (double)h_npX/64), ceil( (double)h_npY/64), ceil( (double)h_npZ/64)); // Dividing the cube into (x*y*z)/64 minicubes.

  // Calling the kernel for heat function.
  mdf_heat_once<<blocksPerGrid, threadsPerBlock>>(d_u0, d_u1, d_npX, d_npY, d_npZ, d_deltaH, d_deltaT, d_alpha, 1e-15, 100.0f);

  // Switch the cubes, since the previous won't be reused, so we don't need to allocate more memory.
  double ***ptr = d_u0;
  d_u0 = d_u1;
  d_u1 = ptr;

  // Free the space in GPU.
  return EXIT_SUCCESS;
}

int onHost() {
  // Define variables to be used in the process
  double ***h_u0;
  double ***h_u1;
  double h_deltaT = 0.0f; //0.01;
  double h_deltaH =0.0f;  //0.25f;
  double h_sizeX = 0.0f;  //1.0f;
  double h_sizeY = 0.0f;  //1.0f;
  double h_sizeZ = 0.0f;  //1.0f;

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

  //printf("p(%u, %u, %u)\n", npX, npY, npZ);
  //Allocing memory
  // Allocating a 3-dimensional grid, used to build the cube.
  // u0 and u1 are the cubes structure. u0 holds the value of the `t` condtion and u1 the `t+1` condition.
  h_u0 = (double***) malloc (h_npZ * sizeof(double**));
  h_u1 = (double***) malloc (h_npZ * sizeof(double**));

  for (unsigned int i = 0; i < h_npZ; i++){
      h_u0[i] = (double**) malloc (h_npY * sizeof(double*));
      h_u1[i] = (double**) malloc (h_npY * sizeof(double*));
  }

  for (unsigned int i = 0; i < h_npZ; i++){
      for (unsigned int j = 0; j < h_npY; j++){
          double *h_aux0 = (double *) malloc (h_npX * sizeof(double));
          double *h_aux1 = (double *) malloc (h_npX * sizeof(double));
          //initial condition - zero in all points
          memset(h_aux0, 0x00, h_npX * sizeof(double));
          memset(h_aux1, 0x00, h_npX * sizeof(double));
          h_u0[i][j] = h_aux0;
          h_u1[i][j] = h_aux1;
      }
  }

  // Call the device to calculate the heating.
  onDevice(h_u0, h_u1, h_npX, h_npY, h_npZ, h_deltaH, h_deltaT);
  // mdf_heat(h_u0, h_u1, h_npX, h_npY, h_npZ, h_deltaH, h_deltaT, 1e-15, 100.0f);
  //mdf_print(u1,  npX, npY, npZ);

  //Free memory
  for (unsigned int i = 0; i < h_npZ; i++){
      for (unsigned int j = 0; j < h_npY; j++){
          free(h_u0[i][j]);
          free(h_u1[i][j]);
      }
  }

  for (unsigned int i = 0; i < h_npZ; i++){
      free(h_u0[i]);
      free(h_u1[i]);
  }

  free(h_u0);
  free(h_u1);

  return EXIT_SUCCESS;
}

int main (int argc, char *argv[]){
    return onHost();
}
