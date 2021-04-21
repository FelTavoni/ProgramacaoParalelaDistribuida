#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

// O kernel processará o espaço do bloco, eliminando todos os múltiplos de k
__global__ void sieve(uint* d_array, uint k) {
     int idx = threadIdx.x + (32 * blockIdx.x);
     if ( (d_array[idx] % k) == 0) { 
          d_array[idx] = 0;
     }
     __syncthreads();
}

extern "C++" void onDevice(uint *ptrK, uint block_size, uint n, uint *h_array) {

     uint k = *ptrK;

     // Alocando os espaços na GPU para processamento.
     uint* d_array;

     // Allocando memória na GPU.
     cudaMalloc((void**)&d_array, (block_size) * sizeof(uint));

     // Copiando conteudo para a GPU.
     cudaMemcpy(d_array, h_array, (block_size) * sizeof(uint), cudaMemcpyHostToDevice);

     dim3 threadsPorBloco(32, 1, 1);
     dim3 blocosPorGrid(ceil( (double)block_size/32), 1, 1);

     // Uso da GPU para eliminar os múltiplos do bloco
     // Invoka o kernel para operações de mod
     sieve<<<blocosPorGrid, threadsPorBloco>>>(d_array, k);
     // Detecção de erro caso o kernel apresente algum...
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) 
          printf("Error: %s\n", cudaGetErrorString(err));

     // Atualiza o array presente na CPU depois do resultado.
     cudaMemcpy(h_array, d_array, block_size * sizeof(uint), cudaMemcpyDeviceToHost);

     // Desaloca os espaços alocados.
     cudaFree(d_array);
}