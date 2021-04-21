#include <iostream>
#include <cuda.h>
#include <math.h>
#include <stdio.h>

typedef unsigned int uint;

// O kernel a seguir obtém o índice da thread operante e verifica se ela está apta para manipular o array.
// Assim, por meio dele, executamos em paralelo (CUDA Cores * SMs) operações em paralelo, dado o particionamento em blocos.
__global__ void sieve(uint* d_array, uint i, uint k, uint n) {
	int idx = threadIdx.x + (32 * blockIdx.x);
	if ((idx >= k) && (idx <= n)) {
		if ( (d_array[idx] % i) == 0) { 
			d_array[idx] = 0;
		}
	}
	__syncthreads();
}

// Função que obtém o valor de entrada presente no arquivo.
uint get_n(char* filename) {
     FILE* infile;
     infile = fopen(filename,"r");
     if (infile != NULL) {
          uint n;
          fscanf(infile,"%i", &n);
          fclose(infile);
          return n;
     } else {
          return 0;
     }
}

void onDevice(uint* h_array, uint n) {
	uint* d_array;

	// Allocando memória na GPU.
	cudaMalloc((void**)&d_array, (n + 1) * sizeof(uint));

	// Copiando conteudo para a GPU.
    cudaMemcpy(d_array, h_array, (n + 1) * sizeof(uint), cudaMemcpyHostToDevice);

    dim3 threadsPorBloco(32, 1, 1);
    dim3 blocosPorGrid(ceil( (double)n/32), 1, 1);

    // Executando o sieve
    for (uint i = 2; i <= ceil(sqrt(n)); ++i) {
    	if (h_array[i] != 0) {
            uint k = i*i;
            if (k <= n) {
				// Invoka o kernel para operações de mod
				sieve<<<blocosPorGrid, threadsPorBloco>>>(d_array, i, k, n);
				// Detecção de erro caso o kernel apresente algum...
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess) 
					printf("Error: %s\n", cudaGetErrorString(err));
				// Atualiza o array presente na CPU depois do resultado.
				cudaMemcpy(h_array, d_array, (n + 1) * sizeof(uint), cudaMemcpyDeviceToHost);
            }
       }
	}

    // Desaloca os espaços alocados.
    cudaFree(d_array);
}

void onHost(int argc, char* argv[]) {

	if (argc == 2) {
		uint n = get_n(argv[1]);
		if (n == 0) {
			std::cout << "Input error.\n";
		}
		// Open the output file
		FILE* outfile;
		outfile = fopen("problem_output.txt","w");
		if (outfile == NULL) {
			std::cout << "Problem opening output file.\n";
		}
		// Initialize array
		uint* h_array = new uint[n+1];
		for (uint i = 2; i <= n; ++i) {
			h_array[i] = i;
		}

		// Invoca o device (GPU) para processamento do array
		onDevice(h_array, n);

		// Exibe a saída no arquivo
		for (uint i = 2; i <= n; ++i)
			if (h_array[i] != 0)
				fprintf(outfile,"%u ", i);
		
		fclose(outfile);
		delete[] h_array;
	} else {
		std::cout << "Usage: ./sieve problem_input" << std::endl;
	}
}

int main(int argc, char* argv[]){
	onHost(argc, argv);
	return 0;
}