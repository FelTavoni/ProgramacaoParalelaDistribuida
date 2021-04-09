/* 
 * Universidade Federal de São Carlos
 * Departamento de Computação
 * 
 * Autor: Felipe Tavoni - 758707
 *
 * O objetivo do código a seguir é desenvolver um código para realização da multiplicação de matrizes em paralelo, usando-se da plataforma CUDA, 
 *	que possibilita o uso de GPUs na resolução de atividades em paralelo.
 *
 * A seguir, paralelizamos a multiplicação da matriz da seguinte forma: como a alocação das atividades de processamento aos SMs é feita por meio
 *	dos blocos, e consequentemente subdividido em warps, podemos então ajustá-la para que a matriz seja repassada a GPU de forma reajustada e 
 *	múltiplo de 32. Isso permitirá uma melhor alocação aos SMs, além de evitar estouro, visto que o número de threads é limitado a 1024.
 *
 * INPUT: Linhas e colunas da matriz A, seguido de linhas e colunas da matriz B. (Determinadas dentro do código)
 * OUTPUT: O resultado da multiplicação das matrizes A e B, localizado na matriz C e o tempo de execução de cada uma delas.
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__ void multiplicaMatriz(float* d_A, float* d_B, float* d_C, int col_a, int col_b, int lin_c, int col_c) {
	// Definindo o índice a ser manipulado pela thread. Isso é necessário dado a presença de múltiplos blocos (dependendo da entrada).
	int lin = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Uma variável temporária é utilizada. Evitamos constante acesso a memória específica.
	float sum = 0.0f;

	/*
	 *	A multiplicação da matriz em um programa sequencial pode ser realizada de acordo com o seguinte trecho de código
	 *
	 *	for(i=0; i < lin_c; i++) {
     *  	for(j=0; j < col_c; j++) {
     *      	for(k=0; k < col_a; k++) {
     *          	C[i*col_c+j] = C[i*col_c+j] + A[i*col_a+k] * B[k*col_b+j];
     *          }
     *      }
     *  }
	 *
	 *	Logo, os 2 for's exernos podem ser eliminados, pois será executado em paralelo, thread por thread!.
	 */

	// Interessante verificar se os índices estão dentro do limite, já que os blocos cobrir mais que os escopo necessário da matriz.
	if ( (lin < lin_c) && (col < col_c) ) {

		// Então, cada thread será responsável pela execução de uma operação de multiplicação.
		for(int k = 0; k < col_a; k++) {
		    sum += d_A[lin * col_a + k] * d_B[k * col_b + col];
		}
		d_C[lin * col_c + col] = sum;

	}
}

// Preparação do ambiente de execução da GPU.
void onDevice(float* h_A, float* h_B, float* h_C, int lin_a, int col_a, int lin_b, int col_b, int lin_c, int col_c) {
	// Declarando os ponteiros das matrizes na GPU.
	float *d_A, *d_B, *d_C;

	// Alocando a memoria no espaço da GPU.
    cudaMalloc((void**)&d_A, lin_a*col_a*sizeof(float));
    cudaMalloc((void**)&d_B, lin_b*col_b*sizeof(float));
    cudaMalloc((void**)&d_C, lin_c*col_c*sizeof(float));

    // Copiando os valores para a memória da GPU e zerando a matriz reultado C.
    cudaMemcpy(d_A, h_A, lin_a * col_a * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, lin_b * col_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, lin_c * col_c * sizeof(float));

    // Ajustando a organização das threads. Como tratamos de matriz 2D, o campo 'z' permanece vazio.
    dim3 threadsPorBloco(32, 32, 1);											// Num threads - 32x32, de modo a facilitar a alocação aos SMs de acordo com os warps.
    dim3 blocosPorGrid(ceil( (double)lin_c/32), ceil( (double)col_c/32), 1);	// Num blocos  - Dividindo igualmente em parcelas. O arredondamento é necessário
    																			//	para evitar 'segmentation fault', acessando toda a matriz.

    // Chamando o kernel, para o então cálculo da multiplicação
    multiplicaMatriz<<< blocosPorGrid, threadsPorBloco >>>(d_A, d_B, d_C, col_a, col_b, lin_c, col_c);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
	    printf("Error: %s\n", cudaGetErrorString(err));

    // Calculado o valor, copiamos então o resultado de volta para a CPU
    cudaMemcpy(h_C, d_C, lin_c*col_c*sizeof(float), cudaMemcpyDeviceToHost);

    // Pronto! Resultado em mão, basta limpar o cenário da execução, desalocando as estruturas.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Fim!
}

// Código a ser executado na CPU
void onHost() {

	float *h_A, *h_B, *h_C;
    // Declaração de variáveis da quantidade de linhas e colunas das matrizes.
    int lin_a, col_a, lin_b, col_b, lin_c, col_c;

    // Quantidade de linhas da matriz A.
    printf("Linhas A: ");   
    scanf("%d", &lin_a);
    
    // Quantidade de colunas da matriz B (e C...dado a restrição do cálculo).
    printf("Colunas A / Linhas B: "); 
    scanf("%d", &col_a);
    lin_b = col_a;
    
    // Quantidade de colunas de B.
    printf("Colunas B: ");  
    scanf("%d", &col_b);
    printf("\n");
    
    // Quantidades de linha da matriz C (resultado da operação) de acordo com a restrição do cálculo.
    lin_c = lin_a;
    col_c = col_b;

    // Alocação dinâmica das matrizes, com linhas em sequência.
    h_A = (float *)malloc(lin_a*col_a*sizeof(float));
    h_B = (float *)malloc(lin_b*col_b*sizeof(float));
    h_C = (float *)malloc(lin_c*col_c*sizeof(float));

    // Atribucao de valores randômicos iniciais as matrizes 
    // *Matrizes contíguas, unidimensionais*
    srand(time(NULL));

    for(int i = 0; i < lin_a * col_a; i++) 
        h_A[i] = (float)rand() / (float)RAND_MAX;; 

    for(int i = 0; i < lin_b * col_b; i++) 
        h_B[i] = (float)rand() / (float)RAND_MAX;;

    // Alocada as estruturas, invocar a execução na GPU
    onDevice(h_A, h_B, h_C, lin_a, col_a, lin_b, col_b, lin_c, col_c);

    // Marcando tempo pós execução.

    // Após o processamento na CPU, temos então nosso resultado armazenado em h_C.
    // Imprimimos o resultado.
    for (int i = 0; i < lin_c; i++) {
    	for (int j = 0; j < col_c; j++) {
    		printf("%5.2f ", h_C[i * col_c + j]);
    	}
    	printf("\n");
    }

    // Utilizada a matriz para determinado propósito, limpamos o cenário da execução novamente, desalocando estruturas alocadas.
    free(h_A);
    free(h_B);
    free(h_C);

    // The End!
}

int main(int argc, char *argv[]) {
	
	onHost();

    return 0;
}