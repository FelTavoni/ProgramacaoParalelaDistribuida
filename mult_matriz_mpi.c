/* 
 * Universidade Federal de São Carlos
 * Departamento de Computação
 * 
 * Autor: Felipe Tavoni - 758707
 *
 * O objetivo do código a seguir é desenvolver um código para realização da multiplicação de matrizes em paralelo, usando computadores interligados em rede.
 * Para tanto, consideraremos o paradigma de passagem de mensagem utilizando a especificação MPI. 
 *
 * Consideraremos o particionamento da matriz com a divisão desta em blocos. Além disso, o programa a seguir não trata multiplicação de matrizes caso a di-
 *	visão de blocos aos nós não seja exata.
 *
 * INPUT: Linhas e colunas da matriz A, seguido de linhas e colunas da matriz B. (Determinadas dentro do código)
 * OUTPUT: O resultado da multiplicação das matrizes A e B, localizado na matriz C e o tempo de execução de cada uma delas.
 * 
 */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "mpi.h"

MPI_Status status;

int main(int argc, char *argv[]) {
	// As variáveis declarandas a seguir são utilizadas por todos os processos. Caso seu valor seja alterado, deve ocorrer um envio aos demais para atualização.

	// Declaração de estruturas para medição do tempo (desempenho).
    float etime;						// Float para cálculo de precisão. (tempo_inicial - tempo_final).
    struct timespec inic, fim;			// Estrutura a armazenar os dados com a chamada de clock_gettime.

    // Variáveis de controle para cálculo da multiplicação.
    int lin_a, col_a, lin_b, col_b, lin_c, col_c;	// Dimensões das matrizes
    int i, j, k;    								// Contadores

    // Variáveis de controle para os nós da aplicação. Número de tarefas e o identificador da tarefa.
    int numtasks, taskid;

    // Determinando as dimensões da matriz.
	lin_a = 100;
	// col_a == col_b
	col_a = 100;
	lin_b = 100; 
	col_b = 100; 
	lin_c = lin_a;
	col_c = col_b;

	// Declarando e alocando as matrizes.
	// A alocação contígua permite um ganho de desempenho, por não possuir acessos randômicos característicos de matrizes 2D.
	float *A = (float *)malloc(lin_a*col_a*sizeof(float));
    float *B = (float *)malloc(lin_b*col_b*sizeof(float));
    float *C = (float *)malloc(lin_c*col_c*sizeof(float));

    // Inicialização do ambiente de execução MPI.
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);		// Obtendo o número de tarefas no ambiente.
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);			// Obtém a identificação 'id' do processo corrente dentro do ambiente.

	// O número de trabalhadores é igual ao total de nós disponíveis, menos 1 (o nó master).
	int numworkers = numtasks - 1;
	// Variáveis utilizadas para comunicação. 'node_col_ini' indica a linha a qual a matriz de começar seu processamento. 'lin_per_workers' indica quantas 
	//  linhas cada matriz deve processar.
	int node_col_ini, lin_per_workers;

	// ******************************
	// NÓ PRINCIPAL (NÓ MESTRE)
	// ******************************

	if (taskid == 0) {
	    // Atribucao de valores randômicos iniciais as matrizes 
	    srand(time(NULL));

	    for(i = 0; i < lin_a * col_a; i++) 
        	A[i] = (float)rand() / (float)RAND_MAX;

    	for(i = 0; i < lin_b * col_b; i++) 
        	B[i] = (float)rand() / (float)RAND_MAX;

        // Iniciando medição do tempo
        clock_gettime(CLOCK_REALTIME, &inic);

	    // Podemos então dividir a matriz em submatrizes com linhas N = total_de_linhas / processos.
	    // Assim, realizamos o envio aos demais processos das informações para processamento.
	    // -- MPI_Send(endereço inicial, bytes a partir do endereço, tipo de dados do envio, id do nó destino, tag da mensagem, MPI_Comm comm) --
	    node_col_ini = 0;																					// Inicializando a variável.
	    lin_per_workers = (lin_a/numworkers);																// Determinando as linhas de cada trabalhador.
	    for (int dest = 1; dest <= numworkers; dest++) {													// Para cada destino...
			MPI_Send(&node_col_ini, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);									//  - Envio da posição incial;
			MPI_Send(&lin_per_workers, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);								//  - Total de linhas a trabalhar;
			MPI_Send(&A[node_col_ini*col_a], lin_per_workers*col_a, MPI_FLOAT,dest,1, MPI_COMM_WORLD);		//  - Os elementos de A[linha_ini] a A[linha_fim];
			MPI_Send(&B[0], lin_b*col_b, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);								//  - Envio da matriz B completa.
			node_col_ini = node_col_ini + lin_per_workers;													// Incremento do número de linhas para próximo envio.
		}

		// Nó mestre então aguarda o retorno do processamento de cada um dos nós trabalhadores, recebendo-os quando disponível.
		// int MPI_Recv(endereço inicial, bytes a partir do endereço, tipo de dados do envio, fonte, tag da mensagem, MPI_Comm comm, MPI_Status *status)
		for (int fonte = 1; fonte <= numworkers; fonte++) {															// De cada fonte...
			MPI_Recv(&node_col_ini, 1, MPI_INT, fonte, 2, MPI_COMM_WORLD, &status);									//  - Recebe a posição inicial que o nó tratou;
      		MPI_Recv(&lin_per_workers, 1, MPI_INT, fonte, 2, MPI_COMM_WORLD, &status);								//  - Recebe a quantidade de linhas tratada pelo nó;
			MPI_Recv(&C[node_col_ini*col_c], lin_per_workers*col_c, MPI_DOUBLE, fonte, 2, MPI_COMM_WORLD, &status);	//  - Recebe então os dados resultantes e grava-os em C na posição correta.
		}

		// Finalizando a medição de tempo.
		clock_gettime(CLOCK_REALTIME, &fim);

		// Impressão da matriz resultado C
		for (i=0; i < lin_c; i++) {
			for (j=0; j < col_c; j++) {
				printf("%5.2f", C[i*col_c + j]);
			}
			printf ("\n");
		}

		// Tempo decorrido: elapsed time
		etime = (fim.tv_sec + fim.tv_nsec/1000000000.) - (inic.tv_sec + inic.tv_nsec/1000000000.);
		printf("Elapsed time: %.3f\n", etime);
	}

	// ******************************
	// DEMAIS NÓS (NÓS TRABALHADORES)
	// ******************************

	// Nós trabalhadores. Recebem o conteúdo enviado pelo nó mestre e processam a multiplicação.
	if (taskid > 0) {

		int fonte = 0;
		MPI_Recv(&node_col_ini, 1, MPI_INT, fonte, 1, MPI_COMM_WORLD, &status);					// Obtém a linha inicial a qual iniciará sua operação;
		MPI_Recv(&lin_per_workers, 1, MPI_INT, fonte, 1, MPI_COMM_WORLD, &status);				// Obtém a parcela das linhas a serem trabalhadas;
		MPI_Recv(&A[0], lin_per_workers*col_a, MPI_DOUBLE, fonte, 1, MPI_COMM_WORLD, &status);	// Armazena o conteúdo recebido em sua matriz A;
		MPI_Recv(&B[0], lin_b*col_b, MPI_DOUBLE, fonte, 1, MPI_COMM_WORLD, &status);			// Armazena o conteúdo recebido em sua matriz B.

		//Multiplicação da matriz.
		for(i=0; i < lin_per_workers; i++) {
            for(j=0; j < col_c; j++) {
            	C[i*col_c+j] = 0.0;
                for(k=0; k < col_a; k++) {
                    C[i*col_c+j] = C[i*col_c+j] + A[i*col_a+k] * B[k*col_b+j];
                }
            }
        }

		MPI_Send(&node_col_ini, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);					// Retorna a linha inicial a qual iniciou;
		MPI_Send(&lin_per_workers, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);				// Retorna a quantidade de linhas trabalhadas;
		MPI_Send(&C[0], lin_per_workers*col_c, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);	// Retorna o resultado da multiplicação.
		// Vale observar que a multiplicação não deve ser grava em um local específico na matriz C. Isso será a cargo da função Recv do nó mestre.
		//	Basta que ele mantenha no ínicio da matriz seus cálculos.
	}

	// Finaliza a execução em MPI.
    MPI_Finalize();

    // Libera as matrizes alocadas.
    free(A);
    free(B);
    free(C);

    return 0;
}