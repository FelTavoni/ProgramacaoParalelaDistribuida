#include <iostream>
#include <stdio.h>
#include <math.h>
#include "mpi.h"

typedef unsigned int uint;

// Variáveis locais de cada processo
int numtasks;
int taskid;
uint n;

// Função que obtém o valor de entrada presente no arquivo.
uint get_n(char* filename) {
     FILE* infile;
     infile = fopen(filename,"r");
     if (infile != NULL) {
          uint n;
          fscanf(infile,"%u", &n);
          fclose(infile);
          return n;
     } else {
          return 0;
     }
}

int main(int argc, char* argv[]){

     // Inicialização do ambiente de execução MPI.
     MPI_Init(&argc, &argv);                           // Quantidade de processos a serem trabalhados.
     MPI_Comm_size(MPI_COMM_WORLD, &numtasks);         // Obtendo o número de tarefas no ambiente.
     MPI_Comm_rank(MPI_COMM_WORLD, &taskid);           // Obtém a identificação 'id' do processo corrente dentro do ambiente.

     // Obtendo a entrada do arquivo. O processo mestre o lê e os outros receberão, via broadcast.
     if (taskid == 0) {
          if (argc == 2) {
               n = get_n(argv[1]);
               if (n == 0) {
                    std::cout << "Input error.\n";
                    MPI_Abort(MPI_COMM_WORLD, 0);
                    return 1;
               }
               // Cria o arquivo se não existir
               FILE* outfile;
               outfile = fopen("problem_output.txt","w");
               if (outfile == NULL) {
                    std::cout << "Problem opening output file.\n";
                    MPI_Abort(MPI_COMM_WORLD, 0);
                    return 1;
               }
               fclose(outfile);
          } else {
               std::cout << "Usage: ./sieve problem_input" << std::endl;
               MPI_Abort(MPI_COMM_WORLD, 0);
          }
     }

     // Realiza uma chamada broadcast para que todos os outros processos recebam a entrada. 
     // Uma barreira aqui se encontra para que o tamanho dos blocos sejam alocados corretamente.
     MPI_Barrier(MPI_COMM_WORLD);
     MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

     // Verificação se o array do mestre possuirá a raíz de n.
     if ( !((double)(n/numtasks) > sqrt(n)) ) {
          std::cout << "Muitos processos declarados! O mestre não cobrirá todos primos..." << std::endl;
          MPI_Finalize();
          return 0;
     }

     // Inicialização do array de cada processo.

     // Divisão dos blocos. Caso a divisão seja inteira, entraremos apenas no segundo for.
     uint block_size = 0, first = 0;
     for (uint i = 0; i < (n % numtasks); ++i) {
          if (taskid == (int)i) {
               block_size = ceil( (double) n / numtasks);
               first = taskid*block_size;
          }
     }
     for (int i = (n % numtasks); i < numtasks; ++i) {
          if (taskid == i) {
               block_size = floor( (double) n / numtasks);
               first = taskid*block_size;
          }
     }

     // Alocando os blocos, agora parcelados, em cada processo.
     uint* array = new uint[block_size];
     for (uint i = 0; i < block_size; ++i) {
          array[i] = first + i;
     }

     // No processo mestre, os elemento '0' e '1' são anulados, dado que não devem influenciar no teste de números primos.
     // O processo mestre então aloca uma estrutura para receber os dados de todos os demais processos.
     uint *all_array = NULL;
     if (taskid == 0) {
          array[0] = 0;
          array[1] = 0;
          all_array = new uint[n+1];
     }

     // Novamente uma barreira para evitar acesso errôneo de memória.
     MPI_Barrier(MPI_COMM_WORLD);

     // Iniciando o crivo...
     // Começamos com o primeiro elemento, o número 2.
     uint k = 2;

     // Dado o número de elementos da série, o número máximo capaz de dividir o elemento n deverá estar antes (ou é) de sua raiz quadrada. Todos os
     //   elementos após a raís de n serão múltiplos de elementos anteriores, já processados. Assim, basta que o algoritmo seja regido pelo array 
     //   do processo 0. Portanto, todos os elemento nos outros arrays já serão primos.
     while (k*k < n) {

          // Para alcançar o próximo múltiplo, temos o quociente menos o resto. A soma desses resultará um múltiplo do quociente.
          if ((array[0] % k) != 0) {
               first = k - (array[0] % k);
          } else {
               first = 0;
          }

          // Eliminando todos os múltiplos de k do array do processo 'taskid'.
          for (uint i = first; i < block_size; i = i + k) {
               array[i] = 0;
          }

          // Após o processamento, o array coloca o número primo no arquivo, e já busca o próximo a ser tratado.
          if (taskid == 0) {
               FILE *outfile;
               outfile = fopen("problem_output.txt","a+"); 
               fprintf(outfile, "%d ", k);
               uint i = 0;
               while (array[i] % k == 0)
                    i++;
               k = i;
               fclose(outfile);
          }
          // Localizado o próximo primo, enviá-lo a todos os outros processo.
          MPI_Bcast(&k, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
          // Uma mera barreira para que todos os outros processos estejam com o 'k' atualizado.
          MPI_Barrier(MPI_COMM_WORLD);

     }

     // Todos os processos enviando ao mestre sua parcela de dados
     MPI_Gather(array, block_size, MPI_UNSIGNED, all_array, block_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

     // Processo mestre então imprime todos os primos
     if (taskid == 0) {
        FILE* outfile;
        outfile = fopen("problem_output.txt","a+");
        for (uint i = 0; i < n+1; i++) {
            if (all_array[i] != 0)
                fprintf(outfile, "%i ", all_array[i]);
        }
        fclose(outfile);
     }

     MPI_Finalize();
}