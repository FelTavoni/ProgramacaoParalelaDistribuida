/* 
 * Universidade Federal de São Carlos
 * Departamento de Computação
 * 
 * Autor: Felipe Tavoni - 758707
 *
 * O objetivo do código a seguir é investigar, usando o mecanismo de paralelização de aplicações com OpenMP, a escalabilidade do programa de 
 *  multiplicação de matrizes. Para tanto, deve-se ajustar o código para a paralização de apenas um dos loops do cálculo de cada vez. 
 *
 * Após estudo e testes com as diferentes abordagens de paralelização, pode-se perceber que a melhor abordagem é a paralelização do for 1, o
 *  for mais externo. O que torna o loop melhor "paralelizável" é o fato de  não ter que reentrar no bloco de paralelização várias vezes, co-
 *  mo ocorre em outros for's, criando e 'destruindo' as threads o tempo todo, aumentando significativamento a "thread overheading".
 *
 * Na função main() abaixo encontra-se um switch, para a execução de diferentes abordagens de paralelização, a fins de comparação. Cada caso
 *  é destinado a explorar uma ocasião possível de paralelização, explicitando o efeito resultante das paralelizações, como prós e contras.
 *  
 * Os tempos medidos nas operações abaixo são apenas referentes à multiplicação das matrizes.
 *
 * INPUT: Linhas da matriz A, Colunas/Linhas das 2 matrizes e colunas da matriz B.
 * OUTPUT: O resultado da multiplicação das matrizes A e B, localizado na matriz C e o tempo de execução de cada uma delas.
 * 
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

float *A, *B, *C;

int main(int argc, char *argv[])
{

    float etime;
    struct timespec inic, fim;

    // Declaração de variáveis da quantidade de linhas e colunas das matrizes.
    int lin_a, col_a, lin_b, col_b, lin_c, col_c;
    int i, j, k;    // Contadores

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
    A = (float *)malloc(lin_a*col_a*sizeof(float));
    B = (float *)malloc(lin_b*col_b*sizeof(float));
    C = (float *)malloc(lin_c*col_c*sizeof(float));

    // Atribucao de valores randômicos iniciais as matrizes 
    // *Matrizes contíguas, unidimensionais*
    srand(time(NULL));

    for(i = 0; i < lin_a * col_a; i++) 
        A[i] = (float)rand() / (float)RAND_MAX; 

    for(i = 0; i < lin_b * col_b; i++) 
        B[i] = (float)rand() / (float)RAND_MAX; 

    // Utilizar o número de máx de threads permitido pelo OpenMp (número de cores).
    int processadores = omp_get_max_threads();
    printf("%d processadores disponíveis! Usando todos.\n\n", processadores);
    omp_set_num_threads(processadores);

    // Cálculo da multiplicacao.

    int opcao;
    printf("Qual métrica deseja realizar?\n1 - Sem paralelização\n2 - Paralelizado o primeiro for\n3 - Paralelizado o segundo for\n4 - Paralelizado o terceiro for (reduction)\n5 - Paralelizado o terceiro for (critical)\n6 - Todos os for's paralelizados\n\nOpcao: ");
    scanf("%d", &opcao);

    float aux;

    switch (opcao) {
        // ** SEM PARALELIZAÇÃO **
        // O código a seguir não apresenta paralelização, simples multiplicação de matrizes.
        case 1:
            clock_gettime(CLOCK_REALTIME, &inic);
            for(i=0; i < lin_c; i++) {
                for(j=0; j < col_c; j++) {
                    for(k=0; k < col_a; k++) {
                        C[i*col_c+j] = C[i*col_c+j] + A[i*col_a+k] * B[k*col_b+j];
                    }
                }
            }
            clock_gettime(CLOCK_REALTIME, &fim);
            break;

        // ** PARALELIZADO O PRIMEIRO FOR **
        //  O primeiro loop paraleliza a aplicação ao dividir as linhas da Matriz C entre as threads. 
        //  Assim, cada thread é responsável por calcular todas as multiplicações e somas referente aos elementos daquela linha.
        //  Ex: thread 0 executa as operações para C[0][0...n], thread 1 executa as operações para C[1][0...n], e assim por diante.
        case 2:
            clock_gettime(CLOCK_REALTIME, &inic);
            #pragma omp parallel for private(j, k)
            for(i=0; i < lin_c; i++) {
                for(j=0; j < col_c; j++) {
                    for(k=0; k < col_a; k++) {
                        C[i*col_c+j] = C[i*col_c+j] + A[i*col_a+k] * B[k*col_b+j];
                    }
                }
            }
            clock_gettime(CLOCK_REALTIME, &fim);
            break;

        // ** PARALELIZANDO O SEGUNDO FOR **
        //  O segundo loop paraleliza a aplicação ao dividir as colunas da Matriz C entre as threads. 
        //  Assim, a cada iteração de linhas (ou seja, a cada linha), cada thread é responsável por calcular a multiplicação do j-ésimo elemento da Matriz C.
        //  Ex: Na iteração da linha 0, a thread 0 executa a multiplicação para C[0][0], thread 1 executa a multiplicação para C[0][1], e assim por diante.
        case 3:
            clock_gettime(CLOCK_REALTIME, &inic);
            for(i=0; i < lin_c; i++) {
                #pragma omp parallel for private(k)
                for(j=0; j < col_c; j++) {
                    for(k=0; k < col_a; k++) {
                        C[i*col_c+j] = C[i*col_c+j] + A[i*col_a+k] * B[k*col_b+j];
                    }
                }
            }
            clock_gettime(CLOCK_REALTIME, &fim);
            break;

        // ** PARALELIZANDO O TERCEIRO FOR (REDUCTION) **
        //  O terceiro loop paraleliza a aplicação ao dividir cada operação de multiplicação para cada thread. 
        //  A cada operação multiplicação da Matriz C, cada thread executa uma multiplicação referente à aquele cálculo. Isso pode gerar inconsistência,
        //      dado a velocidade de execução. As threads podem sobreescrever valores referentes à soma. Assim, a diretiva 'reduction' priva uma variável
        //      com o objetivo de evitar a concorrência (e bloqueio) de escrita das threads a um mesmo local.
        //  Ex: Na operação para C[0][0], a thread 0 executa a multiplicação de A[0][0]*B[0][0], guardando em aux o resultado. A thread 1 executa a multi-
        //      plicação de A[0][1]*B[1][0], guardando em aux o resultado e assim por diante (com aux privado). Ao fim, aux é escrito em C[0][0].
        case 4:
            clock_gettime(CLOCK_REALTIME, &inic);
            for(i=0; i < lin_c; i++) {
                for(j=0; j < col_c; j++) {
                    aux = 0.0;
                    #pragma omp parallel for reduction(+ : aux)
                    for(k=0; k < col_a; k++) {
                        aux += A[i*col_a+k] * B[k*col_b+j];
                    }
                    C[i*col_c+j] = aux;
                }
            }
            clock_gettime(CLOCK_REALTIME, &fim);
            break;
            
        // ** PARALELIZANDO O TERCEIRO FOR (CRITICAL) **
        //  O terceiro loop paraleliza a aplicação ao dividir cada operação de um elemento da multiplicação da matriz C a cada thread. 
        //  A cada operação multiplicação da Matriz C, cada thread executa uma multiplicação referente à aquele cálculo. Isso pode gerar inconsistência,
        //      dado a velocidade de execução. As threads podem sobreescrever valores referentes à soma. Assim, a diretiva 'critical' indica ao compila-
        //      dor uma região critica, no qual a escrita deve ser bloqueante, evitando a inconsistência, mas causando uma queda no desempenho dependen-
        //      do o número de threads.
        //  Ex: Na operação para C[0][0], a thread 0 executa a multiplicação de A[0][0]*B[0][0], e requisita a escrita em C[0][0]. A thread 1 executa a 
        //      multiplicação de A[0][1]*B[1][0], e o mesmo ocorre. Ao fim, as escritas são então processadas.
        case 5:
            clock_gettime(CLOCK_REALTIME, &inic);
            for(i=0; i < lin_c; i++) {
                for(j=0; j < col_c; j++) {
                    #pragma omp parallel for
                    for(k=0; k < col_a; k++) {
                        #pragma omp critical
                        C[i*col_c+j] = C[i*col_c+j] + A[i*col_a+k] * B[k*col_b+j];
                    }
                }
            }
            clock_gettime(CLOCK_REALTIME, &fim);
            break;

        // ** PARALELIZANDO TODOS OS FOR'S **
        //  Ao paralelizar todos os for's, teremos um crescimento exponencial de threads, cada uma realizando uma função dor for's acima. 
        //  A princípio, essa operação não causará significativos ganhos, dado a recorrente troca de contexto entra as threads para execução.
        //  Pensando no exemplo, utilizando 4 cores, teremos 64 threads criadas para dividirem 4 cores disponíveis...
        case 6:
            omp_set_nested(1); // Ativando o paralelismo aninhado (Cada thread virará uma thread master ao encontrar um bloco, que criará um novo time).
            clock_gettime(CLOCK_REALTIME, &inic);
            #pragma omp parallel for private(j, k)
            for(i=0; i < lin_c; i++) {
                #pragma omp parallel for private(k)
                for(j=0; j < col_c; j++) {
                    aux = 0.0;
                    #pragma omp parallel for reduction(+ : aux)
                    for(k=0; k < col_a; k++) {
                        aux += A[i*col_a+k] * B[k*col_b+j];
                    }
                    C[i*col_c+j] = aux;
                }
            }
            clock_gettime(CLOCK_REALTIME, &fim);
            break;
        
        default:
            printf("Nenhuma opção escolhida! Interrompendo execução...");
            exit(0);
    }

    // Tempo decorrido: elapsed time
    etime = (fim.tv_sec + fim.tv_nsec/1000000000.) - (inic.tv_sec + inic.tv_nsec/1000000000.);
    printf("Elapsed time: %.3f\n", etime);

    return 0;
}