# O Crivo de Eratostenes

O estudo seguinte apresenta 3 diferentes propostas de paralelização ao algoritmo desenvolvido pelo matemático grego *Eratóstenes*, o terceiro bibliotecário-chefe da Biblioteca de Alexandria, no qual consiste em localizar todos os números primos de um intervalo n.

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/Sieve_of_Eratosthenes_animation.gif" alt="Sieve of Eratosthenes Example" style="display: block; margin: auto; width: 50%;">

---

## Preparação do ambiente

### Pré-requisitos

Para ser capaz de rodar os programas abaixo, explorando sua máxima performance, é necessário que o ambiente seja dotado de uma GPU e ser multicore. Além disso, se faz necessária a instalação da plataforma [CUDA](https://developer.nvidia.com/cuda-zone), além do compilador para [MPI](https://www.open-mpi.org/).

## Testando as aplicações

Os estudos a seguir foram construídos a partir do algoritmo sequencial fornecido pela [Maratona de Programação Paralela](http://lspd.mackenzie.br/marathon/08/problems.html) em sua terceira edição. Uma cópia do código sequencial pode ser visualizado [aqui](sieve.cpp).

### Programa OpenMP

O programa com OpenMP particiona a execução entre os múltiplos cores do ambiente por meio da criação de novas threads. Uma estratégia de paralelização pode ser criada a partir da paralelização de um for com a primitiva *Dynamic*, permitindo que threads executem uma nova iteração a medida que ficam ociosas. O número de threads a serem trabalhadas pelo sistema pode ser ajustada com o comando `export OMP_NUM_THREADS=<numero-de-threads>`.

Para executar o [programa OpenMP](sieve_openmp.cpp), o compile com o seguinte comando:
 
` g++ sieve_OMP.cpp -Wall -pedantic -O3 -Wno-unused-result -o sieve_OMP -fopenmp `

E o execute com:

` ./sieve_OMP <entrada>.in `

### Programa OpenMPI

O programa com OpenMPI particiona a execução entre os múltiplos processos compartilhados em um ambiente. Ao analisar uma propriedade interessante do crivo, o maior divisor primo de um número n se encontra no intervalo de 0 a raíz de n. Dessa forma, particionamos os dados em múltiplos processos, facilitando o processamento sobre eles.

Para executar o [programa OpenMPI](sieve_mpi.cpp), o compile com o seguinte comando:
 
` mpiCC sieve_mpi.cpp -Wall -pedantic -O3 -Wno-unused-result -o sieve_mpi `

E o execute com:

` mpirun -np <numero-de-processos> ./sieve_mpi <entrada>.in `

### Programa CUDA

O programa com CUDA envia o processamento à GPU e o serializa entre os cores disponíveis no componente. Com isso, a estratégia adotada consiste em enviar à placa de vídeo os dados e realizar sobre estes as operações de módulo, de modo a eliminar os múltiplos de determinado número, tudo em paralelo.

Para executar o [programa CUDA](sieve_cuda.cu), o compile com o seguinte comando:
 
` nvcc sieve_cuda.cu -o sieve_cuda `

E o execute com:

` ./sieve_cuda <entrada>.in `

### Programa Híbrido

Por fim, o programa híbrido combina a eficiência da divisão do espaço em diferentes processos proposta pela aplicação MPI, somado a paralelização serial provida pela GPU, que paraleliza cálculos em massa, eficiente ao tratar blocos grandes de dados.

Para executar o programa Híbrido ([MPI](sieve_hybrid_mpi.cpp) + [CUDA](sieve_hybrid_cuda.cu)), o compile com os seguintes comandos:
 
` mpiCC -c sieve_hybrid_mpi.cpp -o sieve_hybrid_mpi.o `

` nvcc -c sieve_hybrid_cuda.cu -o sieve_hybrid_cuda.o `

` mpiCC sieve_hybrid_mpi.o sieve_hybrid_cuda.o -L/usr/local/cuda/lib64 -lcudart -o sieve_hybrid `

E o execute com:

` mpirun -np <numero-de-processos> ./sieve_hybrid <entrada>.in `

## Autores

- **Felipe Tavoni** - *Graduando na Universidade Federal de São Carlos.*

- **Guilherme Locca Salomão** - *Graduando na Universidade Federal de São Carlos.*