# Programação Paralela e Distribuída

Seja bem vindo! Repositório pessoal referente à displina Programação Paralela e Distribuída. Local usado para armazenar os trabalhos da disciplina, assim como programas relacionados ao assunto.

## Index dos programas
- [mult_matriz_openmp.c](https://github.com/FelTavoni/ProgramacaoParalelaDistribuida/blob/main/mult_matriz_openmp.c)
    - Programa para análise de paralelização sobre multiplicação de matrizes, utilizando OpenMP. Para compilação, usar `gcc mult_matriz_openmp.c -o mult_matriz_openmp -O2 -fopenmp` e para execução `./mult_matriz_openmp`.
- [mult_matriz_mpi.c](https://github.com/FelTavoni/ProgramacaoParalelaDistribuida/blob/main/mult_matriz_mpi.c)
    - Programa para análise de paralelização sobre multiplicação de matrizes, utilizando MPI. É necessário possui a extensão MPI. Para compilação, usar `mpicc mult_matriz_mpi.c -o mult_matriz_mpi` e para execução `mpirun -np <numero-de-processos> ./mult_matriz_mpi`.

### Autor

Felipe Tavoni - Graduando na Universidade Federal de São Carlos.