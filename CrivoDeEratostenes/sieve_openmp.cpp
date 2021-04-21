#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

typedef unsigned int uint;

// Gets the input value from the input file
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
     if (argc == 2) {
         
          uint n = get_n(argv[1]);
          if (n == 0) {
               std::cout << "Input error.\n";
               return 1;
          }
          // Open the output file
          FILE* outfile;
          outfile = fopen("problem_output.txt","w");
          if (outfile == NULL) {
               std::cout << "Problem opening output file.\n";
               return 1;
          }
          // Initialize array
          uint* array = new uint[n+1];
          for (uint i = 2; i <= n; ++i) {
               array[i] = i;
          }

          // Start the sieve

          // Removendo todos os pares
          #pragma omp parallel for
          for (uint i = 2; i <= n; ++i) {
               if (array[i] % 2 == 0) {
                    array[i] = 0;
               }
          }

          #pragma omp parallel for schedule(dynamic)
          for (uint i = 3; i <= n; i = i + 2) {
               if (array[i] != 0) {
                    // It's a prime. Print it!
                    uint k = i*i;
                    if (k <= n) {
                         for (uint j = k; j <= n; ++j) {
                              if (j % i == 0) array[j] = 0;
                         }
                    }
               }
          }

          for (uint i = 2; i < n; i++) {
              if (array[i] != 0)
                fprintf(outfile,"%u ", array[i]);
          }

          fclose(outfile);
          delete[] array;

          
     } else {
          std::cout << "Usage: ./sieve problem_input" << std::endl;
     }
}