#include <stdio.h>
void magma_xerbla(char *name , int *info){
printf("MAGMA Error: On Routine %s argument number %d had an illegal value\n",name , *info);
}
