#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "magma_amc_utils.h"
/*split N in two part according to a ratio, and return the size of the first part*/
int NSplit(int N, double ratio){

    if(N==0) 
        return N;
    else 
        return max((int) (N * ratio),1);
}

/* Abort the execution and print a message*/
int magma_amc_abort(char *pattern, ...)
{
    va_list args;
    int len;
    char buffer[512];

    va_start(args, pattern);
        len = vsnprintf(buffer, 512 - 1, pattern, args);
    va_end(args);

    printf("%s\n",buffer);

#ifdef _WIN32
    system("pause");
#endif
    exit(1);
}

/* Number of blocks for thread tid after a distribution over P threads*/
int numBlock2p(int tid, int NBblock, int P)
{
/*
tid:    thread number
NBblock: number fo blocks
P:        Number de threads.
*/
    int rest,nb;

    rest = NBblock % P;
    nb = (NBblock-rest)/P;

    if(tid<rest) nb+=1;

    return nb;

}

/* local position of thread tid in a block column distributed over P threads */
int indexBlock2p(int tid, int NBblock, int P)
{
/*
tid:    identifiant du thread
NBblock: Nombre de blocs totals
P:        Nombre de threads actifs.
*/
    int rest,loc;

    rest = NBblock % P;

    loc = tid*(NBblock - rest)/P;

    if(tid<rest) loc+=tid;
    else loc+=rest;
return loc;
}