#ifndef MAGMA_UTILS_H
#define MAGMA_UTILS_H

#ifndef min
#define min(a,b)   ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b)   ((a > b) ? a : b)
#endif

/*split N in two part according to a ratio, and return the size of the first part*/
int NSplit(int N, double ratio);

/* Abort the execution and print a message*/
int magma_async_abort(char *pattern, ...);

/* Number of blocks for thread tid after a distribution over P threads*/
int numBlock2p(int tid, int NBblock, int P);

/* local position of thread tid in a block column distributed over P threads */
int indexBlock2p(int tid, int NBblock, int P);

#endif