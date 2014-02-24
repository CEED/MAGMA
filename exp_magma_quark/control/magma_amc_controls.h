/* 
    -- MAGMA (version 1.3) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/
/*The routines control the initialisation of the routines called in magma_amc*/
#ifndef MAGMA_ASYNC_CONTROLS_H
#define MAGMA_ASYNC_CONTROLS_H

int magma_amc_init(int P, double dcpu, int Pr, int nb);
int magma_amc_finalize();
#endif
