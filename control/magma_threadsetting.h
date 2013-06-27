/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Azzam Haidar
*/

#ifndef _MAGMA_THREADSETTING_H_
#define _MAGMA_THREADSETTING_H_

#ifdef __cplusplus
extern "C" {
#endif
/***************************************************************************//**
 *  Internal routines
 **/
void magma_setlapack_multithreads(int numthreads);
void magma_setlapack_sequential();
void magma_setlapack_numthreads(int numthreads);
magma_int_t magma_get_numthreads();
/***************************************************************************/
#ifdef __cplusplus
}
#endif

#endif
