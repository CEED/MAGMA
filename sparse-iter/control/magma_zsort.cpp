/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from 
//  the IO functions provided by MatrixMarket

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"


using namespace std;




/**
    Purpose
    -------

    Sorts an array of integers.

    Arguments
    ---------

    @param[in]
    x           magma_index_t*
                array to sort

    @param[in]
    size        magma_int_t 
                size of the array to sort

    @param[in]
    first       magma_int_t 
                pointer to first element

    @param[in]
    last        magma_int_t 
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zsort(
    magma_index_t x, 
    magma_int_t size,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t pivot,j,temp,i;

    if ( size != last-first-1 ){

        printf('warning: the size of the array to sort is not consistent to first and last element.\n');
    }

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while(x[i]<=x[pivot]&&i<last)
                i++;
            while(x[j]>x[pivot])
                j--;
            if(i<j){
                temp=x[i];
                x[i]=x[j];
                x[j]=temp;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        magma_zsort( x, (j-1)-first-1, first, j-1);
        magma_zsort( x, last-(j+1)-1j+1, last);

    }

    return MAGMA_SUCCESS;

}

