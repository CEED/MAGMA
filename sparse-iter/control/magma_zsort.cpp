/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from
//  the IO functions provided by MatrixMarket

#include "magmasparse_internal.h"


#define SWAP(a, b) { tmp = val[a]; val[a] = val[b]; val[b] = tmp; }

/**
    Purpose
    -------

    Sorts an array of integers.

    Arguments
    ---------

    @param[in,out]
    x           magma_index_t*
                array to sort

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
magma_zindexsort(
    magma_index_t *x,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_index_t pivot,j,temp,i;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( x[i]<=x[pivot] && i<last )
                i++;
            while( x[j]>x[pivot] )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_zindexsort( x, first, j-1, queue ));
        CHECK( magma_zindexsort( x, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of integers, updates a respective array of values.

    Arguments
    ---------

    @param[in,out]
    x           magma_index_t*
                array to sort
                
    @param[in,out]
    y           magmaDoubleComplex*
                array to sort

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
magma_zindexsortval(
    magma_index_t *x,
    magmaDoubleComplex *y,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_index_t pivot,j,temp,i;
    magmaDoubleComplex tempval;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( x[i]<=x[pivot] && i<last )
                i++;
            while( x[j]>x[pivot] )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
                tempval = y[i];
                y[i] = y[j];
                y[j] = tempval;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        
        tempval=y[pivot];
        y[pivot]=y[j];
        y[j]=tempval;
        CHECK( magma_zindexsortval( x, y, first, j-1, queue ));
        CHECK( magma_zindexsortval( x, y, j+1, last, queue ));
    }
cleanup:
    return info;
}



/**
    Purpose
    -------

    Identifies the kth smallest/largest element in an array.

    Arguments
    ---------
    
    @param[in,out]
    val         magmaDoubleComplex*
                Target array, will be modified during operation.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in]
    k           magma_int_t
                Element to be identified (largest/smallest).
                
    @param[in]
    r           magma_int_t
                rule how to ort: '1' -> largest, '0' -> smallest
                
    @param[out]
    element     magmaDoubleComplex*
                location of the respective element
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zmorderstatistics(
    magmaDoubleComplex *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    magmaDoubleComplex *element,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t i, st;
    magmaDoubleComplex tmp;
 
    if( r == 0 ){
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( MAGMA_Z_ABS(val[i]) > MAGMA_Z_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st++;
        }
     
        SWAP(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_zmorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_zmorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    } else {
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( MAGMA_Z_ABS(val[i]) < MAGMA_Z_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st++;
        }
     
        SWAP(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_zmorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_zmorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    }
    
cleanup:
    return info;
}

