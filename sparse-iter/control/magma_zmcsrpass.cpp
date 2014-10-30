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

#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"


using namespace std;




/**
    Purpose
    -------

    Passes a CSR matrix to MAGMA.

    Arguments
    ---------

    @param
    m           magma_int_t 
                number of rows

    @param
    n           magma_int_t 
                number of columns

    @param
    row         magma_index_t*
                row pointer

    @param
    col         magma_index_t*
                column indices

    @param
    val         magmaDoubleComplex*
                array containing matrix entries

    @param
    A           magma_z_sparse_matrix*
                matrix in magma sparse matrix format

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t 
magma_zcsrset( 
    magma_int_t m, magma_int_t n, 
    magma_index_t *row, magma_index_t *col, magmaDoubleComplex *val,
    magma_z_sparse_matrix *A )
{
    A->num_rows = m;
    A->num_cols = n;
    A->nnz = row[m];
    A->storage_type = Magma_CSR;
    A->memory_location = Magma_CPU;
    A->val = val;
    A->col = col;
    A->row = row;

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Passes a MAGMA matrix to CSR structure.

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                magma sparse matrix in CSR format

    @param
    m           magma_int_t 
                number of rows

    @param
    n           magma_int_t 
                number of columns

    @param
    row         magma_index_t*
                row pointer

    @param
    col         magma_index_t*
                column indices

    @param
    val         magmaDoubleComplex*
                array containing matrix entries


    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t magma_zcsrget( 
    magma_z_sparse_matrix A,
    magma_int_t *m, magma_int_t *n, 
    magma_index_t **row, magma_index_t **col, magmaDoubleComplex **val )
{

    if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSR ){

        *m = A.num_rows;
        *n = A.num_cols;
        *val = A.val;
        *col = A.col;
        *row = A.row;
    } else {
        magma_z_sparse_matrix A_CPU, A_CSR;
        magma_z_mtransfer( A, &A_CPU, A.memory_location, Magma_CPU ); 
        magma_z_mconvert( A_CPU, &A_CSR, A_CPU.storage_type, Magma_CSR ); 
        magma_zcsrget( A_CSR, m, n, row, col, val );
        magma_z_mfree( &A_CSR );
        magma_z_mfree( &A_CPU );
    }
    return MAGMA_SUCCESS;
}


