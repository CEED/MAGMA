/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>

#define AVOID_DUPLICATES
//#define NANCHECK

/**
    Purpose
    -------
    This function does an iterative ILU sweep.

    Arguments
    ---------
    
    @param[in,out]
    A          magma_z_matrix*
                Current ILU approximation 
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.

    @param[in,out]
    L           magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_sweep_list(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n");fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L->nnz; e++){
        // as we look at the lower triangular, 
        // col<row, i.e. disregard last element in row
        if( L->list[e] > 0 ){
            magma_int_t i,j,icol,jcol,jold;
            
            magma_index_t row = L->rowidx[ e ];
            magma_index_t col = L->col[ e ];
            
            //printf("(%d,%d) ", row, col);fflush(stdout);
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A->row[row]; i<A->row[row+1]; i++){
                if( A->col[i] == col ){
                    A_e = A->val[i];
                    i = A->row[row+1];
                }
            }
            
            //now do the actual iteration
            i = L->row[ row ]; 
            j = U->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magma_int_t endj = U->row[ col+1 ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                jold = j;
                icol = L->col[i];
                jcol = U->col[j];
                if( icol == jcol ){
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i++;
                    j++;
                }
                else if( icol<jcol ){
                    i++;
                }
                else {
                    j++;
                }
            }while( i<endi && j<endj );
            sum = sum - lsum;
            
            // write back to location e
            L->val[ e ] =  ( A_e - sum ) / U->val[jold];
        } else if( L->list[e]==0 ){ // end check whether part of L
            L->val[ e ] = MAGMA_Z_ONE; // lower triangular has diagonal equal 1
        }
        
    }// end omp parallel section
    
   #pragma omp parallel for
    for( magma_int_t e=0; e<U->nnz; e++){
        // as we look at the lower triangular, 
        // col<row, i.e. disregard last element in row
        if( U->list[e] != -1 ){
            magma_int_t i,j,icol,jcol,jold;
            
            magma_index_t row = U->rowidx[ e ];
            magma_index_t col = U->col[ e ];
            
            //printf("(%d,%d) ", row, col);fflush(stdout);
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A->row[row]; i<A->row[row+1]; i++){
                if( A->col[i] == col ){
                    A_e = A->val[i];
                    i = A->row[row+1];
                }
            }
            
            //now do the actual iteration
            i = L->row[ row ]; 
            j = U->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magma_int_t endj = U->row[ col+1 ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                jold = j;
                icol = L->col[i];
                jcol = U->col[j];
                if( icol == jcol ){
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i++;
                    j++;
                }
                else if( icol<jcol ){
                    i++;
                }
                else {
                    j++;
                }
            }while( i<endi && j<endj );
            sum = sum - lsum;
            
            // write back to location e
            U->val[ e ] =  ( A_e - sum );
        } 
        
    }// end omp parallel section
    
    
    
    return info;
}



/**
    Purpose
    -------
    This function does an iterative ILU sweep.

    Arguments
    ---------
    

    @param[in,out]
    L           magma_z_matrix
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_residuals_list(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // printf("start\n");fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L_new->nnz; e++){
        // as we look at the lower triangular, 
        // col<row, i.e. disregard last element in row
        {
            magma_int_t i,j,icol,jcol,jold;
            
            magma_index_t row = L_new->rowidx[ e ];
            magma_index_t col = L_new->col[ e ];
            if( row != 0 || col != 0 ){
                
                // printf("(%d,%d) ", row, col);fflush(stdout);
                magmaDoubleComplex A_e = MAGMA_Z_ZERO;
                // check whether A contains element in this location
                for( i = A.row[row]; i<A.row[row+1]; i++){
                    if( A.col[i] == col ){
                        A_e = A.val[i];
                        i = A.row[row+1];
                    }
                }
                
                //now do the actual iteration
                i = L.row[ row ]; 
                j = U.row[ col ];
                magma_int_t endi = L.row[ row+1 ];
                magma_int_t endj = U.row[ col+1 ];
                magmaDoubleComplex sum = MAGMA_Z_ZERO;
                magmaDoubleComplex lsum = MAGMA_Z_ZERO;
                do{
                    lsum = MAGMA_Z_ZERO;
                    jold = j;
                    icol = L.col[i];
                    jcol = U.col[j];
                    if( icol == jcol ){
                        lsum = L.val[i] * U.val[j];
                        sum = sum + lsum;
                        i++;
                        j++;
                    }
                    else if( icol<jcol ){
                        i++;
                    }
                    else {
                        j++;
                    }
                }while( i<endi && j<endj );
                sum = sum - lsum;
                
                // write back to location e
                if( row>col ){
                    L_new->val[ e ] =  ( A_e - sum ) / U.val[jold];
                } else {
                    L_new->val[ e ] =  ( A_e - sum );
                }
            } else {
                L_new->val[ e ] = MAGMA_Z_ZERO;
            }
        } 
        
    }// end omp parallel section
    
    return info;
}



/**
    Purpose
    -------
    This function does an iterative ILU sweep.

    Arguments
    ---------
    
    @param[in,out]
    A          magma_z_matrix*
                Current ILU approximation 
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.

    @param[in,out]
    L           magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_sweep_linkedlist(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n");fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L->nnz; e++){
        // as we look at the lower triangular, 
        // col<row, i.e. disregard last element in row
        if( L->list[e] > 0 ){
            magma_int_t i,j,icol,jcol,jold;
            
            magma_index_t row = L->rowidx[ e ];
            magma_index_t col = L->col[ e ];
            
            //printf("(%d,%d) ", row, col);fflush(stdout);
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A->row[row]; i<A->row[row+1]; i++){
                if( A->col[i] == col ){
                    A_e = A->val[i];
                    i = A->row[row+1];
                }
            }
            
            //now do the actual iteration
            i = L->row[ row ]; 
            j = U->row[ col ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                jold = j;
                icol = L->col[i];
                jcol = U->col[j];
                if( icol == jcol ){
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i = L->list[i];
                    j = U->list[j];
                }
                else if( icol<jcol ){
                    i = L->list[i];
                }
                else {
                    j = U->list[j];
                }
            }while( i!=0 && j!=0 );
            sum = sum - lsum;
            
            // write back to location e
            L->val[ e ] =  ( A_e - sum ) / U->val[jold];
        } else if( L->list[e]==0 ){ // end check whether part of L
            L->val[ e ] = MAGMA_Z_ONE; // lower triangular has diagonal equal 1
        }
        
    }// end omp parallel section
    
   #pragma omp parallel for
    for( magma_int_t e=0; e<U->nnz; e++){
        // as we look at the lower triangular, 
        // col<row, i->e-> disregard last element in row
        if( U->list[e] != -1 ){
            magma_int_t i,j,icol,jcol,jold;
            
            magma_index_t row = U->rowidx[ e ];
            magma_index_t col = U->col[ e ];
            
            //printf("(%d,%d) ", row, col);fflush(stdout);
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A->row[row]; i<A->row[row+1]; i++){
                if( A->col[i] == col ){
                    A_e = A->val[i];
                    i = A->row[row+1];
                }
            }
            
            //now do the actual iteration
            i = L->row[ row ]; 
            j = U->row[ col ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                jold = j;
                icol = L->col[i];
                jcol = U->col[j];
                if( icol == jcol ){
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i = L->list[i];
                    j = U->list[j];
                }
                else if( icol<jcol ){
                    i = L->list[i];
                }
                else {
                    j = U->list[j];
                }
            }while( i!=0 && j!=0 );
            sum = sum - lsum;
            
            // write back to location e
            U->val[ e ] =  ( A_e - sum );
        } 
        
    }// end omp parallel section
    
    
    
    return info;
}


/**
    Purpose
    -------
    This function does an iterative ILU sweep.

    Arguments
    ---------
    

    @param[in,out]
    L           magma_z_matrix
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_residuals_linkedlist(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // printf("start\n");fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L_new->nnz; e++){
        // as we look at the lower triangular, 
        // col<row, i.e. disregard last element in row
        {
            magma_int_t i,j,icol,jcol,jold;
            
            magma_index_t row = L_new->rowidx[ e ];
            magma_index_t col = L_new->col[ e ];
            
            // printf("(%d,%d) ", row, col);fflush(stdout);
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A.row[row]; i<A.row[row+1]; i++){
                if( A.col[i] == col ){
                    A_e = A.val[i];
                }
            }
            
            //now do the actual iteration
            i = L.row[ row ]; 
            j = U.row[ col ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                jold = j;
                icol = L.col[i];
                jcol = U.col[j];
                if( icol == jcol ){
                    lsum = L.val[i] * U.val[j];
                    sum = sum + lsum;
                    i = L.list[i];
                    j = U.list[j];
                }
                else if( icol<jcol ){
                    i = L.list[i];
                }
                else {
                    j = U.list[j];
                }
            }while( i!=0 && j!=0 );
            sum = sum - lsum;
            
            // write back to location e
            if( row>col ){
                L_new->val[ e ] =  ( A_e - sum ) / U.val[jold];
            } else {
                L_new->val[ e ] =  ( A_e - sum );
            }
        } 
        
    }// end omp parallel section
    
    return info;
}


/**
    Purpose
    -------
    This function creates a col-pointer and a linked list along the columns 
    for a row-major CSR matrix

    Arguments
    ---------
    
    @param[in]
    A           magma_z_matrix
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.

    @param[in,out]
    AC          magma_z_matrix*
                The matrix A but with row-pointer being for col-major, same with 
                linked list. The values, col and row indices are unchanged.
                The respective pointers point to the entities of A.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_colmajor(
    magma_z_matrix A,
    magma_z_matrix *AC,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t i,j;
    magma_int_t num_threads=1;
    magma_index_t *checkrow;
    magma_int_t el_per_block;
    AC->val=A.val;
    AC->col=A.rowidx;
    AC->rowidx=A.col;
    AC->row=NULL;
    AC->list=NULL;
    AC->memory_location = Magma_CPU;
    AC->storage_type = Magma_CSRLIST;
    AC->num_rows = A.num_rows;
    AC->num_cols = A.num_cols;
    AC->nnz = A.nnz;
    AC->true_nnz = A.true_nnz;
    
    CHECK( magma_index_malloc_cpu( &checkrow, A.true_nnz ));
    
    CHECK( magma_index_malloc_cpu( &AC->row, A.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &AC->list, A.true_nnz ));
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
   // num_threads = 1;
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        checkrow[i] = -1;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.true_nnz; i++ ){
        AC->list[i] = -1;
    }
     el_per_block = magma_ceildiv( A.num_rows, num_threads );
    // printf("blocksize:%d\n", el_per_block);
    
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        for(magma_int_t i=0; i<A.true_nnz; i++ ){
            //magma_index_t col = A.rowidx[ i ];
            magma_index_t row = A.col[ i ];
            if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block)  ){
                // printf("thread %d handling row %d\n", id, row );
                //if( A.rowidx[ i ] < A.col[ i ]){
                //    printf("illegal element:(%d,%d)\n", A.rowidx[ i ], A.col[ i ] );
                //}
                if( checkrow[row] == -1 ){
                    // printf("thread %d write in row pointer at row[%d] = %d\n", id, row, i);
                    AC->row[ row ] = i; 
                    AC->list[ i ] = 0;    
                    checkrow[ row ] = i;
                } else {
                    // printf("thread %d list[%d] = %d\n", id, checkrow[ row ], i);
                    AC->list[ checkrow[ row ] ] = i;  
                    AC->list[ i ] = 0; 
                    checkrow[ row ] = i;    
                }
            }
        }
    }
    /*
        
    for( i=560; i<570; i++ ){
        j = AC->row[i];
        while( j != 0 ){
            printf("(%d,%d) -> ", AC->rowidx[j], AC->col[j]);
            j=AC->list[j];
        }
        printf("\n");
    }*/
    

cleanup:
    magma_free_cpu( checkrow );
    return info;
}

/**
    Purpose
    -------
    This routine reorders the matrix (inplace) for easier access.

    Arguments
    ---------
                
    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.
                
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_reorder(
    magma_z_matrix *LU,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magmaDoubleComplex *val=NULL;;
    magma_index_t *col=NULL;
    magma_index_t *row=NULL;
    magma_index_t *rowidx=NULL;
    magma_index_t *list=NULL;
    
    magmaDoubleComplex *valt=NULL;;
    magma_index_t *colt=NULL;                                       
    magma_index_t *rowt=NULL;
    magma_index_t *rowidxt=NULL;
    magma_index_t *listt=NULL;
    
    magma_int_t nnz=0;
    
    
    CHECK( magma_zmalloc_cpu( &val, LU->true_nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, LU->true_nnz ));
    CHECK( magma_index_malloc_cpu( &col, LU->true_nnz ));
    CHECK( magma_index_malloc_cpu( &row, LU->num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &list, LU->true_nnz ));
    /*
    #pragma omp parallel for
    for( magma_int_t z=0; z < LU->true_nnz; z++ ){
        list[ z ] = -1;
        col[z] = -5;
        rowidx[z] = -7;
    }*/
    
    // do two sweeps to allow for parallel insertion
    row[ 0 ] = 0;
    #pragma omp parallel for
    for(magma_int_t rowc=0; rowc<LU->num_rows; rowc++){
        magma_index_t loc_nnz = 0;
        magma_int_t el = LU->row[ rowc ];
        do{
            loc_nnz++;
            el = LU->list[ el ];
        }while( el != 0 );
        row[ rowc+1 ] = loc_nnz;    
    }
    
    // global count
    for( magma_int_t i = 0; i<LU->num_rows; i++ ){
        nnz = nnz + row[ i+1 ];
        row[ i+1 ] = nnz;
    }
    
    LU->nnz = nnz;
    // parallel insertion
    #pragma omp parallel for
    for(magma_int_t rowc=0; rowc<LU->num_rows; rowc++){
        magma_int_t el = LU->row[ rowc ];
        magma_int_t offset = row[ rowc ];
        magma_index_t loc_nnz = 0;
        do{
            magmaDoubleComplex valt = LU->val[ el ];
            magma_int_t loc = offset+loc_nnz;
#ifdef NANCHECK
            if(magma_z_isnan_inf( valt ) ){
                info = MAGMA_ERR_NAN;
                el = 0;
            } else 
#endif
            {
                val[ loc ] = valt;
                col[ loc ] = LU->col[ el ];
                rowidx[ loc ] = rowc;
                list[ loc ] = loc+1;
                loc_nnz++;
                el = LU->list[ el ]; 
            }
            
        }while( el != 0 );
        list[ offset+loc_nnz - 1 ] = 0;
    }
    
    listt = LU->list;
    rowt = LU->row;
    rowidxt = LU->rowidx;
    valt = LU->val;
    colt = LU->col;
    
    LU->list = list;
    LU->row = row;
    LU->rowidx = rowidx;
    LU->val = val;
    LU->col = col;
    
    list = listt;
    row = rowt;
    rowidx = rowidxt;
    val = valt;
    col = colt;
    /*
        
    for(int i=0; i<25; i++ ){
        int j = LU->row[i];
        while( j != 0 ){
            printf("%d(%d,%d) ->%d  ",j, LU->col[j], LU->rowidx[j],LU->list[j]);
            j=LU->list[j];
        }
        printf("\n");
    }
    */

cleanup:
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( row );
    magma_free_cpu( rowidx );
    magma_free_cpu( list );
    return info;
}


/**
    Purpose
    -------
    This function creates a col-pointer and a linked list along the columns 
    for a row-major CSR matrix

    Arguments
    ---------
    
    @param[in]
    A           magma_z_matrix
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.

    @param[in,out]
    AC          magma_z_matrix*
                The matrix A but with row-pointer being for col-major, same with 
                linked list. The values, col and row indices are unchanged.
                The respective pointers point to the entities of A. Already allocated.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_colmajorup(
    magma_z_matrix A,
    magma_z_matrix *AC,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t i,j;
    magma_int_t num_threads=1;
    magma_index_t *checkrow;
    magma_int_t el_per_block;
    AC->nnz=A.nnz;
    AC->true_nnz=A.true_nnz;
    AC->val=A.val;
    AC->col=A.rowidx;
    AC->rowidx=A.col;
    
    CHECK( magma_index_malloc_cpu( &checkrow, A.num_rows ));
    /*
    for( i=0; i<10; i++ ){
        j = A.row[i];
        do{
            printf("(%d,%d) -> ", A.rowidx[j],A.col[j]);
            j=A.list[j];
        }while( j != 0 );
        printf("\n");
    }
    */
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        checkrow[i] = -1;
    }
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<AC->true_nnz; i++ ){
        AC->list[ i ] = 0;    
    }
     el_per_block = magma_ceildiv( A.num_rows, num_threads );
    
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        for(magma_int_t i=0; i<A.nnz; i++ ){
            if( A.list[ i ]!= -1 ){
                magma_index_t col = A.rowidx[ i ];
                magma_index_t row = A.col[ i ];
                if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block)  ){
                    if( checkrow[row] == -1 ){
                        AC->row[ row ] = i; 
                        AC->list[ i ] = 0;    
                        checkrow[ row ] = i;
                    } else {
                        AC->list[ checkrow[ row ] ] = i;  
                        AC->list[ i ] = 0; 
                        checkrow[ row ] = i;    
                    }
                }
            }
        }
    }
    /*
            
    for( i=736; i<740; i++ ){
        j = AC->row[i];
        do{
            printf("(%d,%d) -> ", AC->col[j], AC->rowidx[j]);
            j=AC->list[j];
        }while( j != 0 );
        printf("\n");
    }*/

cleanup:
    magma_free_cpu( checkrow );
    return info;
}




/**
    Purpose
    -------
    This function creates a col-pointer and a linked list along the columns 
    for a row-major CSR matrix

    Arguments
    ---------
    
    @param[in]
    A           magma_z_matrix
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.

    @param[in,out]
    AC          magma_z_matrix*
                The matrix A but with row-pointer being for col-major, same with 
                linked list. The values, col and row indices are unchanged.
                The respective pointers point to the entities of A.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_colmajor_old(
    magma_z_matrix A,
    magma_z_matrix *AC,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t i,j;

    AC->val=A.val;
    AC->col=A.rowidx;
    AC->rowidx=A.col;
    AC->row=NULL;
    AC->list=NULL;
    AC->memory_location = Magma_CPU;
    AC->storage_type = Magma_CSRLIST;
    AC->num_rows = A.num_rows;
    AC->num_cols = A.num_cols;
    AC->nnz = A.nnz;
    
    CHECK( magma_index_malloc_cpu( &AC->row, A.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &AC->list, A.true_nnz ));
    #pragma omp parallel for
    for( i=0; i<A.num_rows; i++ ){
        magma_int_t last, checkrow, checkel, count=0;
        for( checkrow=0; checkrow<A.num_rows; checkrow++ ){
            checkel = A.row[ checkrow ];  
            do{
                if( A.col[checkel]==i ){
                    if( count == 0 ){
                        AC->row[ i ] = checkel;
                        AC->list[ checkel ] = 0;
                        last = checkel;
                        count++;
                    } else {
                        AC->list[ last ] = checkel;
                        AC->list[ checkel ] = 0;
                        last = checkel;
                        count++;
                    }
                }
                checkel = A.list[ checkel ];
            }while( checkel!=0 );
            
        }
    }
    /*
    for( i=15; i<25; i++ ){
        j = AC->row[i];
        while( j != 0 ){
            printf("(%d,%d) -> ", AC->col[j], AC->rowidx[j]);
            j=AC->list[j];
        }
        printf("\n");
    }
    */
cleanup:
    return info;
}






/**
    Purpose
    -------
    Inserts for the iterative dynamic ILU an new element in the (empty) place. 

    Arguments
    ---------
                
    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced in L.
                
    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced in U.
                
    @param[in]
    rm_locL     magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    rm_locU     magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    L_new       magma_z_matrix
                Elements that will be inserted in L stored in COO format (unsorted).
                
    @param[in]
    U_new       magma_z_matrix
                Elements that will be inserted in U stored in COO format (unsorted).

    @param[in,out]
    L           magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in,out]
    U           magma_z_matrix*
                matrix where new elements are inserted. Row-major.
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in,out]
    UR          magma_z_matrix*
                Same matrix as U, but column-major.
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_insert(
    magma_int_t *num_rmL,
    magma_int_t *num_rmU,
    magma_index_t *rm_locL,
    magma_index_t *rm_locU,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_z_matrix *UR,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t i = 0;
    magma_int_t lasti = 0;
    magma_int_t num_threads=-1;
    magma_int_t el_per_block;

    //printf("%%candidates for L:%d\n", *num_rmL);
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    
    // first part L
    #pragma omp parallel
    {
    magma_int_t id = omp_get_thread_num();
    magma_int_t el = L_new->row[id];
    magma_int_t lr = 0;
    magma_int_t loc_lr = rm_locL[id];
    while( el>-1 ){
        magma_int_t loc = L->nnz + loc_lr;
        loc_lr++;
        magma_index_t new_row = L_new->rowidx[ el ]; 
        magma_index_t new_col = L_new->col[ el ];
        magma_index_t old_rowstart = L->row[ new_row ];
        //printf("%%candidate for L: (%d,%d) tid %d\n", new_row, new_col, id);
        //printf("%%candidate for L: tid %d %d < %d (%d,%d) going to %d+%d = %d\n", id, add_loc[id], rm_locL[id+1], new_row, new_col, L->nnz, add_loc[id]-1, loc);fflush(stdout);
        if( new_col < L->col[ old_rowstart ] ){
            //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
            L->row[ new_row ] = loc;
            L->list[ loc ] = old_rowstart;
            L->rowidx[ loc ] = new_row;
            L->col[ loc ] = new_col;
            L->val[ loc ] = MAGMA_Z_ZERO;
        }
        else if( new_col == L->col[ old_rowstart ] ){
            ;//printf("%% tried to insert duplicate in L! case 1 tid %d location %d (%d,%d) = (%d,%d)\n", id, r, new_row,new_col,L->rowidx[ old_rowstart ], L->col[ old_rowstart ]);fflush(stdout);
        }
        else{
            magma_int_t j = old_rowstart;
            magma_int_t jn = L->list[j];
            // this will finish, as we consider the lower triangular
            // and we always have the diagonal!
            while( j!=0 ){
                if( L->col[jn]==new_col ){
                    //printf("%% tried to insert duplicate case 1 2 in L thread %d: (%d %d) \n", id, new_row, new_col);fflush(stdout);
                    j=0; //break;
                }else if( L->col[jn]>new_col ){

                    //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
                    L->list[j]=loc;
                    L->list[loc]=jn;
                    L->rowidx[ loc ] = new_row;
                    L->col[ loc ] = new_col;
                    L->val[ loc ] = MAGMA_Z_ZERO;
                    j=0; //break;
                    
                } else{
                    j=jn;
                    jn=L->list[jn];
                }
            }
        }
        el = L_new->list[ el ];
    }
    }
    /*
    magma_int_t lr = 0;
    magma_int_t loc_lr = rm_locL[id];
    for( magma_int_t r=0; r<L_new->nnz; r++ ){ 
    //for( magma_int_t r=L_new->row[id*el_per_block]; r<L_new->row[(id+1)*el_per_block]; r++ ){ 
      //  if( loc_lr < rm_locL[id+1] ){
      {
            if( L_new->list[r] == id ){
                magma_int_t loc = L->nnz + loc_lr;
                loc_lr++;
                magma_index_t new_row = L_new->rowidx[ r ]; 
                magma_index_t new_col = L_new->col[ r ];
                magma_index_t old_rowstart = L->row[ new_row ];
                //printf("%%candidate for L: (%d,%d) tid %d\n", new_row, new_col, id);
                //printf("%%candidate for L: tid %d %d < %d (%d,%d) going to %d+%d = %d\n", id, add_loc[id], rm_locL[id+1], new_row, new_col, L->nnz, add_loc[id]-1, loc);fflush(stdout);
                if( new_col < L->col[ old_rowstart ] ){
                    //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
                    L->row[ new_row ] = loc;
                    L->list[ loc ] = old_rowstart;
                    L->rowidx[ loc ] = new_row;
                    L->col[ loc ] = new_col;
                    L->val[ loc ] = MAGMA_Z_ZERO;
                }
                else if( new_col == L->col[ old_rowstart ] ){
                    printf("%% tried to insert duplicate in L! case 1 tid %d location %d (%d,%d) = (%d,%d)\n", id, r, new_row,new_col,L->rowidx[ old_rowstart ], L->col[ old_rowstart ]);fflush(stdout);
                }
                else{
                    magma_int_t j = old_rowstart;
                    magma_int_t jn = L->list[j];
                    // this will finish, as we consider the lower triangular
                    // and we always have the diagonal!
                    while( j!=0 ){
                        if( L->col[jn]==new_col ){
                            printf("%% tried to insert duplicate case 1 2 in L thread %d: (%d %d) \n", id, new_row, new_col);fflush(stdout);
                            j=0; //break;
                        }else if( L->col[jn]>new_col ){
    
                            //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
                            L->list[j]=loc;
                            L->list[loc]=jn;
                            L->rowidx[ loc ] = new_row;
                            L->col[ loc ] = new_col;
                            L->val[ loc ] = MAGMA_Z_ZERO;
                            j=0; //break;
                            
                        } else{
                            j=jn;
                            jn=L->list[jn];
                        }
                    }
                }
            }
        }
    }
    }
  //  printf(" done insert L. now U: %d\n", U_new->nnz);fflush(stdout);
    */
    // second part U
    // U is stored as col-major, so U->row is a column-pointer, col = rowidx and rowidx = col
    // UR has the rowpointer
    // inserting requires to change all these entities
    
    // second part U
    #pragma omp parallel
    {
    magma_int_t id = omp_get_thread_num();
    magma_int_t el = U_new->row[id];
    magma_int_t ur = 0;
    magma_int_t loc_ur = rm_locU[id];
    while( el>-1 ){
        magma_int_t loc = U->nnz + loc_ur;
        loc_ur++;
        magma_index_t new_col = U_new->rowidx[ el ];    // we flip these entities to have the same
        magma_index_t new_row = U_new->col[ el ];    // situation like in the lower triangular case
        //printf("%%candidate for U: (%d,%d) tid %d\n", new_row, new_col, id);
        if( new_row < new_col ){
        printf("%% illegal candidate %d for U: (%d,%d)'\n", el, new_row, new_col);
        //exit(-1);
        }
        //printf("%% candidate %d for U: (%d,%d)'\n", el, new_row, new_col);
        magma_index_t old_rowstart = U->row[ new_row ];
        //printf("%%candidate for U: tid %d %d < %d (%d,%d) going to %d+%d+%d+%d = %d\n", id, add_loc[id+1]-1, rm_locU[id+1], new_row, new_col, U->nnz, rm_locU[id], id, add_loc[id+1]-1, loc);fflush(stdout);
        if( new_col < U->col[ old_rowstart ] ){
            //  printf("%% insert in U as first element: (%d,%d)'\n", new_row, new_col);
            U->row[ new_row ] = loc;
            U->list[ loc ] = old_rowstart;
            U->rowidx[ loc ] = new_row;
            U->col[ loc ] = new_col;
            U->val[ loc ] = MAGMA_Z_ZERO;
        }
        else if( new_col == U->col[ old_rowstart ] ){
            ;//printf("%% tried to insert duplicate in U! case 1 single element (%d,%d) at %d \n", new_row, new_col, r);
        }
        else{
            magma_int_t j = old_rowstart;
            magma_int_t jn = U->list[j];
            while( j!=0 ){
                if( U->col[j]==new_col ){
                    //printf("%% tried to insert duplicate case 1 2 in U thread %d: (%d %d) \n", id, new_row, new_col);
                    j=0; //break;
                }else if( U->col[jn]>new_col ){

                    U->list[j]=loc;
                    U->list[loc]=jn;
                    U->rowidx[ loc ] = new_row;
                    U->col[ loc ] = new_col;
                    U->val[ loc ] = MAGMA_Z_ZERO;
                    j=0; //break;
                } else{
                    j=jn;
                    jn=U->list[jn];
                }
            }
        }
        el = U_new->list[el];
    }
    }                
    
cleanup:
     return info;
}




/**
    Purpose
    -------
    Inserts for the iterative dynamic ILU an new element in the (empty) place. 

    Arguments
    ---------
                
    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced in L.
                
    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced in U.
                
    @param[in]
    rm_locL     magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    rm_locU     magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    L_new       magma_z_matrix
                Elements that will be inserted in L stored in COO format (unsorted).
                
    @param[in]
    U_new       magma_z_matrix
                Elements that will be inserted in U stored in COO format (unsorted).

    @param[in,out]
    L           magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in,out]
    U           magma_z_matrix*
                matrix where new elements are inserted. Row-major.
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in,out]
    UR          magma_z_matrix*
                Same matrix as U, but column-major.
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_insert_new(
    magma_int_t *num_rmL,
    magma_int_t *num_rmU,
    magma_index_t *rm_locL,
    magma_index_t *rm_locU,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_z_matrix *UR,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=-1;
    magma_int_t el_per_block;

    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    
#ifdef AVOID_DUPLICATES
    // first part L
    #pragma omp parallel
    {
    magma_int_t id = omp_get_thread_num();
    magma_int_t lr = 0;
    magma_int_t loc_lr = rm_locL[id];
    magma_int_t lbound = L_new->row[(id+1)*el_per_block];
    if( id == num_threads-1 ){
        lbound = L_new->nnz;    
    }
    for( magma_int_t r=L_new->row[(id)*el_per_block]; r<lbound; r++ ){ 
        if( loc_lr < rm_locL[id+1] ){
        // case only element in row
        //{
        if( L_new->list[r] == id ){
        //{
            //L_new->list[r] = -1;
            //magma_int_t loc = L->nnz + r;
            magma_int_t loc;
            
            loc = L->nnz + loc_lr;
            loc_lr++;
            //add_loc[id]++;
            
            // if(L->col[ loc ] != 0){
            // printf("something wrong: tid:%d  Lnnz:%d %d %d %d = %d (%d,%d)!\n", id, L->nnz, rm_locL[id], id, add_loc[id+1], loc, L->rowidx[loc],L->col[ loc ]); exit(-1);   
            // }
            
            magma_index_t new_row = L_new->rowidx[ r ]; 
            magma_index_t new_col = L_new->col[ r ];
            magma_index_t old_rowstart = L->row[ new_row ];
            //printf("%%candidate for L: tid %d %d < %d (%d,%d) going to %d+%d = %d\n", id, add_loc[id], rm_locL[id+1], new_row, new_col, L->nnz, add_loc[id]-1, loc);fflush(stdout);
            if( new_col < L->col[ old_rowstart ] ){
                //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
                L->row[ new_row ] = loc;
                L->list[ loc ] = old_rowstart;
                L->rowidx[ loc ] = new_row;
                L->col[ loc ] = new_col;
                L->val[ loc ] = MAGMA_Z_ZERO;
            }
            else if( new_col == L->col[ old_rowstart ] ){
                printf("%% tried to insert duplicate in L! case 1 tid %d location %d (%d,%d) = (%d,%d)\n", id, r, new_row,new_col,L->rowidx[ old_rowstart ], L->col[ old_rowstart ]);fflush(stdout);
            }
            else{
                magma_int_t j = old_rowstart;
                magma_int_t jn = L->list[j];
                // this will finish, as we consider the lower triangular
                // and we always have the diagonal!
                while( j!=0 ){
                    if( L->col[jn]==new_col ){
                        printf("%% tried to insert duplicate case 1 2 in L thread %d: (%d %d) \n", id, new_row, new_col);fflush(stdout);
                        j=0; //break;
                    }else if( L->col[jn]>new_col ){

                        //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
                        L->list[j]=loc;
                        L->list[loc]=jn;
                        L->rowidx[ loc ] = new_row;
                        L->col[ loc ] = new_col;
                        L->val[ loc ] = MAGMA_Z_ZERO;
                        j=0; //break;
                        
                    } else{
                        j=jn;
                        jn=L->list[jn];
                    }
                }
            }
        }
    }
    }
    }
  //  printf(" done insert L. now U: %d\n", U_new->nnz);fflush(stdout);
    
    // second part U
    // U is stored as col-major, so U->row is a column-pointer, col = rowidx and rowidx = col
    // UR has the rowpointer
    // inserting requires to change all these entities
    
    #pragma omp parallel
    {
    magma_int_t id = omp_get_thread_num();
    magma_int_t ur = 0;
    magma_int_t loc_ur = rm_locU[id];
    magma_int_t lbound = U_new->row[(id+1)*el_per_block];
    if( id == num_threads-1 ){
        lbound = U_new->nnz;    
    }
    for( magma_int_t r=U_new->row[(id)*el_per_block]; r<lbound; r++ ){ 
        if( loc_ur < rm_locU[id+1] ){
        if( U_new->list[r] == id ){
        //{
            //U_new->list[r] = -1;
            magma_int_t loc;
            loc = U->nnz + loc_ur;
            loc_ur++;
            
            
            
            magma_index_t new_col = U_new->rowidx[ r ];    // we flip these entities to have the same
            magma_index_t new_row = U_new->col[ r ];    // situation like in the lower triangular case
            
            if( new_row < new_col ){
                printf("%% illegal candidate %d for U: (%d,%d)'\n", r, new_row, new_col);
                //exit(-1);
            }
            magma_index_t old_rowstart = U->row[ new_row ];
             //printf("%%candidate for U: tid %d %d < %d (%d,%d) going to %d+%d+%d+%d = %d\n", id, add_loc[id+1]-1, rm_locU[id+1], new_row, new_col, U->nnz, rm_locU[id], id, add_loc[id+1]-1, loc);fflush(stdout);
            if( new_col < U->col[ old_rowstart ] ){
                //  printf("%% insert in U as first element: (%d,%d)'\n", new_row, new_col);
                U->row[ new_row ] = loc;
                U->list[ loc ] = old_rowstart;
                U->rowidx[ loc ] = new_row;
                U->col[ loc ] = new_col;
                U->val[ loc ] = MAGMA_Z_ZERO;
            }
            else if( new_col == U->col[ old_rowstart ] ){
                printf("%% tried to insert duplicate in U! case 1 single element (%d,%d) at %d \n", new_row, new_col, r);
            }
            else{
                magma_int_t j = old_rowstart;
                magma_int_t jn = U->list[j];
                while( j!=0 ){
                    if( U->col[j]==new_col ){
                        printf("%% tried to insert duplicate case 1 2 in U thread %d: (%d %d) \n", id, new_row, new_col);
                        j=0; //break;
                    }else if( U->col[jn]>new_col ){

                        U->list[j]=loc;
                        U->list[loc]=jn;
                        U->rowidx[ loc ] = new_row;
                        U->col[ loc ] = new_col;
                        U->val[ loc ] = MAGMA_Z_ZERO;
                        j=0; //break;
                    } else{
                        j=jn;
                        jn=U->list[jn];
                    }
                }
            }
        }
    }
    }
    }                
                
                
                
                
                
#else    
    
    // first part L
    #pragma omp parallel
    {
    magma_int_t id = omp_get_thread_num();
    magma_int_t lr = 0;
    magma_int_t loc_lr = rm_locL[id];
    for (magma_int_t r=0; r<L_new->nnz; r++){
        
        if( loc_lr < rm_locL[id+1] ){
        // case only element in row
        //{
        if( L_new->list[r] == id ){
        //{
            //L_new->list[r] = -1;
            //magma_int_t loc = L->nnz + r;
            magma_int_t loc;
            
            loc = L->nnz + loc_lr;
            loc_lr++;
            //add_loc[id]++;
            
            // if(L->col[ loc ] != 0){
            // printf("something wrong: tid:%d  Lnnz:%d %d %d %d = %d (%d,%d)!\n", id, L->nnz, rm_locL[id], id, add_loc[id+1], loc, L->rowidx[loc],L->col[ loc ]); exit(-1);   
            // }
            
            magma_index_t new_row = L_new->rowidx[ r ]; 
            magma_index_t new_col = L_new->col[ r ];
            magma_index_t old_rowstart = L->row[ new_row ];
            //printf("%%candidate for L: tid %d %d < %d (%d,%d) going to %d+%d = %d\n", id, add_loc[id], rm_locL[id+1], new_row, new_col, L->nnz, add_loc[id]-1, loc);fflush(stdout);
            if( new_col < L->col[ old_rowstart ] ){
                //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
                L->row[ new_row ] = loc;
                L->list[ loc ] = old_rowstart;
                L->rowidx[ loc ] = new_row;
                L->col[ loc ] = new_col;
                L->val[ loc ] = MAGMA_Z_ZERO;
            }
            else if( new_col == L->col[ old_rowstart ] ){
                printf("%% tried to insert duplicate in L! case 1 tid %d location %d (%d,%d) = (%d,%d)\n", id, r, new_row,new_col,L->rowidx[ old_rowstart ], L->col[ old_rowstart ]);fflush(stdout);
            }
            else{
                magma_int_t j = old_rowstart;
                magma_int_t jn = L->list[j];
                // this will finish, as we consider the lower triangular
                // and we always have the diagonal!
                while( j!=0 ){
                    if( L->col[jn]==new_col ){
                        printf("%% tried to insert duplicate case 1 2 in L thread %d: (%d %d) \n", id, new_row, new_col);fflush(stdout);
                        j=0; //break;
                    }else if( L->col[jn]>new_col ){

                        //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
                        L->list[j]=loc;
                        L->list[loc]=jn;
                        L->rowidx[ loc ] = new_row;
                        L->col[ loc ] = new_col;
                        L->val[ loc ] = MAGMA_Z_ZERO;
                        j=0; //break;
                        
                    } else{
                        j=jn;
                        jn=L->list[jn];
                    }
                }
            }
        }
    }
    }
    }
  //  printf(" done insert L. now U: %d\n", U_new->nnz);fflush(stdout);
    
    // second part U
    // U is stored as col-major, so U->row is a column-pointer, col = rowidx and rowidx = col
    // UR has the rowpointer
    // inserting requires to change all these entities
    
    #pragma omp parallel
    {
    magma_int_t id = omp_get_thread_num();
    magma_int_t ur = 0;
    magma_int_t loc_ur = rm_locU[id];
    for (magma_int_t r=0; r<U_new->nnz; r++){
        
        
        // printf("%%candidate %d for U' with %d\n", r);
        if( loc_ur < rm_locU[id+1] ){
        if( U_new->list[r] == id ){
        //{
            //U_new->list[r] = -1;
            magma_int_t loc;
            loc = U->nnz + loc_ur;
            loc_ur++;
            
            
            
            magma_index_t new_col = U_new->rowidx[ r ];    // we flip these entities to have the same
            magma_index_t new_row = U_new->col[ r ];    // situation like in the lower triangular case
            
            if( new_row < new_col ){
                printf("%% illegal candidate %d for U: (%d,%d)'\n", r, new_row, new_col);
                //exit(-1);
            }
            magma_index_t old_rowstart = U->row[ new_row ];
             //printf("%%candidate for U: tid %d %d < %d (%d,%d) going to %d+%d+%d+%d = %d\n", id, add_loc[id+1]-1, rm_locU[id+1], new_row, new_col, U->nnz, rm_locU[id], id, add_loc[id+1]-1, loc);fflush(stdout);
            if( new_col < U->col[ old_rowstart ] ){
                //  printf("%% insert in U as first element: (%d,%d)'\n", new_row, new_col);
                U->row[ new_row ] = loc;
                U->list[ loc ] = old_rowstart;
                U->rowidx[ loc ] = new_row;
                U->col[ loc ] = new_col;
                U->val[ loc ] = MAGMA_Z_ZERO;
            }
            else if( new_col == U->col[ old_rowstart ] ){
                printf("%% tried to insert duplicate in U! case 1 single element (%d,%d) \n", new_row, new_col);
            }
            else{
                magma_int_t j = old_rowstart;
                magma_int_t jn = U->list[j];
                while( j!=0 ){
                    if( U->col[j]==new_col ){
                        printf("%% tried to insert duplicate case 1 2 in U thread %d: (%d %d) \n", id, new_row, new_col);
                        j=0; //break;
                    }else if( U->col[jn]>new_col ){

                        U->list[j]=loc;
                        U->list[loc]=jn;
                        U->rowidx[ loc ] = new_row;
                        U->col[ loc ] = new_col;
                        U->val[ loc ] = MAGMA_Z_ZERO;
                        j=0; //break;
                    } else{
                        j=jn;
                        jn=U->list[jn];
                    }
                }
            }
        }
    }
    }
    }
#endif
    
cleanup:
     return info;
}








/**
    Purpose
    -------
    This function identifies the candidates like they appear as ILU1 fill-in.

    Arguments
    ---------
    
    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.
                
    @param[in]
    U           magma_z_matrix
                Current upper triangular factor transposed.
                
    @param[in]
    UR          magma_z_matrix
                Current upper triangular factor - col-pointer and col-list.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for L in COO format.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for U in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_candidates(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix UR,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    real_Double_t start, end;
    magma_index_t *numaddL;
    magma_index_t *numaddU;
    magma_index_t *rowidxL, *rowidxU, *colL, *colU;
    magma_index_t *rowidxLt, *rowidxUt, *colLt, *colUt;
    
    magma_int_t unsym = 1;
    
    magma_int_t avoid_duplicates = 1;
    
    CHECK( magma_index_malloc_cpu( &numaddL, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &numaddU, U.num_rows+1 ));
    
    CHECK( magma_index_malloc_cpu( &rowidxL, L_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &rowidxU, U_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &colL, L_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &colU, U_new->true_nnz ));
    
    
    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
        numaddL[i] = 0;
        numaddU[i] = 0;
    }
    

     
    // how to determine candidates:
    // for each node i, look at any "intermediate" neighbor nodes numbered
    // less, and then see if this neighbor has another neighbor j numbered
    // more than the intermediate; if so, fill in is (i,j) if it is not
    // already nonzero
start = magma_sync_wtime( queue );
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        //while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            
            
            // second loop first element over row - only for elements larger the intermediate
            //while( L.list[ el2 ] != 0 ) {
            for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_int_t s=L.row[cand_row];
                magma_int_t e=L.row[cand_row+1];
                for( magma_index_t checkel=s; checkel<e; checkel++){
                    checkcol = L.col[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=e;
                    }
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numaddrowL++;
                    //numaddL[ row+1 ]++;
                }
                //el2 = L.list[ el2 ];
            }
        }
        numaddU[ row+1 ] = numaddrowU;
        numaddL[ row+1 ] = numaddrowL;
    }
            
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        //while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if( cand_col < row ){
                    // check whether this element already exists
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_int_t s=L.row[cand_row];
                    magma_int_t e=L.row[cand_row+1];
                    for( magma_index_t checkel=s; checkel<e; checkel++){
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=e;
                        }
                    }
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                        numaddrowL++;
                        //numaddL[ row+1 ]++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    for( magma_index_t checkel=U.row[cand_col]; checkel<U.row[cand_col+1]; checkel++){
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=U.row[cand_col+1];
                        }
                    }
                    if( exist == 0 ){
                        numaddrowU++;
                        //numaddU[ row+1 ]++;
                    }
                }
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            //el1 = L.list[ el1 ];
        }
        numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
    }
    
    //#######################
    if( unsym == 1 ){
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){

        magma_index_t start1 = UR.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
                          // printf("row:%d el:%d\n", row, el1);
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            //while( L.list[ el2 ] != 0 ) {
            for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                //printf("check cand (%d,%d)\n",cand_row, cand_col);
                // check whether this element already exists
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_int_t s=L.row[cand_row];
                    magma_int_t e=L.row[cand_row+1];
                    for( magma_index_t checkel=s; checkel<e; checkel++){
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=e;
                        }
                    }
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        numaddrowL++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    for( magma_index_t checkel=U.row[cand_col]; checkel<U.row[cand_col+1]; checkel++){
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=U.row[cand_col+1];
                        }
                    }
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        numaddrowU++;
                    }
                }
                //el2 = L.list[ el2 ];
            }
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
        
        numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
    }
    
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            if( col1 < row ){
                printf("row:%d col1:%d\n", row, col1);
                exit(-1);
            }
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
            // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                for( magma_index_t checkel=U.row[cand_col]; checkel<U.row[cand_col+1]; checkel++){
                    checkcol = U.col[ checkel ];
                    if( checkcol == cand_row ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=U.row[cand_col+1];
                    }
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numaddrowU++;
                }
                
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
         
        numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
    } //loop over all rows
    }
    
    //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
    
    // #########################################################################
    
        
    // get the total candidate count
    L_new->nnz = 0;
    U_new->nnz = 0;
    numaddL[ 0 ] = L_new->nnz;
    numaddU[ 0 ] = U_new->nnz;
    L_new->row[ 0 ] = 0;
    U_new->row[ 0 ] = 0;
    for( magma_int_t i = 0; i<L.num_rows; i++ ){
        L_new->nnz=L_new->nnz + numaddL[ i+1 ];
        U_new->nnz=U_new->nnz + numaddU[ i+1 ];
        numaddL[ i+1 ] = L_new->nnz;
        numaddU[ i+1 ] = U_new->nnz;
        L_new->row[ i+1 ] = 0;
        U_new->row[ i+1 ] = 0;

    }

    if( L_new->nnz > L_new->true_nnz ){
       magma_free_cpu( L_new->val );
       magma_free_cpu( L_new->row );
       magma_free_cpu( L_new->rowidx );
       magma_free_cpu( L_new->col );
       magma_free_cpu( L_new->list );
       magma_zmalloc_cpu( &L_new->val, L_new->nnz*2 );
       magma_index_malloc_cpu( &L_new->rowidx, L_new->nnz*2 );
       magma_index_malloc_cpu( &L_new->col, L_new->nnz*2 );
       magma_index_malloc_cpu( &L_new->row, L_new->nnz*2 );
       magma_index_malloc_cpu( &L_new->list, L_new->nnz*2 ); 
       L_new->true_nnz = L_new->nnz*2;
    }
    
    if( U_new->nnz > U_new->true_nnz ){
       magma_free_cpu( U_new->val );
       magma_free_cpu( U_new->row );
       magma_free_cpu( U_new->rowidx );
       magma_free_cpu( U_new->col );
       magma_free_cpu( U_new->list );
       magma_zmalloc_cpu( &U_new->val, U_new->nnz*2 );
       magma_index_malloc_cpu( &U_new->rowidx, U_new->nnz*2 );
       magma_index_malloc_cpu( &U_new->col, U_new->nnz*2 );
       magma_index_malloc_cpu( &U_new->row, U_new->nnz*2 );
       magma_index_malloc_cpu( &U_new->list, U_new->nnz*2 ); 
       U_new->true_nnz = U_new->nnz*2;
    }
    printf("candidates :%d %d\n", L_new->nnz, U_new->nnz);
    // #########################################################################
    
    start = magma_sync_wtime( queue );
    // insertion here
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        //while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            //while( L.list[ el2 ] != 0 ) {
            for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_int_t s=L.row[cand_row];
                magma_int_t e=L.row[cand_row+1];
                for( magma_index_t checkel=s; checkel<e; checkel++){
                    checkcol = L.col[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=e;
                    }
                }
#ifdef AVOID_DUPLICATES
                    for( magma_index_t checkel=numaddL[cand_row]; checkel<offsetL+laddL; checkel++){
                        checkcol = L_new->col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=offsetL+laddL;
                        }
                    }
#endif
                
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                    //add in the next location for this row
                    // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                    L_new->rowidx[ offsetL + laddL ] = cand_row;
                    L_new->col[ offsetL + laddL ] = cand_col;
                    // L_new->list[ numaddL[row] + laddL ] =  -1;
                    // L_new->row[ numaddL[row] + laddL ] =  -1;
                    laddL++; 
                }
                //el2 = L.list[ el2 ];
            }
            
        }
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    }
            
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        //while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_int_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_int_t s=L.row[cand_row];
                    magma_int_t e=L.row[cand_row+1];
                    for( magma_index_t checkel=s; checkel<e; checkel++){
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=e;
                        }
                    }
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=numaddL[cand_row]; checkel<offsetL+laddL; checkel++){
                            checkcol = L_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetL+laddL;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                        //add in the next location for this row
                        // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        L_new->rowidx[ offsetL + laddL ] = cand_row;
                        L_new->col[ offsetL + laddL ] = cand_col;
                        // L_new->list[ numaddL[row] + laddL ] = -1;
                        // L_new->row[ numaddL[row] + laddL ] = -1;
                        laddL++; 
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    for( magma_index_t checkel=U.row[cand_col]; checkel<U.row[cand_col+1]; checkel++){
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=U.row[cand_col+1];
                        }
                    }
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=numaddU[cand_row]; checkel<offsetU+laddU; checkel++){
                            checkcol = U_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetU+laddU;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //  printf("---------------->>>  candidate in U at (%d, %d) stored in %d\n", cand_row, cand_col, numaddU[row] + laddU);
                        //add in the next location for this row
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ offsetU + laddU ] = cand_col;
                        U_new->rowidx[ offsetU + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++; 
                        if( cand_row > cand_col )
                            printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
                    }
                }
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            //el1 = L.list[ el1 ];
        }
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    }
        
    
    //#######################
    if( unsym == 1 ){
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            if( col1 < row ){
                printf("row:%d col1:%d\n", row, col1);
                exit(-1);
            }
                
              
          // printf("row:%d el:%d\n", row, el1);
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            //while( L.list[ el2 ] != 0 ) {
            for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                // check whether this element already exists
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_int_t s=L.row[cand_row];
                    magma_int_t e=L.row[cand_row+1];
                    for( magma_index_t checkel=s; checkel<e; checkel++){
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=e;
                        }
                    }
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=numaddL[cand_row]; checkel<offsetL+laddL; checkel++){
                            checkcol = L_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetL+laddL;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        L_new->rowidx[ offsetL + laddL ] = cand_row;
                        L_new->col[ offsetL + laddL ] = cand_col;
                        // L_new->list[ numaddL[row] + laddL ] =  -1;
                        // L_new->row[ numaddL[row] + laddL ] =  -1;
                        laddL++; 
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    for( magma_index_t checkel=U.row[cand_col]; checkel<U.row[cand_col+1]; checkel++){
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=U.row[cand_col+1];
                        }
                    }
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=numaddU[cand_row]; checkel<offsetU+laddU; checkel++){
                            checkcol = U_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetU+laddU;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ offsetU + laddU ] = cand_col;
                        U_new->rowidx[ offsetU + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++; 
                    }
                }
                //el2 = L.list[ el2 ];
            }
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
        
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    }
    
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        magma_index_t el1 = start1;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            if( col1 < row ){
                printf("row:%d col1:%d\n", row, col1);
                exit(-1);
            }
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                for( magma_index_t checkel=U.row[cand_col]; checkel<U.row[cand_col+1]; checkel++){
                    checkcol = U.col[ checkel ];
                    if( checkcol == cand_row ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=U.row[cand_col+1];
                    }
                }
#ifdef AVOID_DUPLICATES
                    for( magma_index_t checkel=numaddU[cand_row]; checkel<offsetU+laddU; checkel++){
                        checkcol = U_new->col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=offsetU+laddU;
                        }
                    }
#endif
               // if it does not exist, increase counter for this location
               // use the entry one further down to allow for parallel insertion
               if( exist == 0 ){
                   // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                   U_new->col[ offsetU + laddU ] = cand_col;
                   U_new->rowidx[ offsetU + laddU ] = cand_row;
                   // U_new->list[ numaddU[row] + laddU ] = -1;
                   // U_new->row[ numaddU[row] + laddU ] = -1;
                   laddU++; 
                   //if( cand_row == 118 && cand_col == 163 )
                   //         printf("checked row insetion %d this element does not yet exist in U: (%d,%d) row starts with (%d,%d)\n", cand_row, cand_row, cand_col, U.col[ checkel ], cand_col );
                   if( cand_row > cand_col )
                       printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
               }
               
               el2 = UR.list[ el2 ];
           }while( el2 != 0 );
            
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
        
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
        
    } //loop over all rows
    }
    
#ifdef AVOID_DUPLICATES
        // #####################################################################
        
        // get the total candidate count
        L_new->nnz = 0;
        U_new->nnz = 0;
        
        for( magma_int_t i = 0; i<L.num_rows; i++ ){
            L_new->nnz=L_new->nnz + L_new->row[ i+1 ];
            U_new->nnz=U_new->nnz + U_new->row[ i+1 ];
            L_new->row[ i+1 ] = L_new->nnz;
            U_new->row[ i+1 ] = U_new->nnz;
        }
        #pragma omp parallel for
        for( magma_int_t i = 0; i<L.num_rows; i++ ){
            for( magma_int_t j=L_new->row[ i ]; j<L_new->row[ i+1 ]; j++ ){
                magma_int_t k = j-L_new->row[ i ]+numaddL[ i ];
                //printf("j:%d, numaddL[ i+1 ]=%d\n", j, numaddL[ i+1 ]);
                colL[ j ]    =  L_new->col[ k ];
                rowidxL[ j ] =  L_new->rowidx[ k ];
            }
            for( magma_int_t j=U_new->row[ i ]; j<U_new->row[ i+1 ]; j++ ){
                magma_int_t k = j-U_new->row[ i ]+numaddU[ i ];
                colU[ j ]    =  U_new->col[ k ];
                rowidxU[ j ] =  U_new->rowidx[ k ];
            }
        }
        

        
        rowidxLt = L_new->rowidx;
        colLt = L_new->col;
        rowidxUt = U_new->rowidx;
        colUt = U_new->col;
        
        L_new->rowidx = rowidxL;
        L_new->col = colL;
        U_new->rowidx = rowidxU;
        U_new->col = colU;
        
        rowidxL = rowidxLt;
        colL = colLt;
        rowidxU = rowidxUt;
        colU = colUt;
        //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
        // #####################################################################
#endif
   
cleanup:
    magma_free_cpu( numaddL );
    magma_free_cpu( numaddU );
    magma_free_cpu( rowidxL );
    magma_free_cpu( rowidxU );
    magma_free_cpu( colL );
    magma_free_cpu( colU );
    return info;
}







/**
    Purpose
    -------
    This function identifies the candidates like they appear as ILU1 fill-in.
    In this version, the matrices are assumed unordered, 
    the linked list is traversed to acces the entries of a row.

    Arguments
    ---------
    
    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.
                
    @param[in]
    U           magma_z_matrix
                Current upper triangular factor transposed.
                
    @param[in]
    UR          magma_z_matrix
                Current upper triangular factor - col-pointer and col-list.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for L in COO format.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for U in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_candidates_linkedlist(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix UR,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=-1;
    
    real_Double_t start, end;
    magma_index_t *numaddL;
    magma_index_t *numaddU;
    magma_index_t *rowidxL, *rowidxU, *colL, *colU;
    magma_index_t *rowidxLt, *rowidxUt, *colLt, *colUt;
    
    magma_int_t unsym = 1;
    
    magma_int_t avoid_duplicates = 1;
    
    CHECK( magma_index_malloc_cpu( &numaddL, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &numaddU, U.num_rows+1 ));
    
    CHECK( magma_index_malloc_cpu( &rowidxL, L_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &rowidxU, U_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &colL, L_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &colU, U_new->true_nnz ));
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    
    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
        numaddL[i] = 0;
        numaddU[i] = 0;
    }
    

     
    // how to determine candidates:
    // for each node i, look at any "intermediate" neighbor nodes numbered
    // less, and then see if this neighbor has another neighbor j numbered
    // more than the intermediate; if so, fill in is (i,j) if it is not
    // already nonzero
// start = magma_sync_wtime( queue );
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        //for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            
            
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
            //for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_index_t checkel=L.row[cand_row];
                do{
                    checkcol = L.col[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=0;
                    }else{
                        checkel = L.list[checkel];
                    }
                }while( checkel != 0 );
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numaddrowL++;
                    //numaddL[ row+1 ]++;
                }
                el2 = L.list[ el2 ];
            }
            el1 = L.list[ el1 ];
        }
        numaddU[ row+1 ] = numaddrowU;
        numaddL[ row+1 ] = numaddrowL;
    }
            
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        //for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if( cand_col < row ){
                    // check whether this element already exists
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=L.row[cand_row];
                    do{
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = L.list[checkel];
                        }
                    }while( checkel != 0 );
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                        numaddrowL++;
                        //numaddL[ row+1 ]++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=U.row[cand_col];
                    do{
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = U.list[checkel];
                        }
                    }while( checkel != 0 );
                    if( exist == 0 ){
                        numaddrowU++;
                        //numaddU[ row+1 ]++;
                    }
                }
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            el1 = L.list[ el1 ];
        }
        numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
    }
    
    //#######################
    if( unsym == 1 ){
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){

        magma_index_t start1 = UR.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
                          // printf("row:%d el:%d\n", row, el1);
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
            //for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                //printf("check cand (%d,%d)\n",cand_row, cand_col);
                // check whether this element already exists
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=L.row[cand_row];
                    do{
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = L.list[checkel];
                        }
                    }while( checkel != 0 );
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        numaddrowL++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=U.row[cand_col];
                    do{
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = U.list[checkel];
                        }
                    }while( checkel != 0 );
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        numaddrowU++;
                    }
                }
                el2 = L.list[ el2 ];
            }
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
        
        numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
    }
    
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            //if( col1 < row ){
            //    printf("row:%d col1:%d\n", row, col1);
            //    exit(-1);
            //}
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
            // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_index_t checkel=U.row[cand_col];
                do{
                    checkcol = U.col[ checkel ];
                    if( checkcol == cand_row ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=0;
                    }else{
                        checkel = U.list[checkel];
                    }
                }while( checkel != 0 );
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numaddrowU++;
                }
                
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
         
        numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
    } //loop over all rows
    }
    
    //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
    
    // #########################################################################
    
      /*
    // get the total candidate count
    L_new->nnz = 0;
    U_new->nnz = 0;
    numaddL[ 0 ] = L_new->nnz;
    numaddU[ 0 ] = U_new->nnz;
    L_new->row[ 0 ] = 0;
    U_new->row[ 0 ] = 0;
    for( magma_int_t i = 0; i<L.num_rows; i++ ){
        L_new->nnz=L_new->nnz + numaddL[ i+1 ];
        U_new->nnz=U_new->nnz + numaddU[ i+1 ];
        numaddL[ i+1 ] = L_new->nnz;
        numaddU[ i+1 ] = U_new->nnz;
        L_new->row[ i+1 ] = 0;
        U_new->row[ i+1 ] = 0;

    }
    */
    // get the total candidate count
    L_new->nnz = 0;
    U_new->nnz = 0;
    numaddL[ 0 ] = L_new->nnz;
    numaddU[ 0 ] = U_new->nnz;
    
    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
            L_new->row[ i ] = 0;
            U_new->row[ i ] = 0;
    }
  /*  if( num_threads > 1 ){
        #pragma omp parallel
        {
            magma_int_t id = omp_get_thread_num();
            if( id == 0 ){
                for( magma_int_t i = 0; i<L.num_rows; i++ ){
                    L_new->nnz=L_new->nnz + numaddL[ i+1 ];
                    numaddL[ i+1 ] = L_new->nnz;
                    L_new->row[ i+1 ] = 0;
                }
            } else if( id == 1 ){
                for( magma_int_t i = 0; i<L.num_rows; i++ ){
                    U_new->nnz=U_new->nnz + numaddU[ i+1 ];
                    numaddU[ i+1 ] = U_new->nnz;
                    U_new->row[ i+1 ] = 0;
                }
            }
        }
    } else {*/
    L_new->nnz = 0;
    U_new->nnz = 0;
    numaddL[ 0 ] = L_new->nnz;
    numaddU[ 0 ] = U_new->nnz;
    for( magma_int_t i = 0; i<L.num_rows; i++ ){
        L_new->nnz=L_new->nnz + numaddL[ i+1 ];
        U_new->nnz=U_new->nnz + numaddU[ i+1 ];
        numaddL[ i+1 ] = L_new->nnz;
        numaddU[ i+1 ] = U_new->nnz;
    }
    //}

        

    if( L_new->nnz > L_new->true_nnz ){
       magma_free_cpu( L_new->val );
       magma_free_cpu( L_new->row );
       magma_free_cpu( L_new->rowidx );
       magma_free_cpu( L_new->col );
       magma_free_cpu( L_new->list );
       magma_zmalloc_cpu( &L_new->val, L_new->nnz*2 );
       magma_index_malloc_cpu( &L_new->rowidx, L_new->nnz*2 );
       magma_index_malloc_cpu( &L_new->col, L_new->nnz*2 );
       magma_index_malloc_cpu( &L_new->row, L_new->nnz*2 );
       magma_index_malloc_cpu( &L_new->list, L_new->nnz*2 ); 
       L_new->true_nnz = L_new->nnz*2;
    }
    
    if( U_new->nnz > U_new->true_nnz ){
       magma_free_cpu( U_new->val );
       magma_free_cpu( U_new->row );
       magma_free_cpu( U_new->rowidx );
       magma_free_cpu( U_new->col );
       magma_free_cpu( U_new->list );
       magma_zmalloc_cpu( &U_new->val, U_new->nnz*2 );
       magma_index_malloc_cpu( &U_new->rowidx, U_new->nnz*2 );
       magma_index_malloc_cpu( &U_new->col, U_new->nnz*2 );
       magma_index_malloc_cpu( &U_new->row, U_new->nnz*2 );
       magma_index_malloc_cpu( &U_new->list, U_new->nnz*2 ); 
       U_new->true_nnz = U_new->nnz*2;
    }
    // #########################################################################
    
    //start = magma_sync_wtime( queue );
    // insertion here
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        //for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
            //for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_index_t checkel=L.row[cand_row];
                do{
                    checkcol = L.col[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=0;
                    }else{
                        checkel = L.list[checkel];
                    }
                }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                    for( magma_index_t checkel=numaddL[cand_row]; checkel<offsetL+laddL; checkel++){
                        checkcol = L_new->col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=offsetL+laddL;
                        }
                    }
#endif
                
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                    //add in the next location for this row
                    // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                    L_new->rowidx[ offsetL + laddL ] = cand_row;
                    L_new->col[ offsetL + laddL ] = cand_col;
                    // L_new->list[ numaddL[row] + laddL ] =  -1;
                    // L_new->row[ numaddL[row] + laddL ] =  -1;
                    laddL++; 
                }
                el2 = L.list[ el2 ];
            }
            el1 = L.list[ el1 ];
        }
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    }
            
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        //for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_int_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=L.row[cand_row];
                    do{
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = L.list[checkel];
                        }
                    }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=numaddL[cand_row]; checkel<offsetL+laddL; checkel++){
                            checkcol = L_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetL+laddL;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                        //add in the next location for this row
                        // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        L_new->rowidx[ offsetL + laddL ] = cand_row;
                        L_new->col[ offsetL + laddL ] = cand_col;
                        // L_new->list[ numaddL[row] + laddL ] = -1;
                        // L_new->row[ numaddL[row] + laddL ] = -1;
                        laddL++; 
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=U.row[cand_col];
                    do{
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = U.list[checkel];
                        }
                    }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=numaddU[cand_row]; checkel<offsetU+laddU; checkel++){
                            checkcol = U_new->col[ checkel ];
                            if( checkcol == cand_row ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetU+laddU;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //  printf("---------------->>>  candidate in U at (%d, %d) stored in %d\n", cand_row, cand_col, numaddU[row] + laddU);
                        //add in the next location for this row
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ offsetU + laddU ] = cand_col;
                        U_new->rowidx[ offsetU + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++; 
                        //if( cand_row > cand_col )
                         //   printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
                    }
                }
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            el1 = L.list[ el1 ];
        }
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    }
        
    
    //#######################
    if( unsym == 1 ){
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            //if( col1 < row ){
            //    printf("row:%d col1:%d\n", row, col1);
            //    exit(-1);
            //}
                
              
          // printf("row:%d el:%d\n", row, el1);
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
            //for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                // check whether this element already exists
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=L.row[cand_row];
                    do{
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = L.list[checkel];
                        }
                    }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=numaddL[cand_row]; checkel<offsetL+laddL; checkel++){
                            checkcol = L_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetL+laddL;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        L_new->rowidx[ offsetL + laddL ] = cand_row;
                        L_new->col[ offsetL + laddL ] = cand_col;
                        // L_new->list[ numaddL[row] + laddL ] =  -1;
                        // L_new->row[ numaddL[row] + laddL ] =  -1;
                        laddL++; 
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=U.row[cand_col];
                    do{
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = U.list[checkel];
                        }
                    }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=numaddU[cand_row]; checkel<offsetU+laddU; checkel++){
                            checkcol = U_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetU+laddU;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ offsetU + laddU ] = cand_col;
                        U_new->rowidx[ offsetU + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++; 
                    }
                }
                el2 = L.list[ el2 ];
            }
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
        
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    }
    
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        magma_index_t el1 = start1;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            //if( col1 < row ){
            //    printf("row:%d col1:%d\n", row, col1);
            //    exit(-1);
            //}
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_index_t checkel=U.row[cand_col];
                do{
                    checkcol = U.col[ checkel ];
                    if( checkcol == cand_row ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=0;
                    }else{
                        checkel = U.list[checkel];
                    }
                }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                    for( magma_index_t checkel=numaddU[cand_row]; checkel<offsetU+laddU; checkel++){
                        checkcol = U_new->col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=offsetU+laddU;
                        }
                    }
#endif
               // if it does not exist, increase counter for this location
               // use the entry one further down to allow for parallel insertion
               if( exist == 0 ){
                   // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                   U_new->col[ offsetU + laddU ] = cand_col;
                   U_new->rowidx[ offsetU + laddU ] = cand_row;
                   // U_new->list[ numaddU[row] + laddU ] = -1;
                   // U_new->row[ numaddU[row] + laddU ] = -1;
                   laddU++; 
                   //if( cand_row == 118 && cand_col == 163 )
                   //         printf("checked row insetion %d this element does not yet exist in U: (%d,%d) row starts with (%d,%d)\n", cand_row, cand_row, cand_col, U.col[ checkel ], cand_col );
                   //if( cand_row > cand_col )
                     //  printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
               }
               
               el2 = UR.list[ el2 ];
           }while( el2 != 0 );
            
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
        
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
        
    } //loop over all rows
    }
    
#ifdef AVOID_DUPLICATES
        // #####################################################################
        
        // get the total candidate count
        L_new->nnz = 0;
        U_new->nnz = 0;
        /*  if( num_threads > 1 ){
            #pragma omp parallel
            {
                magma_int_t id = omp_get_thread_num();
                if( id == 0 ){
                    for( magma_int_t i = 0; i<L.num_rows; i++ ){
                        L_new->nnz=L_new->nnz + L_new->row[ i+1 ];
                        L_new->row[ i+1 ] = L_new->nnz;
                    }
                } else if( id == 1 ){
                    for( magma_int_t i = 0; i<L.num_rows; i++ ){
                        U_new->nnz=U_new->nnz + U_new->row[ i+1 ];
                        U_new->row[ i+1 ] = U_new->nnz;
                    }
                }
            }
        } else {
            for( magma_int_t i = 0; i<L.num_rows; i++ ){
                L_new->nnz=L_new->nnz + L_new->row[ i+1 ];
                U_new->nnz=U_new->nnz + U_new->row[ i+1 ];
                L_new->row[ i+1 ] = L_new->nnz;
                U_new->row[ i+1 ] = U_new->nnz;
            }
        }*/
        for( magma_int_t i = 0; i<L.num_rows; i++ ){
            L_new->nnz=L_new->nnz + L_new->row[ i+1 ];
            U_new->nnz=U_new->nnz + U_new->row[ i+1 ];
            L_new->row[ i+1 ] = L_new->nnz;
            U_new->row[ i+1 ] = U_new->nnz;
        }

        #pragma omp parallel for
        for( magma_int_t i = 0; i<L.num_rows; i++ ){
            for( magma_int_t j=L_new->row[ i ]; j<L_new->row[ i+1 ]; j++ ){
                magma_int_t k = j-L_new->row[ i ]+numaddL[ i ];
                //printf("j:%d, numaddL[ i+1 ]=%d\n", j, numaddL[ i+1 ]);
                colL[ j ]    =  L_new->col[ k ];
                rowidxL[ j ] =  L_new->rowidx[ k ];
            }
        }
        #pragma omp parallel for
        for( magma_int_t i = 0; i<L.num_rows; i++ ){
            for( magma_int_t j=U_new->row[ i ]; j<U_new->row[ i+1 ]; j++ ){
                magma_int_t k = j-U_new->row[ i ]+numaddU[ i ];
                colU[ j ]    =  U_new->col[ k ];
                rowidxU[ j ] =  U_new->rowidx[ k ];
            }
        }
        

        
        rowidxLt = L_new->rowidx;
        colLt = L_new->col;
        rowidxUt = U_new->rowidx;
        colUt = U_new->col;
        
        L_new->rowidx = rowidxL;
        L_new->col = colL;
        U_new->rowidx = rowidxU;
        U_new->col = colU;
        
        rowidxL = rowidxLt;
        colL = colLt;
        rowidxU = rowidxUt;
        colU = colUt;
        //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
        // #####################################################################
#endif
   
cleanup:
    magma_free_cpu( numaddL );
    magma_free_cpu( numaddU );
    magma_free_cpu( rowidxL );
    magma_free_cpu( rowidxU );
    magma_free_cpu( colL );
    magma_free_cpu( colU );
    return info;
}







/**
    Purpose
    -------
    This function identifies the candidates like they appear as ILU1 fill-in.

    Arguments
    ---------
    
    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.
                
    @param[in]
    U           magma_z_matrix
                Current upper triangular factor transposed.
                
    @param[in]
    UR          magma_z_matrix
                Current upper triangular factor - col-pointer and col-list.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for L in COO format.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for U in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_candidates_onesweep(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix UR,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    real_Double_t start, end;
    magma_index_t *numaddL;
    magma_index_t *numaddU;
    magma_index_t *rowidxL, *rowidxU, *colL, *colU;
    magma_index_t *rowidxLt, *rowidxUt, *colLt, *colUt;
    magma_int_t num_threads=1, chunksize;
    magma_int_t startidx, endidx, offset;
    magma_int_t el_per_block;
    
    magma_int_t unsym = 1;
    
    magma_int_t avoid_duplicates = 1;
    
    CHECK( magma_index_malloc_cpu( &numaddL, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &numaddU, U.num_rows+1 ));
    
    CHECK( magma_index_malloc_cpu( &rowidxL, L_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &rowidxU, U_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &colL, L_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &colU, U_new->true_nnz ));
    
    
    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
        L_new->row[ i ] = 0;
        U_new->row[ i ] = 0;
    }


     
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    
    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
        numaddL[i] = 0;
        numaddU[i] = 0;
    }
    
    start = magma_sync_wtime( queue );
    // insertion here
    // parallel loop
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t lbound = (id+1)*el_per_block;
        if( id == num_threads-1 ){
            lbound = L_new->num_rows;    
        }
        printf("%%thread %d works from %d to %d\n", id, (id)*el_per_block, lbound);
        for( magma_int_t row=(id)*el_per_block; row<lbound; row++ ){
        
        magma_index_t start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = id*L_new->blocksize+numaddL[id+1];
        magma_int_t offsetU = id*U_new->blocksize+numaddU[id+1];
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        //while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            //while( L.list[ el2 ] != 0 ) {
            for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                for( magma_index_t checkel=L.row[cand_row]; checkel<L.row[cand_row+1]; checkel++){
                    checkcol = L.col[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=L.row[cand_row+1];
                    }
                }
#ifdef AVOID_DUPLICATES
                    for( magma_index_t checkel=id*L_new->blocksize; checkel<offsetL+laddL; checkel++){
                        checkcol = L_new->col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=offsetL+laddL;
                        }
                    }
#endif
                
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                    //add in the next location for this row
                    // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                    L_new->rowidx[ offsetL + laddL ] = cand_row;
                    L_new->col[ offsetL + laddL ] = cand_col;
                    // L_new->list[ numaddL[row] + laddL ] =  -1;
                    // L_new->row[ numaddL[row] + laddL ] =  -1;
                    laddL++; 
                }
                //el2 = L.list[ el2 ];
            }
            
        }
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
        numaddL[row+1] = numaddL[row+1]+laddL;
        numaddU[row+1] = numaddU[row+1]+laddU;
        printf("%%thread %d candidates: %d %d\n", id, L_new->row[row+1], U_new->row[row+1]);
    }
    }
            
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t lbound = (id+1)*el_per_block;
        if( id == num_threads-1 ){
            lbound = L_new->num_rows;    
        }
        printf("%%thread %d works from %d to %d\n", id, (id)*el_per_block, lbound);
        for( magma_int_t row=(id)*el_per_block; row<lbound; row++ ){
        magma_index_t start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = id*L_new->blocksize+numaddL[id+1];
        magma_int_t offsetU = id*U_new->blocksize+numaddU[id+1];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        //while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_int_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    for( magma_index_t checkel=L.row[cand_row]; checkel<L.row[cand_row+1]; checkel++){
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=L.row[cand_row+1];
                        }
                    }
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=id*L_new->blocksize; checkel<offsetL+laddL; checkel++){
                            checkcol = L_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetL+laddL;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                        //add in the next location for this row
                        // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        L_new->rowidx[ offsetL + laddL ] = cand_row;
                        L_new->col[ offsetL + laddL ] = cand_col;
                        // L_new->list[ numaddL[row] + laddL ] = -1;
                        // L_new->row[ numaddL[row] + laddL ] = -1;
                        laddL++; 
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    for( magma_index_t checkel=U.row[cand_col]; checkel<U.row[cand_col+1]; checkel++){
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=U.row[cand_col+1];
                        }
                    }
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=id*U_new->blocksize; checkel<offsetU+laddU; checkel++){
                            checkcol = U_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetU+laddU;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //  printf("---------------->>>  candidate in U at (%d, %d) stored in %d\n", cand_row, cand_col, numaddU[row] + laddU);
                        //add in the next location for this row
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ offsetU + laddU ] = cand_col;
                        U_new->rowidx[ offsetU + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++; 
                        if( cand_row > cand_col )
                            printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
                    }
                }
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            //el1 = L.list[ el1 ];
        }

        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
        numaddL[row+1] = numaddL[row+1]+laddL;
        numaddU[row+1] = numaddU[row+1]+laddU;
    }
    }
        
    
    //#######################
    if( unsym == 1 ){
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t lbound = (id+1)*el_per_block;
        if( id == num_threads-1 ){
            lbound = L_new->num_rows;    
        }
        printf("%%thread %d in unsym works from %d to %d\n", id, (id)*el_per_block, lbound);
        for( magma_int_t row=(id)*el_per_block; row<lbound; row++ ){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = id*L_new->blocksize+numaddL[id+1];
        magma_int_t offsetU = id*U_new->blocksize+numaddU[id+1];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            if( col1 < row ){
                printf("row:%d col1:%d\n", row, col1);
                exit(-1);
            }
                
              
          // printf("row:%d el:%d\n", row, el1);
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            //while( L.list[ el2 ] != 0 ) {
            for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                // check whether this element already exists
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    for( magma_index_t checkel=L.row[cand_row]; checkel<L.row[cand_row+1]; checkel++){
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=L.row[cand_row+1];
                        }
                    }
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=id*L_new->blocksize; checkel<offsetL+laddL; checkel++){
                            checkcol = L_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetL+laddL;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        L_new->rowidx[ offsetL + laddL ] = cand_row;
                        L_new->col[ offsetL + laddL ] = cand_col;
                        // L_new->list[ numaddL[row] + laddL ] =  -1;
                        // L_new->row[ numaddL[row] + laddL ] =  -1;
                        laddL++; 
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    for( magma_index_t checkel=U.row[cand_col]; checkel<U.row[cand_col+1]; checkel++){
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=U.row[cand_col+1];
                        }
                    }
#ifdef AVOID_DUPLICATES
                        for( magma_index_t checkel=id*U_new->blocksize; checkel<offsetU+laddU; checkel++){
                            checkcol = U_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetU+laddU;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ offsetU + laddU ] = cand_col;
                        U_new->rowidx[ offsetU + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++; 
                    }
                }
                //el2 = L.list[ el2 ];
            }
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
        
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
        numaddL[row+1] = numaddL[row+1]+laddL;
        numaddU[row+1] = numaddU[row+1]+laddU;
    }
    }
    
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t lbound = (id+1)*el_per_block;
        if( id == num_threads-1 ){
            lbound = L_new->num_rows;    
        }
        printf("%%thread %d works from %d to %d\n", id, (id)*el_per_block, lbound);
        for( magma_int_t row=(id)*el_per_block; row<lbound; row++ ){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = id*L_new->blocksize+numaddL[id+1];
        magma_int_t offsetU = id*U_new->blocksize+numaddU[id+1];
        magma_index_t el1 = start1;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            if( col1 < row ){
                printf("row:%d col1:%d\n", row, col1);
                exit(-1);
            }
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                for( magma_index_t checkel=U.row[cand_col]; checkel<U.row[cand_col+1]; checkel++){
                    checkcol = U.col[ checkel ];
                    if( checkcol == cand_row ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=U.row[cand_col+1];
                    }
                }
#ifdef AVOID_DUPLICATES
                    for( magma_index_t checkel=id*U_new->blocksize; checkel<offsetU+laddU; checkel++){
                        checkcol = U_new->col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=offsetU+laddU;
                        }
                    }
#endif
               // if it does not exist, increase counter for this location
               // use the entry one further down to allow for parallel insertion
               if( exist == 0 ){
                   // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                   U_new->col[ offsetU + laddU ] = cand_col;
                   U_new->rowidx[ offsetU + laddU ] = cand_row;
                   // U_new->list[ numaddU[row] + laddU ] = -1;
                   // U_new->row[ numaddU[row] + laddU ] = -1;
                   laddU++; 
                   //if( cand_row == 118 && cand_col == 163 )
                   //         printf("checked row insetion %d this element does not yet exist in U: (%d,%d) row starts with (%d,%d)\n", cand_row, cand_row, cand_col, U.col[ checkel ], cand_col );
                   if( cand_row > cand_col )
                       printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
               }
               
               el2 = UR.list[ el2 ];
           }while( el2 != 0 );
            
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
        
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
        numaddL[row+1] = numaddL[row+1]+laddL;
        numaddU[row+1] = numaddU[row+1]+laddU;
         printf("%%thread %d candidates: %d < %d and %d < %d\n", id, L_new->row[row+1], L_new->blocksize, U_new->row[row+1], U_new->blocksize);
        
    } //loop over all rows
    }
    
    }
    
#ifdef AVOID_DUPLICATES
        // #####################################################################
        
        // get the total candidate count
        L_new->nnz = 0;
        U_new->nnz = 0;
        
        for( magma_int_t i = 0; i<L.num_rows; i++ ){
            L_new->nnz=L_new->nnz + L_new->row[ i+1 ];
            U_new->nnz=U_new->nnz + U_new->row[ i+1 ];
            L_new->row[ i+1 ] = L_new->nnz;
            U_new->row[ i+1 ] = U_new->nnz;
        }
        printf("candidates :%d %d\n", L_new->nnz, U_new->nnz);
        
        for( magma_int_t i = 0; i<num_threads; i++ ){
            startidx = L_new->row[i*el_per_block];
            endidx = L_new->row[(i+1)*el_per_block];
            if( i == num_threads-1 ){
                endidx = L_new->nnz;    
            }
            offset = i*L_new->blocksize;
            printf("copy %d-%d -> %d-%d\n", offset, offset+endidx-startidx, startidx, endidx);
            #pragma omp parallel for
            for( magma_int_t j=0; j<endidx-startidx; j++ ){
                colL[ j+startidx ]    =  L_new->col[ offset + j ];
                rowidxL[ j+startidx ] =  L_new->rowidx[ offset + j ];    
            }
        }
        
        for( magma_int_t i = 0; i<num_threads; i++ ){
            startidx = U_new->row[i*el_per_block];
            endidx = U_new->row[(i+1)*el_per_block];
            if( i == num_threads-1 ){
                endidx = U_new->nnz;    
            }
            offset = i*U_new->blocksize;
            printf("copy %d-%d -> %d-%d\n", offset, offset+endidx-startidx, startidx, endidx);
            #pragma omp parallel for
            for( magma_int_t j=0; j<endidx-startidx; j++ ){
                colU[ j+startidx ]    =  U_new->col[ offset + j ];
                rowidxU[ j+startidx ] =  U_new->rowidx[ offset + j ];    
            }
        }

        rowidxLt = L_new->rowidx;
        colLt = L_new->col;
        rowidxUt = U_new->rowidx;
        colUt = U_new->col;
        
        L_new->rowidx = rowidxL;
        L_new->col = colL;
        U_new->rowidx = rowidxU;
        U_new->col = colU;
        
        rowidxL = rowidxLt;
        colL = colLt;
        rowidxU = rowidxUt;
        colU = colUt;
        //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
        // #####################################################################
#endif
   
cleanup:
    magma_free_cpu( numaddL );
    magma_free_cpu( numaddU );
    magma_free_cpu( rowidxL );
    magma_free_cpu( rowidxU );
    magma_free_cpu( colL );
    magma_free_cpu( colU );
    return info;
}





/**
    Purpose
    -------
    This function identifies the candidates like they appear as ILU1 fill-in.

    Arguments
    ---------
    
    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.
                
    @param[in]
    U           magma_z_matrix
                Current upper triangular factor transposed.
                
    @param[in]
    UR          magma_z_matrix
                Current upper triangular factor - col-pointer and col-list.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for L in COO format.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for U in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_candidates_o(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix UR,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    real_Double_t start, end;
    magma_index_t *numaddL;
    magma_index_t *numaddU;
    magma_index_t start2, start1;
    
    CHECK( magma_index_malloc_cpu( &numaddL, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &numaddU, U.num_rows+1 ));
    
    
    
    
    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
        numaddL[i] = 0;
        numaddU[i] = 0;
    }
     
    // how to determine candidates:
    // for each node i, look at any "intermediate" neighbor nodes numbered
    // less, and then see if this neighbor has another neighbor j numbered
    // more than the intermediate; if so, fill in is (i,j) if it is not
    // already nonzero
start = magma_sync_wtime( queue );
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        start1 = L.row[ row ];
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // first check the lower triangular
            start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t nextel, checkel = L.row[ cand_row ];//el1;
                magma_index_t checkcol = L.col[ checkel ];
                while( checkel!=0 ) {
                    nextel = L.list[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        nextel=0;
                    }
                    checkel = nextel;
                    checkcol = L.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numaddL[ row+1 ]++;
                }
                el2 = L.list[ el2 ];
            }
            // now check the upper triangular
            start2 = UR.row[ col1 ];
            el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t nextel, checkel = L.row[ cand_row ];//el1;
                    magma_index_t checkcol = L.col[ checkel ];
                    while( checkel!=0 ) {
                        nextel = L.list[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            nextel=0;
                        }
                        checkel = nextel;
                        checkcol = L.col[ checkel ];
                    }
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                        numaddL[ row+1 ]++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t nextel, checkel = U.row[ cand_col ];//el1;
                    magma_index_t checkcol = U.col[ checkel ];
                    do{
                        nextel = U.list[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            nextel=0;
                        }
                        checkel = nextel;
                        checkcol = U.col[ checkel ];
                    }while( checkel!=0 );
                    // printf("\n");
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        numaddU[ row+1 ]++;
                    }
                }
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            el1 = L.list[ el1 ];
        }
        
        //#######################
        start1 = UR.row[ row ];
        
        el1 = start1;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
                          // printf("row:%d el:%d\n", row, el1);
            // first check the lower triangular
            start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                //printf("check cand (%d,%d)\n",cand_row, cand_col);
                // check whether this element already exists
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkel = L.row[ cand_row ];//el1;
                    magma_index_t checkcol = L.col[ checkel ];
                    while( checkel!=0 ) {
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            break;
                        }
                        checkel = L.list[ checkel ];
                        checkcol = L.col[ checkel ];
                    }
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        numaddL[ row+1 ]++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkel = U.row[ cand_col ];//el1;
                    magma_index_t checkcol = U.col[ checkel ];
                    do{
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            break;
                        }
                        checkel = U.list[ checkel ];
                        checkcol = U.col[ checkel ];
                    }while( checkel!=0 );
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        numaddU[ row+1 ]++;
                    }
                }
                el2 = L.list[ el2 ];
            }
            // now check the upper triangular
            start2 = UR.row[ col1 ];
            el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
            // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = U.row[ cand_col ];//el1;
                magma_index_t checkcol = U.col[ checkel ];
                do{
                    if( checkcol == cand_row ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    checkel = U.list[ checkel ];
                    checkcol = U.col[ checkel ];
                }while( checkel!=0 );
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numaddU[ row+1 ]++;
                }
                
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
         
        
    } //loop over all rows
    end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
    
    // #########################################################################
    
        
    // get the total candidate count
    L_new->nnz = 0;
    U_new->nnz = 0;
    numaddL[ 0 ] = L_new->nnz;
    numaddU[ 0 ] = U_new->nnz;
    for( magma_int_t i = 0; i<L.num_rows; i++ ){
        L_new->nnz=L_new->nnz + numaddL[ i+1 ];
        U_new->nnz=U_new->nnz + numaddU[ i+1 ];
        numaddL[ i+1 ] = L_new->nnz;
        numaddU[ i+1 ] = U_new->nnz;
    }
    // printf("cand count:%d %d\n", L_new->nnz, U_new->nnz);
    if( L_new->nnz > L.nnz*20 || U_new->nnz > U.nnz*20 ){
        printf("error: more candidates than space allocated: max(%d,%d) > %d. Increase candidate allocation.\n", L_new->nnz, U_new->nnz, L.nnz*20 );
        goto cleanup;
    }
    
    // #########################################################################
    
    start = magma_sync_wtime( queue );
    // insertion here
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            
            // first check the lower triangular
            start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = L.row[ cand_row ];//el1;
                magma_index_t checkcol = L.col[ checkel ];
                while( checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    checkel = L.list[ checkel ];
                    checkcol = L.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                    //add in the next location for this row
                    // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                    L_new->rowidx[ numaddL[row] + laddL ] = cand_row;
                    L_new->col[ numaddL[row] + laddL ] = cand_col;
                    // L_new->list[ numaddL[row] + laddL ] =  -1;
                    // L_new->row[ numaddL[row] + laddL ] =  -1;
                    laddL++; 
                }
                el2 = L.list[ el2 ];
            }
            
            // now check the upper triangular
            start2 = UR.row[ col1 ];
            el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
             do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkel = L.row[ cand_row ];//el1;
                    magma_index_t checkcol = L.col[ checkel ];
                    while( checkel!=0 ) {
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            break;
                        }
                        checkel = L.list[ checkel ];
                        checkcol = L.col[ checkel ];
                    }
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                        //add in the next location for this row
                        // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        L_new->rowidx[ numaddL[row] + laddL ] = cand_row;
                        L_new->col[ numaddL[row] + laddL ] = cand_col;
                        // L_new->list[ numaddL[row] + laddL ] = -1;
                        // L_new->row[ numaddL[row] + laddL ] = -1;
                        laddL++; 
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkel = U.row[ cand_col ];//el1;
                    magma_index_t checkcol = U.col[ checkel ];
                    do{
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            break;
                        }
                        checkel = U.list[ checkel ];
                        checkcol = U.col[ checkel ];
                    }while( checkel!=0 );
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //  printf("---------------->>>  candidate in U at (%d, %d) stored in %d\n", cand_row, cand_col, numaddU[row] + laddU);
                        //add in the next location for this row
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ numaddU[row] + laddU ] = cand_col;
                        U_new->rowidx[ numaddU[row] + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++; 
                        if( cand_row > cand_col )
                            printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
                    }
                }
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );
            
            el1 = L.list[ el1 ];
        }
        
        
        //#######################
        
        
        start1 = UR.row[ row ];
        
        el1 = start1;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            if( col1 < row ){
                printf("row:%d col1:%d\n", row, col1);
                exit(-1);
            }
                
                
                          // printf("row:%d el:%d\n", row, el1);
            // first check the lower triangular
            start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                // check whether this element already exists
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkel = L.row[ cand_row ];//el1;
                    magma_index_t checkcol = L.col[ checkel ];
                    while( checkel!=0 ) {
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            break;
                        }
                        checkel = L.list[ checkel ];
                        checkcol = L.col[ checkel ];
                    }
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        L_new->rowidx[ numaddL[row] + laddL ] = cand_row;
                        L_new->col[ numaddL[row] + laddL ] = cand_col;
                        // L_new->list[ numaddL[row] + laddL ] =  -1;
                        // L_new->row[ numaddL[row] + laddL ] =  -1;
                        laddL++; 
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkel = U.row[ cand_col ];//el1;
                    magma_index_t checkcol = U.col[ checkel ];
                    do{
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            break;
                        }
                        checkel = U.list[ checkel ];
                        checkcol = U.col[ checkel ];
                    }while( checkel!=0 );
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ numaddU[row] + laddU ] = cand_col;
                        U_new->rowidx[ numaddU[row] + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++; 
                        if( cand_row > cand_col )
                            printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
                    }
                }
                el2 = L.list[ el2 ];
            }
            // now check the upper triangular
            start2 = UR.row[ col1 ];
            el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            do{
               magma_index_t col2 = UR.col[ el2 ];
               magma_index_t cand_row = row;
               magma_index_t cand_col = col2;
           // check whether this element already exists
               magma_int_t exist = 0;
               magma_index_t checkel = U.row[ cand_col ];//el1;
               magma_index_t checkcol = U.col[ checkel ];
               do{
                   if( checkcol == cand_row ){
                       // element included in LU and nonzero
                       exist = 1;
                       break;
                   }
                   checkel = U.list[ checkel ];
                   checkcol = U.col[ checkel ];
               }while( checkel!=0 );
               // if it does not exist, increase counter for this location
               // use the entry one further down to allow for parallel insertion
               if( exist == 0 ){
                   // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                   U_new->col[ numaddU[row] + laddU ] = cand_col;
                   U_new->rowidx[ numaddU[row] + laddU ] = cand_row;
                   // U_new->list[ numaddU[row] + laddU ] = -1;
                   // U_new->row[ numaddU[row] + laddU ] = -1;
                   laddU++; 
                   //if( cand_row == 118 && cand_col == 163 )
                   //         printf("checked row insetion %d this element does not yet exist in U: (%d,%d) row starts with (%d,%d)\n", cand_row, cand_row, cand_col, U.col[ checkel ], cand_col );
                   if( cand_row > cand_col )
                       printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
               }
               
               el2 = UR.list[ el2 ];
           }while( el2 != 0 );
            
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );
        
        
        
    } //loop over all rows
    end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
   
cleanup:
    magma_free_cpu( numaddL );
    magma_free_cpu( numaddU );
    return info;
}




 
/**
    Purpose
    -------
    This routine removes matrix entries from the structure that are smaller
    than the threshold.

    Arguments
    ---------

    @param[out]
    thrs        magmaDoubleComplex*
                Thrshold for removing elements.
                
    @param[out]
    num_rm      magma_int_t*
                Number of Elements that have been removed.
                
    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[out]
    rm_loc      magma_index_t*
                List containing the locations of the elements deleted.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_rm_thrs_save(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *LU,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    // never forget elements
    // magma_int_t offset = LU_new->diameter;
    // never forget last rm
    magma_int_t offset = 0;
    
    #pragma omp parallel for
    for( magma_int_t r=0; r < LU->num_rows; r++ ) {
        magma_int_t i = LU->row[r];
        magma_int_t lasti=i;
        magma_int_t nexti=LU->list[i];
        while( nexti!=0 ){
            if( MAGMA_Z_ABS( LU->val[ i ] ) <  MAGMA_Z_ABS(*thrs) ){
                // the condition nexti!=0 esures we never remove the diagonal
                    //magmaDoubleComplex valt = LU->val[ i ];
                    //LU->val[ i ] = MAGMA_Z_ZERO;
                    LU->list[ i ] = -1;
                    if( LU->col[ i ] == r ){
                        printf("error: try to rm diagonal in L.\n");   
                    }
                    omp_set_lock(&(counter));
                    rm_loc[ count_rm ] = i; 
                    // keep it as potential fill-in candidate
                    // LU_new->col[ count_rm+offset ] = LU->col[ i ];
                    // LU_new->rowidx[ count_rm+offset ] = r;
                    // LU_new->val[ count_rm+offset ] = valt; // MAGMA_Z_MAKE(1e-14,0.0);
                    // printf(" in L rm: (%d,%d) [%d] \n", r, LU->col[ i ], count_rm); fflush(stdout);
                    count_rm++;
                    omp_unset_lock(&(counter));
                    // either the headpointer or the linked list has to be changed
                    // headpointer if the deleted element was the first element in the row
                    // linked list to skip this element otherwise
                    if( LU->row[r] == i ){
                            LU->row[r] = nexti;
                            lasti=i;
                            i = nexti;
                            nexti = LU->list[nexti];
                    }
                    else{
                        LU->list[lasti] = nexti;
                        i = nexti;
                        nexti = LU->list[nexti];
                    }
            }
            else{
                lasti = i;
                i = nexti;
                nexti = LU->list[nexti];
            }
            
        }
    }
    // never forget elements
    // LU_new->diameter = count_rm+LU_new->diameter;
    // not forget the last rm
    LU_new->diameter = count_rm;
    LU_new->nnz = LU_new->diameter;
    *num_rm = count_rm;

    omp_destroy_lock(&(counter));
    return info;
}



/**
    Purpose
    -------
    This routine removes matrix entries from the structure that are smaller
    than the threshold. It only counts the elements deleted, does not
    save the locations.
    

    Arguments
    ---------

    @param[out]
    thrs        magmaDoubleComplex*
                Thrshold for removing elements.
                
    @param[out]
    num_rm      magma_int_t*
                Number of Elements that have been removed.
                
    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    LUC         magma_z_matrix*
                Corresponding col-list.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[out]
    rm_loc      magma_index_t*
                List containing the locations of the elements deleted.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_rm_thrs(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *LU,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    magma_int_t num_threads = -1;
    magma_int_t el_per_block;
    
    // never forget elements
    // magma_int_t offset = LU_new->diameter;
    // never forget last rm
    magma_int_t offset = 0;
    
    
    double bound = MAGMA_Z_ABS(*thrs);
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    el_per_block = magma_ceildiv( LU->num_rows, num_threads );
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++ ){
        rm_loc[i] = 0;
    }
    
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t lbound = (id+1)*el_per_block;
        if( id == num_threads-1 ){
            lbound = LU->num_rows;    
        }
        magma_int_t loc_rm_count = 0;
        for( magma_int_t r=(id)*el_per_block; r<lbound; r++ ){
            
            magma_int_t i = LU->row[r];
            magma_int_t lasti=i;
            magma_int_t nexti=LU->list[i];
            while( nexti!=0 ){
                if( MAGMA_Z_ABS( LU->val[ i ] ) < bound ){
                        loc_rm_count++; 
                        if( LU->row[r] == i ){
                       //     printf(" removed as first element in U rm: (%d,%d) at %d count [%d] \n", r, LU->col[ i ], i, count_rm); fflush(stdout);
                                LU->row[r] = nexti;
                                lasti=i;
                                i = nexti;
                                nexti = LU->list[nexti];
                        }
                        else{
                          //  printf(" removed in linked list in U rm: (%d,%d) at %d count [%d] \n", r, LU->col[ i ], i, count_rm); fflush(stdout);
                            LU->list[lasti] = nexti;
                            i = nexti;
                            nexti = LU->list[nexti];
                        }
                }
                else{
                    lasti = i;
                    i = nexti;
                    nexti = LU->list[nexti];
                }
                
            }
        }
        rm_loc[ id ] = rm_loc[ id ] + loc_rm_count;
    }
    
    for(int j=0; j<num_threads; j++){
        count_rm = count_rm + rm_loc[j];
    }
    
    // never forget elements
    // LU_new->diameter = count_rm+LU_new->diameter;
    // not forget the last rm
    LU_new->diameter = count_rm;
    LU_new->nnz = LU_new->diameter;
    *num_rm = count_rm;
  //  printf(" removed elements:%d\n", *num_rm);

    return info;
}


/**
    Purpose
    -------
    This routine removes matrix entries from the structure that are smaller
    than the threshold. It is very similar to the magma_zmpilut_rm_thrs
    routine, however, also changes the col-list provided by LC.

    Arguments
    ---------

    @param[out]
    thrs        magmaDoubleComplex*
                Thrshold for removing elements.
                
    @param[out]
    num_rm      magma_int_t*
                Number of Elements that have been removed.
                
    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    LUC         magma_z_matrix*
                Corresponding col-list.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[out]
    rm_loc      magma_index_t*
                List containing the locations of the elements deleted.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_rm_thrsrc(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *LU,
    magma_z_matrix *LUC,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    // never forget elements
    // magma_int_t offset = LU_new->diameter;
    // never forget last rm
    magma_int_t offset = 0;
    
    #pragma omp parallel for
    for( magma_int_t r=0; r < LU->num_rows; r++ ) {
        magma_int_t i = LU->row[r];
        magma_int_t lasti=i;
        magma_int_t nexti=LU->list[i];
        while( nexti!=0 ){
            if( MAGMA_Z_ABS( LU->val[ i ] ) <  MAGMA_Z_ABS(*thrs) ){
                // the condition nexti!=0 esures we never remove the diagonal
                    //magmaDoubleComplex valt = LU->val[ i ];
                    //LU->val[ i ] = MAGMA_Z_ZERO;
                    LU->list[ i ] = -1;
                    magma_index_t rm_colidx = LU->col[i];
                    if( rm_colidx == r ){
                        printf("error: try to rm diagonal in L.\n");   
                    }
                    omp_set_lock(&(counter));
                    rm_loc[ count_rm ] = i; 
                    // keep it as potential fill-in candidate
                    //LU_new->col[ count_rm+offset ] = LU->col[ i ];
                    //LU_new->rowidx[ count_rm+offset ] = r;
                    //LU_new->val[ count_rm+offset ] = valt; // MAGMA_Z_MAKE(1e-14,0.0);
                    // printf(" in L rm: (%d,%d) [%d] \n", r, LU->col[ i ], count_rm); fflush(stdout);
                    // printf(" in U rm: (%d,%d) at %d count [%d] \n", r, LU->col[ i ], i, count_rm); fflush(stdout);
                    count_rm++;
                    omp_unset_lock(&(counter));
                    // either the headpointer or the linked list has to be changed
                    // headpointer if the deleted element was the first element in the row
                    // linked list to skip this element otherwise
                    if( LU->row[r] == i ){
                   //     printf(" removed as first element in U rm: (%d,%d) at %d count [%d] \n", r, LU->col[ i ], i, count_rm); fflush(stdout);
                            LU->row[r] = nexti;
                            lasti=i;
                            i = nexti;
                            nexti = LU->list[nexti];
                    }
                    else{
                      //  printf(" removed in linked list in U rm: (%d,%d) at %d count [%d] \n", r, LU->col[ i ], i, count_rm); fflush(stdout);
                        LU->list[lasti] = nexti;
                        i = nexti;
                        nexti = LU->list[nexti];
                    }
            }
            else{
                lasti = i;
                i = nexti;
                nexti = LU->list[nexti];
            }
            
        }
    }
    
    // now handle col-list in LC
    #pragma omp parallel for
    for( magma_int_t r=0; r < LUC->num_rows; r++ ) {
        magma_int_t i = LUC->row[r];
        magma_int_t lasti=i;
        i=LUC->list[i]; 
        magma_int_t nexti=LUC->list[i];   
        while( i!=0 ){
            if( LU->list[i] == -1 ){// sync here if multiple elements per column!
                LUC->list[lasti] = LUC->list[i];    
                LUC->list[i] = -1;
                // printf(" removed in linked list in UR rm: (%d,%d) at %d count [%d] \n", LUC->rowidx[ i ], LUC->col[ i ], i, count_rm); fflush(stdout);
                i=nexti;
                nexti=LUC->list[nexti];  
            } else {
                lasti=i;
                i=nexti;
                nexti=LUC->list[nexti];  
            }
            
        }
        
    }
    
    
    // never forget elements
    // LU_new->diameter = count_rm+LU_new->diameter;
    // not forget the last rm
    LU_new->diameter = count_rm;
    LU_new->nnz = LU_new->diameter;
    *num_rm = count_rm;

    omp_destroy_lock(&(counter));
    return info;
}



/**
    Purpose
    -------
    This is a helper routine counting elements in a matrix in unordered
    Magma_CSRLIST format.

    Arguments
    ---------
                
    @param[in]
    L           magma_z_matrix*
                Matrix in Magm_CSRLIST format
                
    @param[out]
    num         magma_index_t*
                Number of elements counted.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_count(
    magma_z_matrix L,
    magma_index_t *num,
    magma_queue_t queue )
{
    magma_int_t info =0;

    (*num)=0;
    magma_int_t check = 1;
    if( 5 < L.col[L.list[L.row[5]]] &&  5 == L.rowidx[L.list[L.row[5]]] ){
        
        printf("check based on: (%d,%d)\n", L.rowidx[L.list[L.row[5]]], L.col[L.list[L.row[5]]]);
        check = -1;
    }
    
    for( magma_int_t r=0; r < L.num_rows; r++ ) {
        magma_int_t i = L.row[r];   
        magma_int_t nexti = L.list[i];   
        do{
            if(check == 1 ){
                if( L.col[i] > r ){
                    printf("error here: (%d,%d)\n",r, L.col[i]);
                    exit(-1);   
                }
            } else if(check == -1 ){
                if( L.col[i] < r ){
                    printf("error here: (%d,%d)\n",r, L.col[i]);
                    exit(-1);   
                }
            }
            if( nexti != 0 && L.col[i] >  L.col[nexti] ){
                printf("error here: %d(%d,%d) -> %d(%d,%d) \n",i,L.rowidx[i], L.col[i], nexti,L.rowidx[nexti], L.col[nexti] );
                exit(-1);   
            }

            (*num)++;
            i = nexti;
            nexti=L.list[nexti]; 

            
        }while( i!=0 );
        
    }


    return info;
}




/**
    Purpose
    -------
    Screens the new candidates for multiple elements in the same row.
    We allow for at most one new element per row.
    This changes the algorithm, but pays off in performance.
    

    Arguments
    ---------
    
    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced.

    @param[in]
    L_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).
                
    @param[in]
    U_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_select_candidates_o(
    magma_int_t num_rm,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=1, chunksize;
    magmaDoubleComplex element;
    magma_int_t i, count = 0;
    omp_lock_t counter;
    omp_init_lock(&(counter));
    magma_int_t duplicate = 0;
    double thrs = 2.0;
    magma_int_t approx, rm_approx, success = 0;
    const magma_int_t ione = 1;
    const magma_int_t iten = 1;
    real_Double_t start, end;
    magma_int_t el_per_block;
    
    magma_index_t *checkcol;
    magma_index_t *checkrow;
    magma_index_t *globalcount;
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    
    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    
    CHECK( magma_index_malloc_cpu( &globalcount, num_threads ));
    CHECK( magma_index_malloc_cpu( &checkcol, L_new->num_rows ));
    CHECK( magma_index_malloc_cpu( &checkrow, L_new->num_rows ));
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->num_rows; i++ ){
        checkcol[i] = 0;
        checkrow[i] = 0;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++ ){
        globalcount[i] = 0;
    }
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){
        L_new->list[i] = 0;
    }
    
    start = magma_sync_wtime( queue );
     //.// Idea: instead of order-statistics, identify 120th largest element, then sweep over, and copy only if not yet included.
    //.// finally, order statistics on remaining elements
       
       info = magma_zparilut_set_approx_thrs( (int)num_rm*thrs, L_new, 1, &element, queue );
        //printf("threshold:%f LU_new.nnz:%d\n", element, LU_new->nnz);
        //CHECK( magma_zorderstatistics(
        //LU_new->val, LU_new->nnz, (int)(num_rm*thrs),  1, &element, queue ) );
        //printf("done orderstatistics\n");
    
        count = 0;
        #pragma omp parallel for
        for( magma_int_t i=0; i<L_new->nnz; i++ ){
            magma_int_t localcount;
            if( MAGMA_Z_ABS(L_new->val[i]) >= MAGMA_Z_ABS(element) ){
                magma_int_t col = L_new->col[i];
                magma_int_t row = L_new->rowidx[i];
                omp_set_lock(&(counter));
                if( checkcol[col] == 0 && checkrow[row] == 0){
                    localcount = count;
                    count=count+1;
                    if( count >num_rm +10 ){
                        i = L_new->nnz;  
                    }
                    checkcol[col] = 1;
                    checkrow[row] = 1;
                    omp_unset_lock(&(counter));
                    L_new->col[localcount] = L_new->col[i];
                    L_new->rowidx[localcount] = row; 
                    magmaDoubleComplex valt = L_new->val[i];
                    L_new->val[i] = MAGMA_Z_ZERO;
                    L_new->list[localcount] = 1;
                    //L_new->val[localcount] = valt;
                } 
                else {
                    omp_unset_lock(&(counter));
                    L_new->val[i] = MAGMA_Z_ZERO;
                }
                
            }
        }

        
        if( count < num_rm ){
             
            while( count < num_rm ){
                printf("%% error: %d < %d -> increase threshold: %.2e\n", count, num_rm, element);
                element = element * MAGMA_Z_MAKE(.5,0.0);
                if( MAGMA_Z_ABS(element)<1e-3 )
                    exit(-1);
                #pragma omp parallel for
                for( magma_int_t i=count; i<L_new->nnz; i++ ){
                    magma_int_t localcount;
                    if( MAGMA_Z_ABS(L_new->val[i]) >= MAGMA_Z_ABS(element) ){
                        magma_int_t col = L_new->col[i];
                        magma_int_t row = L_new->rowidx[i];
                        omp_set_lock(&(counter));
                        if( checkcol[col] == 0 && checkrow[row] == 0){
                            localcount = count;
                            count=count+1;
                            if( count >num_rm +10 ){
                                i = L_new->nnz;  
                            }
                            checkcol[col] = 1;
                            checkrow[row] = 1;
                            omp_unset_lock(&(counter));
                            L_new->col[localcount] = L_new->col[i];
                            L_new->rowidx[localcount] = row; 
                            magmaDoubleComplex valt = L_new->val[i];
                            L_new->val[i] = MAGMA_Z_ZERO;
                            L_new->list[localcount] = 1;
                            //L_new->val[localcount] = valt;
                        } 
                        else {
                            omp_unset_lock(&(counter));
                            L_new->val[i] = MAGMA_Z_ZERO;
                        }
                        
                    }
                }
                printf("%% now: %d > %d\n", count, num_rm);
            }
            
             printf("%% finally: %d > %d\n", count, num_rm);
          //exit(-1);
        //} else if ( count > num_rm*thrs ) {
        //  printf("%% error: something wrong.\n", count, num_rm); fflush(stdout);
        } 
    
   printf("%% candidates:%d\n",  num_rm); fflush(stdout);
   end = magma_sync_wtime( queue ); printf("runtime copy part 1: %.2e\n", end-start);fflush(stdout);
cleanup:
    magma_free_cpu( checkcol );
    magma_free_cpu( checkrow );
    omp_destroy_lock(&(counter));
    return info;
}



/**
    Purpose
    -------
    Screens the new candidates for multiple elements in the same row.
    We allow for at most one new element per row.
    This changes the algorithm, but pays off in performance.
    

    Arguments
    ---------
    
    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced.

    @param[in]
    L_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).
                
    @param[in]
    U_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_select_candidates_m(
    magma_int_t num_rm,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=1, chunksize;
    magmaDoubleComplex element;
    magma_int_t i, count = 0;
    omp_lock_t counter;
    omp_init_lock(&(counter));
    magma_int_t duplicate = 0;
    double thrs = 2.0;
    magma_int_t approx, rm_approx, success = 0;
    const magma_int_t ione = 1;
    const magma_int_t iten = 1;
    real_Double_t start, end;
    magma_int_t el_per_block;
    
    magma_index_t *checkcol;
    magma_index_t *checkrow;
    magma_index_t *globalcount;
    
    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    
    CHECK( magma_index_malloc_cpu( &globalcount, num_threads ));
    CHECK( magma_index_malloc_cpu( &checkcol, L_new->num_rows ));
    CHECK( magma_index_malloc_cpu( &checkrow, L_new->num_rows ));
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->num_rows; i++ ){
        checkcol[i] = 0;
        checkrow[i] = 0;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++ ){
        globalcount[i] = 0;
    }
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){
        L_new->list[i] = 0;
    }
    
     //.// Idea: instead of order-statistics, identify 120th largest element, then sweep over, and copy only if not yet included.
    //.// finally, order statistics on remaining elements
       
       info = magma_zparilut_set_approx_thrs( (int)num_rm*thrs, L_new, 1, &element, queue );
        //printf("threshold:%f LU_new.nnz:%d\n", element, LU_new->nnz);
        //CHECK( magma_zorderstatistics(
        //LU_new->val, LU_new->nnz, (int)(num_rm*thrs),  1, &element, queue ) );
        //printf("done orderstatistics\n");

   
    //###################
    // try a completely different approach: form blocks...
    count = 0;
    

    
     el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    // printf("blocksize:%d\n", el_per_block);
    start = magma_sync_wtime( queue );
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        for( magma_int_t i=0; i<L_new->nnz; i++ ){
            magma_int_t localcount;
            if( MAGMA_Z_ABS(L_new->val[i]) >= MAGMA_Z_ABS(element) ){
                
                magma_int_t row = L_new->rowidx[i];
                if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block) ){
                    if( checkrow[row] == 0 ){
                        magma_int_t col = L_new->col[i];
                        omp_set_lock(&(counter));
                        if( checkcol[col] == 0 ){
                            localcount = count;
                            count=count+1;
                            if( count >num_rm +10 ){
                                i = L_new->nnz;  
                            }
                            checkcol[col] = 1;
                            omp_unset_lock(&(counter));  
                            checkrow[row] = 1; 
                            //L_new->col[localcount] = col;
                            //L_new->rowidx[localcount] = row; 
                            L_new->list[i] = 1; 
                            //L_new->val[i] = MAGMA_Z_ZERO;
                        } else {
                            omp_unset_lock(&(counter));
                            //L_new->val[i] = MAGMA_Z_ZERO;  
                        }
                      
                    }
                    
                }
            }
        }
    }
   
    
   

      if( count < num_rm ){
             
            while( count < num_rm ){
                printf("%% error: %d < %d -> increase threshold: %.2e\n", count, num_rm, element);
                element = element * MAGMA_Z_MAKE(.5,0.0);
                if( MAGMA_Z_ABS(element)<1e-15 )
                    exit(-1);
                
                
                /*
                #pragma omp parallel for
                for( magma_int_t i=count; i<L_new->nnz; i++ ){
                    magma_int_t localcount;
                    if( MAGMA_Z_ABS(L_new->val[i]) >= MAGMA_Z_ABS(element) ){
                        magma_int_t col = L_new->col[i];
                        magma_int_t row = L_new->rowidx[i];
                        omp_set_lock(&(counter));
                        if( checkcol[col] == 0 && checkrow[row] == 0){
                            localcount = count;
                            count=count+1;
                            if( count >num_rm +10 ){
                                i = L_new->nnz;  
                            }
                            checkcol[col] = 1;
                            checkrow[row] = 1;
                            omp_unset_lock(&(counter));
                            L_new->col[localcount] = L_new->col[i];
                            L_new->rowidx[localcount] = row; 
                            //magmaDoubleComplex valt = L_new->val[i];
                            L_new->val[i] = MAGMA_Z_ZERO;
                            L_new->list[localcount] = 1;
                            //L_new->val[localcount] = valt;
                        } 
                        else {
                            omp_unset_lock(&(counter));
                            L_new->val[i] = MAGMA_Z_ZERO;
                        }
                        
                    }
                }
                printf("%% now: %d > %d\n", count, num_rm);
                */
                #pragma omp parallel
                {
                    magma_int_t id = omp_get_thread_num();
                    for( magma_int_t i=0; i<L_new->nnz; i++ ){
                        magma_int_t localcount;
                        if( MAGMA_Z_ABS(L_new->val[i]) >= MAGMA_Z_ABS(element) ){
                            
                            magma_int_t row = L_new->rowidx[i];
                            if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block) ){
                                if( checkrow[row] == 0 ){
                                    magma_int_t col = L_new->col[i];
                                    omp_set_lock(&(counter));
                                    if( checkcol[col] == 0 ){
                                        localcount = count;
                                        count=count+1;
                                        if( count >num_rm +10 ){
                                            i = L_new->nnz;  
                                        }
                                        checkcol[col] = 1;
                                        omp_unset_lock(&(counter));  
                                        checkrow[row] = 1; 
                                        //L_new->col[localcount] = col;
                                        //L_new->rowidx[localcount] = row; 
                                        L_new->list[i] = 1; 
                                        //L_new->val[i] = MAGMA_Z_ZERO;
                                    } else {
                                        omp_unset_lock(&(counter));
                                        //L_new->val[i] = MAGMA_Z_ZERO;  
                                    }
                                  
                                }
                                
                            }
                        }
                    }
                }
            }
             printf("%% finally: %d > %d\n", count, num_rm);
          //exit(-1);
        //} else if ( count > num_rm*thrs ) {
        //  printf("%% error: something wrong.\n", count, num_rm); fflush(stdout);
        } 
    
   printf("%% candidates:%d\n",  num_rm); fflush(stdout);
    
   end = magma_sync_wtime( queue ); printf("runtime copy part 1: %.2e\n", end-start);fflush(stdout);
   
cleanup:
    magma_free_cpu( checkcol );
    magma_free_cpu( checkrow );
    omp_destroy_lock(&(counter));
    return info;
}



/**
    Purpose
    -------
    Screens the new candidates for multiple elements in the same row.
    We allow for at most one new element per row.
    This changes the algorithm, but pays off in performance.
    

    Arguments
    ---------
    
    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    rm_loc      magma_int_t*
                Number of Elements that are replaced by distinct threads.

    @param[in]
    L_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).
                
    @param[in]
    U_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_select_candidates(
    magma_int_t *num_rm,
    magma_int_t *rm_loc,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=1, chunksize;
    magmaDoubleComplex element;
    magma_int_t i, count = 0;
    double thrs = 1.01;
    magma_int_t approx, rm_approx, success = 0;
    const magma_int_t ione = 1;
    const magma_int_t iten = 1;
    real_Double_t start, start2, end2, end;
    magma_int_t el_per_block;
    magma_int_t loops =0;
    magma_int_t add_bound = 0;
    double bound1;
    double bound2;
    
    magma_index_t *bound=NULL, *checkcol=NULL;
    start = magma_sync_wtime( queue );
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }

    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    //start = magma_sync_wtime( queue );

    CHECK( magma_index_malloc_cpu( &bound, num_threads+1 ));
    //end = magma_sync_wtime( queue );//printf("alloc: %.2e\n", end-start);fflush(stdout);
     //start = magma_sync_wtime( queue );
     
#ifndef AVOID_DUPLICATES 
    CHECK( magma_index_malloc_cpu( &checkcol, L_new->num_rows ));
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->num_rows; i++ ){
        checkcol[i] = 0;
    }
#endif   
    
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads+1; i++ ){
        rm_loc[i] = 0;
        bound[i] = (*num_rm); 
        //printf("%%bound[%d]: %d \n", i, bound[i]);
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){
        L_new->list[i] = -5;
    }
    //end = magma_sync_wtime( queue ); //printf("stuff part 1: %.2e\n", end-start);fflush(stdout);
    
    //start = magma_sync_wtime( queue );
    info = magma_zparilut_set_approx_thrs( (int)(*num_rm)*thrs, L_new, 1, &element, queue );
    //end = magma_sync_wtime( queue ); //printf("threshold: %.2e\n", end-start);fflush(stdout);
    count = 0;
         
    
    
    
    
    //start2 = magma_sync_wtime( queue );
    while( count < (int)((double)*num_rm*.95) ){
        bound1 = MAGMA_Z_ABS(element);
        bound2 = MAGMA_Z_ABS(element*MAGMA_Z_MAKE(.9,0.0));
        //bound2 = MAGMA_Z_ABS(element*MAGMA_Z_MAKE(.8,0.0));
        count = 0;
         //printf("threshold:%.2e\n", MAGMA_Z_ABS(element));fflush(stdout);
         if(MAGMA_Z_ABS(element)<1e-15){
            (*num_rm) = count;   
         }
        //start = magma_sync_wtime( queue );
      
#ifdef AVOID_DUPLICATES
        
        #pragma omp parallel
        {
            magma_int_t id = omp_get_thread_num();
            magma_int_t lbound = L_new->row[(id+1)*el_per_block];
            if( id == num_threads-1 ){
                lbound = L_new->nnz;    
            }
            for( magma_int_t i=L_new->row[(id)*el_per_block]; i<lbound; i++ ){
                    magma_int_t col = L_new->rowidx[i];
                    double val = MAGMA_Z_ABS(L_new->val[i]);
                    if( val >= bound1 ){
                            L_new->list[i] = -1;
                    } else if ( val >= bound2 ){
                            L_new->list[i] = -2;
                    } 
            }
        }
        
        //end = magma_sync_wtime( queue ); //printf("elementcomparison: %.2e\n", end-start);fflush(stdout);
        //start = magma_sync_wtime( queue );
        #pragma omp parallel
        {
            magma_int_t id = omp_get_thread_num();
            magma_int_t loc_bound = bound[id];
            magma_int_t rm_locc=0;
            magma_int_t lbound = L_new->row[(id+1)*el_per_block];
            if( id == num_threads-1 ){
                lbound = L_new->nnz;    
            }
            for( magma_int_t i=L_new->row[(id)*el_per_block]; i<lbound; i++ ){
                    if( L_new->list[i] == -1 ){
                    L_new->list[i] = id;
                    rm_locc++;
                    if( rm_locc > loc_bound ) {
                        i=lbound;
                    }
                }
            }
            rm_loc[id+1] = rm_locc;
        }
        //end = magma_sync_wtime( queue ); //printf("mark part 1: %.2e\n", end-start);fflush(stdout);
        for(int j=0; j<num_threads+1; j++){
            count = count + rm_loc[j];
        }      
        //printf("elements: %d  %d allow for additional %d\n", count, *num_rm, (int)((*num_rm)-count)/num_threads );
        add_bound = (int)((*num_rm)-count)/(max(1,num_threads/2));
        //start = magma_sync_wtime( queue );
        loops = 0;
        while( count < (int)((double)*num_rm*.95) ){
            loops++;
            printf("%% add loop %d: elements: %d  %d allow for additional %d out of %d\n", loops, count, *num_rm, add_bound, L_new->nnz );
            #pragma omp parallel
            {
                magma_int_t id = omp_get_thread_num();
                magma_int_t loc_bound = bound[id];
                magma_int_t rm_locc=0;
                magma_int_t lbound = L_new->row[(id+1)*el_per_block];
                if( id == num_threads-1 ){
                    lbound = L_new->nnz;    
                }
                for( magma_int_t i=L_new->row[(id)*el_per_block]; i<lbound; i++ ){ 
                        if( L_new->list[i] == -2 ){
                        L_new->list[i] = id;
                        rm_locc++;
                        if( rm_locc >= add_bound ) {
                            i=lbound;
                        }
                    }
                }
                rm_loc[id+1] = rm_loc[id+1] + rm_locc;
                //printf("tid %d added elements: %d < %d\n", id, rm_locc, add_bound);
            }
            count = 0;
            for(int j=0; j<num_threads+1; j++){
                count = count + rm_loc[j];
            }    
            //add_bound = add_bound + (int)((*num_rm)-count)*2;
            if(loops > 3 ){
                break;
            }
        }
        
        //end = magma_sync_wtime( queue ); //printf("mark part 2: %.2e\n", end-start);fflush(stdout);
        element = element * MAGMA_Z_MAKE(.7,0.0);
        //printf("elements: %d  %d\n", count, *num_rm);
        
#else
    
        #pragma omp parallel
        {
            magma_int_t id = omp_get_thread_num();
            for( magma_int_t i=0; i<L_new->nnz; i++ ){
                magma_int_t col = L_new->rowidx[i];
                if( (col < (id+1)*el_per_block) && (col >=(id)*el_per_block) ){
                //if( col%num_threads == id ){
                    double val = MAGMA_Z_ABS(L_new->val[i]);
                    if( val >= bound1 ){
                        if( checkcol[col] == 0 ){
                            checkcol[col] = 1;
                            L_new->list[i] = -1;
                        }
                    } else if ( val >= bound2 ){
                        if( checkcol[col] == 0 ){
                            checkcol[col] = 1;
                            L_new->list[i] = -2;
                        }
                    }
                }
            }
        }
        
        //end = magma_sync_wtime( queue ); //printf("elementcomparison: %.2e\n", end-start);fflush(stdout);
        //start = magma_sync_wtime( queue );
        #pragma omp parallel
        {
            magma_int_t id = omp_get_thread_num();
            magma_int_t loc_bound = bound[id];
            magma_int_t rm_locc=0;
            
            for( magma_int_t i=0; i<L_new->nnz; i++ ){
                if( L_new->list[i] == -1 ){
                    magma_int_t row = L_new->rowidx[i];
                    if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block) ){
                    //if( row%num_threads == id ){
                        L_new->val[i] = MAGMA_Z_ZERO;
                        L_new->list[i] = id;
                        rm_locc++;
                        if( rm_locc > loc_bound ) {
                            i=L_new->nnz;
                        }
                    }
                }
            }
            rm_loc[id+1] = rm_locc;
        }
        //end = magma_sync_wtime( queue ); //printf("mark part 1: %.2e\n", end-start);fflush(stdout);
        for(int j=0; j<num_threads+1; j++){
            count = count + rm_loc[j];
        }      
        #pragma omp parallel
        for(int j=0; j<num_threads; j++){
            bound[j] = rm_loc[j+1] + ((*num_rm)-count); 
        }
        //start = magma_sync_wtime( queue );
        if( count < *num_rm) {
            #pragma omp parallel
            {
                magma_int_t id = omp_get_thread_num();
                magma_int_t loc_bound = bound[id];
                magma_int_t rm_locc=rm_loc[id+1];
                for( magma_int_t i=0; i<L_new->nnz; i++ ){
                    if( L_new->list[i] == -2 ){
                        magma_int_t row = L_new->rowidx[i];
                        if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block) ){
                        //if( row%num_threads == id ){
                            L_new->val[i] = MAGMA_Z_ZERO;
                            L_new->list[i] = id;
                            rm_locc++;
                            if( rm_locc > loc_bound ) {
                                i=L_new->nnz;
                            }
                        }
                    }
                }
                rm_loc[id+1] = rm_locc;
            }
            
            for(int j=0; j<num_threads+1; j++){
                count = count + rm_loc[j];
            }    
            #pragma omp parallel
            for(int j=0; j<num_threads; j++){
                bound[j] = rm_loc[j+1] + ((*num_rm)-count); 
            }
        }
        
        //end = magma_sync_wtime( queue ); //printf("mark part 2: %.2e\n", end-start);fflush(stdout);
        element = element * MAGMA_Z_MAKE(.5,0.0);
#endif
    }
    //end2 = magma_sync_wtime( queue ); //printf("all loop: %.2e\n", end2-start2);fflush(stdout);
    count = 0;
    for(int j=0; j<num_threads+1; j++){
            if(rm_loc[j]<0){
                rm_loc[j] = 0;   
            }
            count = count + rm_loc[j];
            rm_loc[j] = count;
            //printf("rm_loc[%d]:%d\n", j,rm_loc[j]);
    }
    *num_rm = count;
     //printf("loops:%d\n", loops);
   
   
cleanup:
    //magma_free_cpu( checkcol );
    magma_free_cpu( bound );
    magma_free_cpu( checkcol );
    return info;
}


/**
    Purpose
    -------
    Screens the new candidates for multiple elements in the same row.
    We allow for at most one new element per row.
    This changes the algorithm, but pays off in performance.
    

    Arguments
    ---------
    
    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    rm_loc      magma_int_t*
                Number of Elements that are replaced by distinct threads.

    @param[in]
    L_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).
                
    @param[in]
    U_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_select_candidates_L(
    magma_int_t *num_rm,
    magma_int_t *rm_loc,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=1, chunksize;
    magmaDoubleComplex element1;
    magmaDoubleComplex element2;
    magma_int_t i, count = 0;
    double thrs1 = 1.00;//25;
    double thrs2 = 1.05;
    magma_int_t approx, rm_approx, success = 0;
    const magma_int_t ione = 1;
    const magma_int_t iten = 1;
    real_Double_t start, start2, end2, end;
    magma_int_t el_per_block;
    magma_int_t cand_per_block;
    magma_int_t loops =0;
    magma_int_t add_bound = 0;
    double bound1;
    double bound2;
    magma_int_t el;
    
    magma_index_t *bound=NULL, *checkcol=NULL;
    magma_index_t *firstelement=NULL, *lastelement=NULL;
    start = magma_sync_wtime( queue );
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }

    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    cand_per_block = magma_ceildiv( L_new->nnz, num_threads );
    //start = magma_sync_wtime( queue );

    CHECK( magma_index_malloc_cpu( &bound, (num_threads)*(num_threads) ));
    CHECK( magma_index_malloc_cpu( &firstelement, (num_threads)*(num_threads) ));
    CHECK( magma_index_malloc_cpu( &lastelement, (num_threads)*(num_threads) ));
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<(num_threads)*(num_threads); i++ ){
        bound[i] = 0;
        firstelement[i] = -1;
        lastelement[i] = -1;
    }
    rm_loc[0] = 0;
    
    //start = magma_sync_wtime( queue );
    info = magma_zparilut_set_approx_thrs( (int)(*num_rm)*thrs1, L_new, 1, &element1, queue );
    //info = magma_zparilut_set_approx_thrs( (int)(*num_rm)*thrs2, L_new, 1, &element2, queue );
    //end = magma_sync_wtime( queue ); //printf("threshold: %.2e\n", end-start);fflush(stdout);
    count = 0;
         
    bound1 = MAGMA_Z_ABS(element1);
    //bound2 = MAGMA_Z_ABS(element2);
    //#pragma omp parallel
    //{
      //  magma_int_t id = omp_get_thread_num();
        //for( magma_int_t r=L_new->row[id*el_per_block]; r<L_new->row[(id+1)*el_per_block]; r++ ){ 
    // blocked
    
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t* first_loc;
        magma_int_t* last_loc;
        magma_int_t* count_loc;
        magma_index_malloc_cpu( &first_loc, (num_threads) );
        magma_index_malloc_cpu( &last_loc, (num_threads) );
        magma_index_malloc_cpu( &count_loc, (num_threads) );
        for(int z=0;z<num_threads;z++){
            first_loc[z] = -1;
            last_loc[z] = -1;
            count_loc[z] = 0;
        }
        magma_int_t lbound = id*cand_per_block;
        magma_int_t ubound = min( (id+1)*cand_per_block, L_new->nnz);
        for( magma_int_t i=lbound; i<ubound; i++ ){ 
            double val = MAGMA_Z_ABS(L_new->val[i]);

            if( val >= bound1 ){
                L_new->list[i] = -5;
                int tid = L_new->rowidx[i]/el_per_block;
                if( first_loc[tid] == -1 ){
                    first_loc[tid] = i;
                    last_loc[tid] = i;
                } else {
                   L_new->list[ last_loc[tid] ] = i;  
                   last_loc[tid] = i;
                }
                count_loc[ tid ]++;
            }
        }
        for(int z=0;z<num_threads;z++){
            firstelement[z+(id*num_threads)] = first_loc[z];
            lastelement[z+(id*num_threads)] = last_loc[z];
            bound[ z+(id*num_threads) ] = count_loc[z];
        }
        magma_free_cpu( first_loc );    
        magma_free_cpu( last_loc );
        magma_free_cpu( count_loc );
    }
    /*
    
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        for( magma_int_t i=id*cand_per_block; i<(id+1)*cand_per_block; i++ ){ 
            double val = MAGMA_Z_ABS(L_new->val[i]);
            if( val >= bound1 ){
                int tid = L_new->rowidx[i]/el_per_block;
                if( firstelement[tid+(id*num_threads)] == -1 ){
                    firstelement[tid+(id*num_threads)] = i;
                    lastelement[tid+(id*num_threads)] = i;
                } else {
                   L_new->list[ lastelement[tid+(id*num_threads)] ] = i;  
                   lastelement[tid+(id*num_threads)] = i;
                }
                bound[ id*num_threads+tid ]++;
            }
        }
    }
    /*
    
    // interleaved
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){ 
        magma_int_t id = omp_get_thread_num();
        double val = MAGMA_Z_ABS(L_new->val[i]);
        if( val >= bound1 ){
            int tid = L_new->rowidx[i]/el_per_block;
            if( firstelement[tid+(id*num_threads)] == -1 ){
                firstelement[tid+(id*num_threads)] = i;
                lastelement[tid+(id*num_threads)] = i;
            } else {
               L_new->list[ lastelement[tid+(id*num_threads)] ] = i;  
               lastelement[tid+(id*num_threads)] = i;
            }
            bound[ id*num_threads+tid ]++;
        }
    }
    */
    
    //}
    // count elements
    
    //printf("%% count:%d vs %d threshold %.2e\n", count, *num_rm, element1  );
    count = 0;
    //rm_loc[0] = 0;
    #pragma omp parallel for
    for(int j=0; j<num_threads; j++){
        for(int z=1;z<num_threads;z++){
            bound[j] += bound[j+z*num_threads];
        }
    }
    
    
    for(int j=0; j<num_threads; j++){
            count = count + bound[j];
            rm_loc[j+1] = count;
            //printf("rm_loc[%d]:%d\n", j,rm_loc[j]);
    }
    *num_rm=count;
    //printf("rm_loc[%d]:%d\n", num_threads,rm_loc[num_threads]);
    /*
    printf("%% count:%d vs %d threshold %.2e\n", count, *num_rm, element1  );
    *num_rm = count;
        for( magma_int_t i=0; i<num_threads; i++){
        for( magma_int_t j=0; j<num_threads; j++){
        printf("%d -> %d \t",firstelement[i+j*num_threads],lastelement[i+j*num_threads]);  
        }printf("\n");
    }
    printf("\n");*/
    //now combine the linked lists...
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++){
        for( magma_int_t j=num_threads-1; j>0; j--){
            if( ( firstelement[ i+(j*num_threads) ] > -1 ) &&
                ( lastelement[ i+((j-1)*num_threads) ] > -1 ) ){   // connect
                    L_new->list[ lastelement[ i+((j-1)*num_threads) ] ] 
                        = firstelement[ i+(j*num_threads) ];
                //lastelement[ i+((j-1)*num_threads) ] = lastelement[ i+(j*num_threads) ];
            } else if( firstelement[ i+(j*num_threads) ] > -1 ) { 
                firstelement[ i+((j-1)*num_threads) ] = firstelement[ i+(j*num_threads) ];
                lastelement[ i+((j-1)*num_threads) ] = lastelement[ i+(j*num_threads) ];
            } 
        }
    }
    // use the rowpointer to start the linked list
    #pragma omp parallel
    for( magma_int_t i=0; i<num_threads; i++){
        L_new->row[i] = firstelement[i];    
    }
    
    /*
    for( magma_int_t i=0; i<num_threads; i++){
        for( magma_int_t j=0; j<num_threads; j++){
        printf("%d -> %d \t",firstelement[i+j*num_threads],lastelement[i+j*num_threads]);  
        }printf("\n");
    }
    
    
    for(magma_int_t z=0;z<num_threads;z++){
    el = L_new->row[z];
    while(el>-1){
        printf("%d(%d,%d)->",el,L_new->rowidx[el],L_new->col[el]);
        el = L_new->list[el];   
    }printf("\n");
    }
    
    */
    
    //printf("%% count:%d vs %d\n", count, *num_rm  );
    /*
    if( count < (int)((double)*num_rm*.95) ){
        printf("%% add loop\n" );
        #pragma omp parallel
        {
            magma_int_t id = omp_get_thread_num();
            magma_int_t start = L_new->nnz/cand_per_block*(id);
            magma_int_t end = L_new->nnz/cand_per_block*(id);
            if( id == num_threads-1 ){
                end = L_new->nnz;    
            }
            magma_int_t num_locrm = 0;
            magma_int_t loc_bound = (*num_rm)-count;
            for( magma_int_t i=start; i<end; i++ ){
                if( L_new->list[i] == -2 ){
                    L_new->list[i] = (L_new->rowidx[i]/el_per_block);
                    num_locrm++;
                    if( num_locrm > loc_bound ){
                        i=end;
                    }
                }
            }
            rm_loc[id] = rm_loc[id] + num_locrm;
        }
    }*/
    /*
   for(int i=0; i<L_new->nnz; i++){
       if( L_new->list[i] > 0 ){
        printf("element L (%d,%d) %.2e -> %d\n", L_new->rowidx[i],  L_new->col[i],  L_new->val[i],  L_new->list[i] );
       }
    }  */
    /*
    // sanity check
       for(int i=0; i<num_threads; i++){
           
           int lbound = el_per_block*i;
           int ubound = el_per_block*(i+1);
           int el = L_new->row[i];
           while(el>-1){
               if( L_new->rowidx[el]<lbound || L_new->rowidx[el]>ubound-1 ){
                   printf("element (%d,%d) outside bound %d--%d for L in block %d\n", L_new->rowidx[i],  L_new->col[i],  lbound, ubound, i);
                   exit(-1);
               }
               el = L_new->list[el];
           }
       }
    
   */
cleanup:
    //magma_free_cpu( checkcol );
    magma_free_cpu( bound );
    magma_free_cpu( firstelement );
    magma_free_cpu( lastelement );
    return info;
}

/**
    Purpose
    -------
    Screens the new candidates for multiple elements in the same row.
    We allow for at most one new element per row.
    This changes the algorithm, but pays off in performance.
    

    Arguments
    ---------
    
    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    rm_loc      magma_int_t*
                Number of Elements that are replaced by distinct threads.

    @param[in]
    L_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).
                
    @param[in]
    U_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmpilut_select_candidates_U(
    magma_int_t *num_rm,
    magma_int_t *rm_loc,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=1, chunksize;
    magmaDoubleComplex element1;
    magmaDoubleComplex element2;
    magma_int_t i, count = 0;
    double thrs1 = 1.00;//25;
    double thrs2 = 1.05;
    magma_int_t approx, rm_approx, success = 0;
    const magma_int_t ione = 1;
    const magma_int_t iten = 1;
    real_Double_t start, start2, end2, end;
    magma_int_t el_per_block;
    magma_int_t cand_per_block;
    magma_int_t loops =0;
    magma_int_t add_bound = 0;
    double bound1;
    double bound2;
    magma_int_t el;
    
    magma_index_t *bound=NULL, *checkcol=NULL;
    magma_index_t *firstelement=NULL, *lastelement=NULL;
    start = magma_sync_wtime( queue );
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }

    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    cand_per_block = magma_ceildiv( L_new->nnz, num_threads );
    //start = magma_sync_wtime( queue );

    CHECK( magma_index_malloc_cpu( &bound, (num_threads)*(num_threads) ));
    CHECK( magma_index_malloc_cpu( &firstelement, (num_threads)*(num_threads) ));
    CHECK( magma_index_malloc_cpu( &lastelement, (num_threads)*(num_threads) ));
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<(num_threads)*(num_threads); i++ ){
        bound[i] = 0;
        firstelement[i] = -1;
        lastelement[i] = -1;
    }
    rm_loc[0] = 0;
    
    
    //start = magma_sync_wtime( queue );
    info = magma_zparilut_set_approx_thrs( (int)(*num_rm)*thrs1, L_new, 1, &element1, queue );
    //info = magma_zparilut_set_approx_thrs( (int)(*num_rm)*thrs2, L_new, 1, &element2, queue );
    //end = magma_sync_wtime( queue ); //printf("threshold: %.2e\n", end-start);fflush(stdout);
    count = 0;
         
    bound1 = MAGMA_Z_ABS(element1);
    //bound2 = MAGMA_Z_ABS(element2);
    //#pragma omp parallel
    //{
      //  magma_int_t id = omp_get_thread_num();
        //for( magma_int_t r=L_new->row[id*el_per_block]; r<L_new->row[(id+1)*el_per_block]; r++ ){ 
    
    // blocked
    
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t* first_loc;
        magma_int_t* last_loc;
        magma_int_t* count_loc;
        magma_index_malloc_cpu( &first_loc, (num_threads) );
        magma_index_malloc_cpu( &last_loc, (num_threads) );
        magma_index_malloc_cpu( &count_loc, (num_threads) );

        for(int z=0;z<num_threads;z++){
            first_loc[z] = -1;
            last_loc[z] = -1;
            count_loc[z] = 0;
        }
        magma_int_t lbound = id*cand_per_block;
        magma_int_t ubound = min( (id+1)*cand_per_block, L_new->nnz);
        for( magma_int_t i=lbound; i<ubound; i++ ){ 
            double val = MAGMA_Z_ABS(L_new->val[i]);

            if( val >= bound1 ){
                L_new->list[i] = -5;
                int tid = L_new->col[i]/el_per_block;
                //if( tid == 4 && L_new->col[i] == 4 )
                 //   printf("element (%d,%d) at location %d going into block %d\n", L_new->rowidx[i], L_new->col[i], i, tid);
                if( first_loc[tid] == -1 ){
                    first_loc[tid] = i;
                    last_loc[tid] = i;
                } else {
                   L_new->list[ last_loc[tid] ] = i;  
                   last_loc[tid] = i;
                }
                count_loc[ tid ]++;
            }
        }
        for(int z=0;z<num_threads;z++){
            firstelement[z+(id*num_threads)] = first_loc[z];
            lastelement[z+(id*num_threads)] = last_loc[z];
            bound[ z+(id*num_threads) ] = count_loc[z];
        }
        magma_free_cpu( first_loc );    
        magma_free_cpu( last_loc );
        magma_free_cpu( count_loc );
    }
    
    
    // interleaved
        /*
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){ 
        magma_int_t id = omp_get_thread_num();
        double val = MAGMA_Z_ABS(L_new->val[i]);
        if( val >= bound1 ){
            int tid = L_new->col[i]/el_per_block;
            if( firstelement[tid+(id*num_threads)] == -1 ){
                firstelement[tid+(id*num_threads)] = i;
                lastelement[tid+(id*num_threads)] = i;
            } else {
               L_new->list[ lastelement[tid+(id*num_threads)] ] = i;  
               lastelement[tid+(id*num_threads)] = i;
            }
            bound[ id*num_threads+tid ]++;
        }
    }
    */
    
    //}
    // count elements
    
    //printf("%% count:%d vs %d threshold %.2e\n", count, *num_rm, element1  );
    count = 0;
    //rm_loc[0] = 0;
    #pragma omp parallel for
    for(int j=0; j<num_threads; j++){
        for(int z=1;z<num_threads;z++){
            bound[j] += bound[j+z*num_threads];
        }
    }
    
    
    for(int j=0; j<num_threads; j++){
            count = count + bound[j];
            rm_loc[j+1] = count;
            //printf("rm_loc[%d]:%d\n", j,rm_loc[j]);
    }
    *num_rm=count;
    
    
    /*    
    printf("%% count:%d vs %d threshold %.2e\n", count, *num_rm, element1  );
    *num_rm = count;
        for( magma_int_t i=0; i<num_threads; i++){
        for( magma_int_t j=0; j<num_threads; j++){
        printf("%d -> %d \t",firstelement[i+j*num_threads],lastelement[i+j*num_threads]);  
        }printf("\n");
    }
    printf("\n");
    
            // sanity check
    for(int j=0;j<num_threads;j++){
        for(int i=0; i<num_threads; i++){
           
           int lbound = el_per_block*i;
           int ubound = el_per_block*(i+1);
           int el = firstelement[i+j*num_threads];
           int lc = 0;
           while(el>-1){
               //printf("%d -> %d \t",el, L_new->list[el]); 
               if( L_new->col[el]<lbound || L_new->col[el]>ubound-1 ){
                   printf("\t%d -> %d \t",firstelement[i+j*num_threads],lastelement[i+j*num_threads]);  
                   printf("%% before combining: element (%d,%d) at %d outside bound %d--%d for U in block %d threadidx %d count %d\n", L_new->rowidx[i],  L_new->col[i],  el, lbound, ubound, i, j, lc);
                   exit(-1);
               }
               el = L_new->list[el];
               lc++;
           }//printf("\n");
       }
    }
       
       
       
    //printf("rm_loc[%d]:%d\n", num_threads,rm_loc[num_threads]);
    /*
    printf("%% count:%d vs %d threshold %.2e\n", count, *num_rm, element1  );
    *num_rm = count;
        for( magma_int_t i=0; i<num_threads; i++){
        for( magma_int_t j=0; j<num_threads; j++){
        printf("%d -> %d \t",firstelement[i+j*num_threads],lastelement[i+j*num_threads]);  
        }printf("\n");
    }
    printf("\n");*/
    //now combine the linked lists...
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++){
        for( magma_int_t j=num_threads-1; j>0; j--){
            if( ( firstelement[ i+(j*num_threads) ] > -1 ) &&
                ( lastelement[ i+((j-1)*num_threads) ] > -1 ) ){   // connect
                    L_new->list[ lastelement[ i+((j-1)*num_threads) ] ] 
                        = firstelement[ i+(j*num_threads) ];
                //lastelement[ i+((j-1)*num_threads) ] = lastelement[ i+(j*num_threads) ];
            } else if( firstelement[ i+(j*num_threads) ] > -1 ) { 
                firstelement[ i+((j-1)*num_threads) ] = firstelement[ i+(j*num_threads) ];
                lastelement[ i+((j-1)*num_threads) ] = lastelement[ i+(j*num_threads) ];
            } 
        }
    }
    // use the rowpointer to start the linked list
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++){
        L_new->row[i] = firstelement[i];    
    }
    
    /*
    for( magma_int_t i=0; i<num_threads; i++){
        for( magma_int_t j=0; j<num_threads; j++){
        printf("%d -> %d \t",firstelement[i+j*num_threads],lastelement[i+j*num_threads]);  
        }printf("\n");
    }
    
    
    for(magma_int_t z=0;z<num_threads;z++){
    el = L_new->row[z];
    while(el>-1){
        printf("%d(%d,%d)->",el,L_new->rowidx[el],L_new->col[el]);
        el = L_new->list[el];   
    }printf("\n");
    }
    */
    
    
    //printf("%% count:%d vs %d\n", count, *num_rm  );
    /*
    if( count < (int)((double)*num_rm*.95) ){
        printf("%% add loop\n" );
        #pragma omp parallel
        {
            magma_int_t id = omp_get_thread_num();
            magma_int_t start = L_new->nnz/cand_per_block*(id);
            magma_int_t end = L_new->nnz/cand_per_block*(id);
            if( id == num_threads-1 ){
                end = L_new->nnz;    
            }
            magma_int_t num_locrm = 0;
            magma_int_t loc_bound = (*num_rm)-count;
            for( magma_int_t i=start; i<end; i++ ){
                if( L_new->list[i] == -2 ){
                    L_new->list[i] = (L_new->rowidx[i]/el_per_block);
                    num_locrm++;
                    if( num_locrm > loc_bound ){
                        i=end;
                    }
                }
            }
            rm_loc[id] = rm_loc[id] + num_locrm;
        }
    }*/
    /*
   for(int i=0; i<L_new->nnz; i++){
       if( L_new->list[i] > 0 ){
        printf("element L (%d,%d) %.2e -> %d\n", L_new->rowidx[i],  L_new->col[i],  L_new->val[i],  L_new->list[i] );
       }
    }  
   
        // sanity check
       for(int i=0; i<num_threads; i++){
           
           int lbound = el_per_block*i;
           int ubound = el_per_block*(i+1);
           int el = L_new->row[i];
           while(el>-1){
               if( L_new->col[el]<lbound || L_new->col[el]>ubound-1 ){
                   printf("element (%d,%d) outside bound %d--%d for U in block %d\n", L_new->rowidx[i],  L_new->col[i],  lbound, ubound, i);
                   exit(-1);
               }
               el = L_new->list[el];
           }
       }*/
    
cleanup:
    //magma_free_cpu( checkcol );
    magma_free_cpu( bound );
    magma_free_cpu( firstelement );
    magma_free_cpu( lastelement );
    return info;
}




#endif  // _OPENMP
