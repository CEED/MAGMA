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
magma_zmpilut_sweep(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n");fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L->nnz; e++)
    {
        // as we look at the lower triangular, 
        // col<row, i.e. disregard last element in row
        magma_int_t i,j,icol,jcol,jold;
        
        magma_index_t row = L->rowidx[ e ];
        magma_index_t col = L->col[ e ];
            
        if( L->row[row+1]-1==e ){ // end check whether part of L
            L->val[ e ] = MAGMA_Z_ONE; // lower triangular has diagonal equal 1
        } else{
            
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
        }
        
    }// end omp parallel section
    
   #pragma omp parallel for
    for( magma_int_t e=0; e<U->nnz; e++){
        // as we look at the lower triangular, 
        // col<row, i.e. disregard last element in row
        //if( U->list[e] != -1 )
        {
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
magma_zmpilut_residuals(
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
magma_zparilut_reorder_o(
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
    
    #pragma omp parallel for
    for( magma_int_t z=0; z < LU->true_nnz; z++ ){
        list[ z ] = -1;
        col[z] = -5;
        rowidx[z] = -7;
    }
    
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
    
    #pragma omp parallel for
    for(magma_int_t rowc=0; rowc<LU->num_rows; rowc++){
        magma_int_t el = LU->row[ rowc ];
        magma_int_t offset = row[ rowc ];
        magma_index_t loc_nnz = 0;
        do{
            magmaDoubleComplex valt = LU->val[ el ];
            if(magma_z_isnan_inf( valt ) ){
                info = MAGMA_ERR_NAN;
                el = 0;
            } else {
                val[ offset+loc_nnz ] = LU->val[ el ];
                col[ offset+loc_nnz ] = LU->col[ el ];
                rowidx[ offset+loc_nnz ] = LU->rowidx[ el ];
                list[ offset+loc_nnz ] = offset+loc_nnz+1;
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
magma_zmpilut_insertsort(
    magma_int_t *num_rmL,
    magma_int_t *num_rmU,
    magma_index_t *addL,
    magma_index_t *addU,
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
    magmaDoubleComplex thrs;
    double bound;
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    el_per_block = magma_ceildiv( U->num_rows, num_threads );
    
    CHECK( magma_index_malloc_cpu( &row, L->num_rows+1 ));
    
    // for L
    CHECK( magma_zparilut_set_approx_thrs( *num_rmL, L_new, 1, &thrs, queue ) );
    
    bound = MAGMA_Z_ABS(thrs);
    // in a first sweep, we count how many elements are inserted in the different rows
    #pragma omp parallel for
    for( magma_int_t i=0; i < L->num_rows; i++ ){
        magma_int_t rowelements=0;
        magma_int_t rowstart = L_new->row[i];
        for( magma_int_t j=L_new->row[i]; j < L_new->row[i+1]; j++ ){
            if( MAGMA_Z_ABS( L_new->val[ j ] ) > bound ){ 
                // no need to care about the values any more - 
                // just flip the new colidx to the beginning
                L_new->col[ rowstart+rowelements] = L_new->col[j];
                rowelements++;
            }
        }
        addL[i] = rowelements;
    }
    
    // create the new rowpointer
    row[0] = 0;
    *num_rmL = 0;
    for( magma_int_t i=0; i < L->num_rows; i++ ){
         *num_rmL = *num_rmL+addL[i];
         row[i+1] = L->row[i+1] + *num_rmL;
    }
    L->nnz = row[L->num_rows];
    L->true_nnz = row[L->num_rows];
    CHECK( magma_zmalloc_cpu( &val, L->true_nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, L->true_nnz ));
    CHECK( magma_index_malloc_cpu( &col, L->true_nnz ));
    
    
    // now sort the new elements in the rows for increasing col index
    #pragma omp parallel for
    for( magma_int_t i=0; i < L->num_rows; i++ ){
        if( addL[i]> 1 ){
            magma_zindexsort( L_new->col, L_new->row[i], L_new->row[i]+addL[i]-1, queue );
        }
    }
    
    // now insert in parallel fashion
    #pragma omp parallel for
    for( magma_int_t i=0; i < L->num_rows; i++ ){
        magma_int_t loc = row[i];
        magma_int_t insert=0, insertnew=0;
        magma_int_t Li = L->row[i];
        magma_int_t Lnewi = L_new->row[i];
        magma_int_t numinsert = L->row[i+1]-L->row[i]+addL[i];
        while( insert+insertnew < numinsert ){
            if( L->col[Li] > L_new->col[Lnewi] && insertnew < addL[i] ){ // insert new element here
                col[loc] = L_new->col[Lnewi];
                val[loc] = MAGMA_Z_ZERO;
                rowidx[loc] = i;  
                Lnewi++;
                insertnew++;
                loc++;
            } else {
                col[loc] = L->col[Li];
                val[loc] = L->val[Li];
                rowidx[loc] = i;
                Li++;
                insert++;
                loc++;
            }
        }
    }
    
    // flip
    rowt = L->row;
    rowidxt = L->rowidx;
    valt = L->val;
    colt = L->col;
    
    L->row = row;
    L->rowidx = rowidx;
    L->val = val;
    L->col = col;
    
    row = rowt;
    rowidx = rowidxt;
    val = valt;
    col = colt;
    
    L->nnz = L->row[L->num_rows];
    L->storage_type = Magma_CSR;
    
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( rowidx );
    val = NULL;
    col = NULL;
    rowidx=NULL;
    
    //##########################################################################
    // now for U
    
    // now the same like for L
    CHECK( magma_zparilut_set_approx_thrs( *num_rmU, U_new, 1, &thrs, queue ) );
    
    bound = MAGMA_Z_ABS(thrs);
    
    // unfortunately, we first need to order this to have the elements in the same row
    // consecutive in memory
    #pragma omp parallel
    for( magma_int_t i=0; i < U_new->num_rows; i++ ){
         addU[i] = 0;
    }
    
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        for(magma_int_t i=0; i<U_new->nnz; i++ ){
            magma_index_t lrow = U_new->col[ i ]; // need to transpose candidates
            if( (lrow < (id+1)*el_per_block) && (lrow >=(id)*el_per_block)  ){
                //printf("thread %d handling row %d\n", id, row );
                if( MAGMA_Z_ABS( U_new->val[ i ] ) < bound ){
                    U_new->col[i] = -1;
                } else{
                    //printf("thread %d handling row %d catch this\n", id, lrow );
                    addU[lrow]++;
                }
            }
        }
    }
    
    // next step: find locations where to insert
    // create the new rowpointer
    U_new->row[0] = 0;
    *num_rmU = 0;
    for( magma_int_t i=0; i < U_new->num_rows; i++ ){
         U_new->row[i+1] = U_new->row[i] + addU[i];
         addU[i] = 0;
    }
    
    // now insert properly
    CHECK( magma_index_malloc_cpu( &rowidx, U_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &col, U_new->true_nnz ));
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        for(magma_int_t i=0; i< U_new->nnz; i++ ){
            magma_index_t lrow = U_new->col[ i ];
            if( (lrow < (id+1)*el_per_block) && (lrow >=(id)*el_per_block)  ){
                rowidx[ U_new->row[lrow]+addU[lrow] ] = lrow;
                col[ U_new->row[lrow]+addU[lrow] ] = U_new->rowidx[i];
                addU[lrow]++;
            }
        }
    }

    
    // flip the structure and free
    rowidxt = U_new->rowidx;
    colt = U_new->col;
    
    U_new->col    = col;
    U_new->rowidx = rowidx;

    col = colt;
    rowidx = rowidxt;
    
    U_new->nnz = U_new->true_nnz;
    
    magma_free_cpu( rowidx );
    magma_free_cpu( col );
    col = NULL;
    rowidx=NULL;

            // now sort the new elements in the rows for increasing col index
    #pragma omp parallel for
    for( magma_int_t i=0; i < U->num_rows; i++ ){
        if( addU[i]> 1 ){
            magma_zindexsort( U_new->col, U_new->row[i], U_new->row[i+1]-1, queue );
        }
    }
    
    
    // create the new rowpointer
    row[0] = 0;
    *num_rmU = 0;
    for( magma_int_t i=0; i < U->num_rows; i++ ){
         *num_rmU = *num_rmU+addU[i];
         row[i+1] = U->row[i+1] + *num_rmU;
    }
    
    U->nnz = row[L->num_rows];
    U->true_nnz = row[L->num_rows];
    CHECK( magma_zmalloc_cpu( &val, U->true_nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, U->true_nnz ));
    CHECK( magma_index_malloc_cpu( &col, U->true_nnz ));

    
    // now insert in parallel fashion
    #pragma omp parallel for
    for( magma_int_t i=0; i < U->num_rows; i++ ){
        magma_int_t loc = row[i];
        magma_int_t insert=0, insertnew=0;
        magma_int_t Li = U->row[i];
        magma_int_t Lnewi = U_new->row[i];
        magma_int_t numinsert = U->row[i+1]-U->row[i]+addU[i];
        while( insert+insertnew < numinsert ){
            if( U->col[Li] > U_new->col[Lnewi] && insertnew < addU[i] ){ // insert new element here
                col[loc] = U_new->col[Lnewi];
                val[loc] = MAGMA_Z_ZERO;
                rowidx[loc] = i;  
                Lnewi++;
                insertnew++;
                loc++;
            } else {
                col[loc] = U->col[Li];
                val[loc] = U->val[Li];
                rowidx[loc] = i;
                Li++;
                insert++;
                loc++;
            }
        }
        //printf("\n\n");
    }
    //exit(-1);
    
    // flip
    rowt = U->row;
    rowidxt = U->rowidx;
    valt = U->val;
    colt = U->col;
    
    U->row = row;
    U->rowidx = rowidx;
    U->val = val;
    U->col = col;
    
    row = rowt;
    rowidx = rowidxt;
    val = valt;
    col = colt;
    
    U->nnz = U->row[U->num_rows];
    U->storage_type = Magma_CSRCOO;
    //for( magma_int_t i=0; i < L->num_rows+1; i++ ){
    //     printf("row[%d]=%d\n", i, U->row[i]);
    //}
    //for( magma_int_t i=0; i < 100; i++ ){
    //     printf("(%d, %d) %.2f\n", U->rowidx[i], U->col[i], U->val[i]);
    //}
    //exit(-1);
    
    
    // free
cleanup:
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( row );
    magma_free_cpu( rowidx );    
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
magma_zmpilut_rmsort(
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
    
    magma_index_t *new_row=NULL, *new_rowidx=NULL, *new_colidx=NULL;
    magmaDoubleComplex *new_val=NULL;
    magma_index_t *row_t=NULL, *rowidx_t=NULL, *colidx_t=NULL;
    magmaDoubleComplex *val_t=NULL;
    
    double bound = MAGMA_Z_ABS(*thrs);
    //bound = 0.1;
    
    CHECK( magma_index_malloc_cpu( &new_row, LU_new->num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &new_rowidx, LU_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &new_colidx, LU_new->true_nnz ));    
    CHECK( magma_zmalloc_cpu( &new_val, LU_new->true_nnz ));
    
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    el_per_block = magma_ceildiv( LU->num_rows, num_threads );
    
    // first run: count for the new row-pointer
    #pragma omp parallel for
    for( magma_int_t i=0; i<LU->num_rows; i++ ){
        magma_int_t rowelements=0;
        for( magma_int_t j=LU->row[i]; j<LU->row[i+1]; j++ ){
            if( MAGMA_Z_ABS( LU->val[ j ] ) > bound ){ 
                rowelements++;
            } else if( LU->col[j] == i ) { //make sure diagonal is included
                rowelements++;
            }
        }
        new_row[i+1] = rowelements;
    }
    
    // get the total element count
    LU->nnz = 0;
    new_row[ 0 ] = 0;
    for( magma_int_t i = 0; i<LU->num_rows; i++ ){
        LU->nnz = LU->nnz + new_row[ i+1 ];
        new_row[ i+1 ] = LU->nnz;
    }
    LU->true_nnz = LU->nnz;
    // second run: compress the matrix into the new structure
    #pragma omp parallel for
    for( magma_int_t i=0; i<LU->num_rows; i++ ){
        magma_int_t insertlocation = new_row[i];
        for( magma_int_t j=LU->row[i]; j<LU->row[i+1]; j++ ){
            if( MAGMA_Z_ABS( LU->val[ j ] ) > bound ){ 
                new_rowidx[insertlocation] = i;
                new_colidx[insertlocation] = LU->col[ j ];
                new_val[insertlocation] = LU->val[ j ];
                insertlocation++;
            } else if( LU->col[j] == i ) { //make sure diagonal is included
                new_rowidx[insertlocation] = i;
                new_colidx[insertlocation] = LU->col[ j ];
                new_val[insertlocation] = LU->val[ j ];
                insertlocation++;
            }
        }
    }
    
    // flip the structure and free
    rowidx_t = LU->rowidx;
    row_t    = LU->row;
    colidx_t = LU->col;
    val_t    = LU->val;
    
    LU->row    = new_row;    
    LU->col    = new_colidx;
    LU->rowidx = new_rowidx;
    LU->val    = new_val;

    new_row    = row_t;
    new_colidx = colidx_t;
    new_rowidx = rowidx_t;
    new_val    = val_t;   
    
cleanup:
    magma_free_cpu( new_row );
    magma_free_cpu( new_rowidx );
    magma_free_cpu( new_colidx );
    magma_free_cpu( new_val );  

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
magma_zmpilut_candlist(
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
                for( magma_index_t checkel=L.row[cand_row]; checkel<L.row[cand_row+1]; checkel++){
                    checkcol = L.col[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=L.row[cand_row+1];
                    }
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numaddrowL++;
                    //numaddL[ row+1 ]++;
                }
                el2 = L.list[ el2 ];
            }
            // #################################################################
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
            
            el1 = L.list[ el1 ];
        }
    
    //##########################################################################
    if( unsym == 1 ){

        start1 = UR.row[ row ];
        el1 = start1;
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
                    for( magma_index_t checkel=L.row[cand_row]; checkel<L.row[cand_row+1]; checkel++){
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=L.row[cand_row+1];
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
                el2 = L.list[ el2 ];
            }
            // #########################################################################
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
         
        numaddU[ row+1 ] = numaddrowU;
        numaddL[ row+1 ] = numaddrowL;
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
    
    // #########################################################################
    
    start = magma_sync_wtime( queue );
    // insertion here
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row];
        magma_int_t offsetU = numaddU[row];
        
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
                for( magma_index_t checkel=L.row[cand_row]; checkel<L.row[cand_row+1]; checkel++){
                    checkcol = L.col[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=L.row[cand_row+1];
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
                el2 = L.list[ el2 ];
            }
            el1 = L.list[el1];
            
        }
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
                    for( magma_index_t checkel=L.row[cand_row]; checkel<L.row[cand_row+1]; checkel++){
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=L.row[cand_row+1];
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
            
            el1 = L.list[ el1 ];
        }
        
    
    //#######################
    if( unsym == 1 ){
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
                    for( magma_index_t checkel=L.row[cand_row]; checkel<L.row[cand_row+1]; checkel++){
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=L.row[cand_row+1];
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
        
        L_new->row[row+1] = laddL;
        U_new->row[row+1] = laddU;
        
    } // unsym
    }   //loop over all rows
    
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



#endif  // _OPENMP
