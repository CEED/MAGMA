/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/


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

extern "C"
magma_int_t read_z_csr_from_binary( magma_int_t* n_row, magma_int_t* n_col, magma_int_t* nnz, magmaDoubleComplex **val, magma_int_t **row, magma_int_t **col, const char * filename ){
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Reads in a matrix stored in coo format from a binary and converts it
    into CSR format. It duplicates the off-diagonal entries in the 
    symmetric case.


    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaDoubleComplex **val             value array of CSR output 
    magma_int_t **row                    row pointer of CSR output
    magma_int_t **col                    column indices of CSR output
    const char * filename                filname of the binary matrix

    =====================================================================  */

  std::fstream binary_test(filename);


  if(binary_test){
    printf("#Start reading...");
    fflush(stdout);
  }
  else{
    printf("#Unable to open file ", filename);
    fflush(stdout);
    exit(1);
  }
  binary_test.close();
  
  
  std::fstream binary_rfile(filename,std::ios::binary|std::ios::in);
  
  
  //read number of rows
  binary_rfile.read(reinterpret_cast<char *>(n_row),sizeof(int));
  
  //read number of columns
  binary_rfile.read(reinterpret_cast<char *>(n_col),sizeof(int));
  
  //read number of nonzeros
  binary_rfile.read(reinterpret_cast<char *>(nnz),sizeof(int));
  
  
  *val = new magmaDoubleComplex[*nnz];
  *col = new int[*nnz];
  *row = new int[*n_row+1];
  
  
  //read row pointer
  for(magma_int_t i=0;i<=*n_row;i++){
    binary_rfile.read(reinterpret_cast<char *>(&(*row)[i]),sizeof(int));
  }
  
  //read col
  for(magma_int_t i=0;i<*nnz;i++){
    binary_rfile.read(reinterpret_cast<char *>(&(*col)[i]),sizeof(int));
  }
  
  //read val
  for(magma_int_t i=0;i<*nnz;i++){
    binary_rfile.read(reinterpret_cast<char *>(&(*val)[i]),sizeof(double));
  }

  binary_rfile.close();
  
  printf("#Finished reading.");
  fflush(stdout);

  return MAGMA_SUCCESS;
}

extern "C"
magma_int_t read_z_csr_from_mtx( magma_storage_t *type, magma_storage_t *location, magma_int_t* n_row, magma_int_t* n_col, magma_int_t* nnz, magmaDoubleComplex **val, magma_int_t **row, magma_int_t **col, const char *filename ){
  
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Reads in a matrix stored in coo format from a Matrix Market (.mtx)
    file and converts it into CSR format. It duplicates the off-diagonal
    entries in the symmetric case.

    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaDoubleComplex **val             value array of CSR output 
    magma_int_t **row                    row pointer of CSR output
    magma_int_t **col                    column indices of CSR output
    const char * filename                filname of the mtx matrix

    =====================================================================  */

  FILE *fid;
  MM_typecode matcode;
    
  fid = fopen(filename, "r");
  
  if (fid == NULL) {
    printf("#Unable to open file %s\n", filename);
    exit(1);
  }
  
  if (mm_read_banner(fid, &matcode) != 0) {
    printf("#Could not process lMatrix Market banner.\n");
    exit(1);
  }
  
  if (!mm_is_valid(matcode)) {
    printf("#Invalid lMatrix Market file.\n");
    exit(1);
  }
  
  if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode))) {
    printf("#Sorry, this application does not support ");
    printf("#Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    printf("#Only sparse real-valued or pattern coordinate matrices are supported\n");
    exit(1);
  }
  
  magma_int_t num_rows, num_cols, num_nonzeros;
  if (mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
    exit(1);
  
  (*type) = Magma_CSR;
  (*location) = Magma_CPU;
  (*n_row) = (magma_int_t) num_rows;
  (*n_col) = (magma_int_t) num_cols;
  (*nnz)   = (magma_int_t) num_nonzeros;

  magma_int_t *coo_col, *coo_row;
  magmaDoubleComplex *coo_val;
  
  coo_col = (magma_int_t *) malloc(*nnz*sizeof(magma_int_t));
  assert(coo_col != NULL);

  coo_row = (magma_int_t *) malloc(*nnz*sizeof(magma_int_t)); 
  assert( coo_row != NULL);

  coo_val = (magmaDoubleComplex *) malloc(*nnz*sizeof(magmaDoubleComplex));
  assert( coo_val != NULL);


  printf("Reading sparse matrix from file (%s):",filename);
  fflush(stdout);


  if (mm_is_real(matcode) || mm_is_integer(matcode)){
    for(magma_int_t i = 0; i < *nnz; ++i){
      magma_int_t ROW ,COL;
      double VAL;  // always read in a double and convert later if necessary
      
      fscanf(fid, " %d %d %lf \n", &ROW, &COL, &VAL);   
      
      coo_row[i] = (magma_int_t) ROW - 1; 
      coo_col[i] = (magma_int_t) COL - 1;
      coo_val[i] = MAGMA_Z_MAKE( VAL, 0.);
    }
  } else {
    printf("Unrecognized data type\n");
    exit(1);
  }
  
  fclose(fid);
  printf(" done\n");
  
  

  if(mm_is_symmetric(matcode)) { //duplicate off diagonal entries
  printf("detected symmetric case\n");
    magma_int_t off_diagonals = 0;
    for(magma_int_t i = 0; i < *nnz; ++i){
      if(coo_row[i] != coo_col[i])
        ++off_diagonals;
    }
    
    magma_int_t true_nonzeros = 2*off_diagonals + (*nnz - off_diagonals);
    
    
    printf("total number of nonzeros: %d\n",*nnz);

    
    
    magma_int_t* new_row = (magma_int_t *) malloc(true_nonzeros*sizeof(magma_int_t)) ; 
    magma_int_t* new_col = (magma_int_t *) malloc(true_nonzeros*sizeof(magma_int_t)) ; 
    magmaDoubleComplex* new_val = (magmaDoubleComplex *) malloc(true_nonzeros*sizeof(magmaDoubleComplex)) ; 
    
    magma_int_t ptr = 0;
    for(magma_int_t i = 0; i < *nnz; ++i) {
        if(coo_row[i] != coo_col[i]) {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
        new_col[ptr] = coo_row[i];  
        new_row[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;  
      } else 
      {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
      }
    }      
    
    free (coo_row);
    free (coo_col);
    free (coo_val);

    coo_row = new_row;  
    coo_col = new_col; 
    coo_val = new_val;   
    
    *nnz = true_nonzeros;
    

  } //end symmetric case
  
  magmaDoubleComplex tv;
  magma_int_t ti;
  
  
  //If matrix is not in standard format, sorting is necessary
  /*
  
    std::cout << "Sorting the cols...." << std::endl;
  // bubble sort (by cols)
  for (int i=0; i<*nnz-1; ++i)
    for (int j=0; j<*nnz-i-1; ++j)
      if (coo_col[j] > coo_col[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }

  std::cout << "Sorting the rows...." << std::endl;
  // bubble sort (by rows)
  for (int i=0; i<*nnz-1; ++i)
    for (int j=0; j<*nnz-i-1; ++j)
      if ( coo_row[j] > coo_row[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }
  std::cout << "Sorting: done" << std::endl;
  
  */
  
  
  (*val) = (magmaDoubleComplex *) malloc(*nnz*sizeof(magmaDoubleComplex)) ;
  assert((*val) != NULL);
  
  (*col) = (magma_int_t *) malloc(*nnz*sizeof(magma_int_t));
  assert((*col) != NULL);
  
  (*row) = (magma_int_t *) malloc((*n_row+1)*sizeof(magma_int_t)) ;
  assert((*row) != NULL);
  

  
  
  // original code from  Nathan Bell and Michael Garland
  // the output CSR structure is NOT sorted!

  for (magma_int_t i = 0; i < num_rows; i++)
    (*row)[i] = 0;
  
  for (magma_int_t i = 0; i < *nnz; i++)
    (*row)[coo_row[i]]++;
  
  
  //cumsum the nnz per row to get Bp[]
  for(magma_int_t i = 0, cumsum = 0; i < num_rows; i++){     
    magma_int_t temp = (*row)[i];
    (*row)[i] = cumsum;
    cumsum += temp;
  }
  (*row)[num_rows] = *nnz;
  
  //write Aj,Ax into Bj,Bx
  for(magma_int_t i = 0; i < *nnz; i++){
    magma_int_t row_  = coo_row[i];
    magma_int_t dest = (*row)[row_];
    
    (*col)[dest] = coo_col[i];
    
    (*val)[dest] = coo_val[i];
    
    (*row)[row_]++;
  }
  
  for(int i = 0, last = 0; i <= num_rows; i++){
    int temp = (*row)[i];
    (*row)[i]  = last;
    last   = temp;
  }
  
  (*row)[*n_row]=*nnz;
     

  for (magma_int_t k=0; k<*n_row; ++k)
    for (magma_int_t i=(*row)[k]; i<(*row)[k+1]-1; ++i) 
      for (magma_int_t j=(*row)[k]; j<(*row)[k+1]-1; ++j) 

      if ( (*col)[j] > (*col)[j+1] ){

        ti = (*col)[j];
        (*col)[j] = (*col)[j+1];
        (*col)[j+1] = ti;

        tv = (*val)[j];
        (*val)[j] = (*val)[j+1];
        (*val)[j+1] = tv;

      }

  return MAGMA_SUCCESS;
}


extern "C"
magma_int_t write_z_csr_mtx( magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, magmaDoubleComplex **val, magma_int_t **row, magma_int_t **col, magma_major_t MajorType, const char *filename ){

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Writes a CSR matrix to a file using Matrix Market format.

    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaDoubleComplex **val             value array of CSR  
    magma_int_t **row                    row pointer of CSR 
    magma_int_t **col                    column indices of CSR 
    magma_int_t MajorType                Row or Column sort
                                         default: 0 = RowMajor, 1 = ColMajor
    const char * filename                output filename for the matrix

    =====================================================================  */

  if( MajorType == MagmaColMajor){
    //to obtain ColMajr output we transpose the matrix 
    //and flip in the output the row and col pointer
    magmaDoubleComplex *new_val;
    magma_int_t *new_row;                    
    magma_int_t *new_col;
    magma_int_t new_n_row;
    magma_int_t new_n_col;
    magma_int_t new_nnz;

    z_transpose_csr( n_row, n_col, nnz, *val, *row, *col, &new_n_row, &new_n_col, &new_nnz, &new_val, &new_row, &new_col);
    printf("Writing sparse matrix to file (%s):",filename);
    fflush(stdout);

    std::ofstream file(filename);
    file << "%%MatrixMarket matrix coordinate real general ColMajor" << std::endl;
    file << new_n_row <<" "<< new_n_col <<" "<< new_nnz << std::endl;
   
    magma_int_t i=0, j=0, rowindex=1;

    for(i=0; i<n_col; i++)
    {    
      magma_int_t rowtemp1=(new_row)[i];
      magma_int_t rowtemp2=(new_row)[i+1];
      for(j=0; j<rowtemp2-rowtemp1; j++)  
        file << ((new_col)[rowtemp1+j]+1) <<" "<< rowindex <<" "<< MAGMA_Z_REAL((new_val)[rowtemp1+j]) << std::endl;
      rowindex++;
    }
    printf(" done\n");

  }
  else{
    printf("Writing sparse matrix to file (%s):",filename);
    fflush(stdout);

    std::ofstream file(filename);
    file << "%%MatrixMarket matrix coordinate real general RowMajor" << std::endl;
    file << n_row <<" "<< n_col <<" "<< nnz << std::endl;
   
    magma_int_t i=0, j=0, rowindex=1;

    for(i=0; i<n_col; i++)
    {
      magma_int_t rowtemp1=(*row)[i];
      magma_int_t rowtemp2=(*row)[i+1];
      for(j=0; j<rowtemp2-rowtemp1; j++)  
        file << rowindex <<" "<< ((*col)[rowtemp1+j]+1) <<" "<< MAGMA_Z_REAL((*val)[rowtemp1+j]) << std::endl;
      rowindex++;
    }
    printf(" done\n");
  }
  return MAGMA_SUCCESS;
}


extern "C"
magma_int_t print_z_csr_mtx( magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, magmaDoubleComplex **val, magma_int_t **row, magma_int_t **col, magma_major_t MajorType ){

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Prints a CSR matrix in Matrix Market format.

    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaDoubleComplex **val             value array of CSR  
    magma_int_t **row                    row pointer of CSR 
    magma_int_t **col                    column indices of CSR
    magma_int_t MajorType                Row or Column sort
                                         default: 0 = RowMajor, 1 = ColMajor

    =====================================================================  */

  if( MajorType == MagmaColMajor ){
    //to obtain ColMajr output we transpose the matrix 
    //and flip in the output the row and col pointer
    magmaDoubleComplex *new_val;
    magma_int_t *new_row;                    
    magma_int_t *new_col;
    magma_int_t new_n_row;
    magma_int_t new_n_col;
    magma_int_t new_nnz;

    z_transpose_csr( n_row, n_col, nnz, *val, *row, *col, &new_n_row, &new_n_col, &new_nnz, &new_val, &new_row, &new_col);

    cout << "%%MatrixMarket matrix coordinate real general ColMajor" << std::endl;
    cout << new_n_row <<" "<< new_n_col <<" "<< new_nnz << std::endl;
   
    magma_int_t i=0, j=0, rowindex=1;

    for(i=0; i<n_col; i++)
    {    
      magma_int_t rowtemp1=(new_row)[i];
      magma_int_t rowtemp2=(new_row)[i+1];
      for(j=0; j<rowtemp2-rowtemp1; j++)  
        cout << ((new_col)[rowtemp1+j]+1) <<" "<< rowindex <<" "<< MAGMA_Z_REAL((new_val)[rowtemp1+j]) << std::endl;
      rowindex++;
    }
  }
  else{
    cout << "%%MatrixMarket matrix coordinate real general RowMajor" << std::endl;
    cout << n_row <<" "<< n_col <<" "<< nnz << std::endl;
   
    magma_int_t i=0, j=0, rowindex=1;

    for(i=0; i<n_col; i++)
    {
      magma_int_t rowtemp1=(*row)[i];
      magma_int_t rowtemp2=(*row)[i+1];
      for(j=0; j<rowtemp2-rowtemp1; j++)  
        cout<< rowindex <<" "<< (*col)[rowtemp1+j]+1 <<" "<< MAGMA_Z_REAL((*val)[rowtemp1+j]) <<endl;
      rowindex++;
    }
  }
  return MAGMA_SUCCESS;
}

extern "C"
magma_int_t print_z_csr( magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, magmaDoubleComplex **val, magma_int_t **row, magma_int_t **col ){

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Prints a CSR matrix in CSR format.

    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaDoubleComplex **val             value array of CSR  
    magma_int_t **row                    row pointer of CSR 
    magma_int_t **col                    column indices of CSR 

    =====================================================================  */

  cout << "Matrix in CSR format (row col val)" << endl;
  cout << n_row <<" "<< n_col <<" "<< nnz <<endl;
   
  magma_int_t i=0,j=0;

  for(i=0; i<n_col; i++)
  {
    magma_int_t rowtemp1=(*row)[i];
    magma_int_t rowtemp2=(*row)[i+1];
    for(j=0; j<rowtemp2-rowtemp1; j++)  
      cout<< (rowtemp1+1) <<" "<< (*col)[rowtemp1+j]+1 <<" "<< MAGMA_Z_REAL((*val)[rowtemp1+j]) <<endl;
  }
  return MAGMA_SUCCESS;
}


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Prints a sparse matrix in CSR format.

    Arguments
    =========

    magma_z_sparse_matrix A              sparse matrix in Magma_CSR format

    =====================================================================  */

extern "C"
magma_int_t magma_z_mvisu( magma_z_sparse_matrix A )
{


    if( A.storage_type != Magma_DENSE ){
        magma_z_sparse_matrix B;
        magma_z_mconvert( A, &B, A.storage_type, Magma_DENSE);
        for( magma_int_t i=0; i<(B.num_rows); i++ ){
          for( magma_int_t j=0; j<B.num_cols; j++ )
            cout << MAGMA_Z_REAL( B.val[i*(B.num_cols)+j] ) << " " ;
          cout << endl;
        }
        free(B.val);
    }
    else{
        for( magma_int_t i=0; i<(A.num_rows); i++ ){
          for( magma_int_t j=0; j<A.num_cols; j++ )
            cout << MAGMA_Z_REAL( A.val[i*(A.num_cols)+j] ) << " " ;
          cout << endl;
        }
    }
  return MAGMA_SUCCESS;
}






















/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Reads in a matrix stored in coo format from a Matrix Market (.mtx)
    file and converts it into CSR format. It duplicates the off-diagonal
    entries in the symmetric case.

    Arguments
    =========

    magma_z_sparse_matrix *A             matrix in magma sparse matrix format
    const char * filename                filname of the mtx matrix

    =====================================================================  */

extern "C"
magma_int_t magma_z_csr_mtx( magma_z_sparse_matrix *A, const char *filename ){

  int csr_compressor = 0;       // checks for zeros in original file

  FILE *fid;
  MM_typecode matcode;
    
  fid = fopen(filename, "r");
  
  if (fid == NULL) {
    printf("#Unable to open file %s\n", filename);
    exit(1);
  }
  
  if (mm_read_banner(fid, &matcode) != 0) {
    printf("#Could not process lMatrix Market banner.\n");
    exit(1);
  }
  
  if (!mm_is_valid(matcode)) {
    printf("#Invalid lMatrix Market file.\n");
    exit(1);
  }
  
  if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode))) {
    printf("#Sorry, this application does not support ");
    printf("#Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    printf("#Only sparse real-valued or pattern coordinate matrices are supported\n");
    exit(1);
  }
  
  magma_int_t num_rows, num_cols, num_nonzeros;
  if (mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
    exit(1);
  
  (A->storage_type) = Magma_CSR;
  (A->memory_location) = Magma_CPU;
  (A->num_rows) = (magma_int_t) num_rows;
  (A->num_cols) = (magma_int_t) num_cols;
  (A->nnz)   = (magma_int_t) num_nonzeros;

  magma_int_t *coo_col, *coo_row;
  magmaDoubleComplex *coo_val;
  
  coo_col = (magma_int_t *) malloc(A->nnz*sizeof(magma_int_t));
  assert(coo_col != NULL);

  coo_row = (magma_int_t *) malloc(A->nnz*sizeof(magma_int_t)); 
  assert( coo_row != NULL);

  coo_val = (magmaDoubleComplex *) malloc(A->nnz*sizeof(magmaDoubleComplex));
  assert( coo_val != NULL);


  printf("Reading sparse matrix from file (%s):",filename);
  fflush(stdout);


  if (mm_is_real(matcode) || mm_is_integer(matcode)){
    for(magma_int_t i = 0; i < A->nnz; ++i){
      magma_int_t ROW ,COL;
      double VAL;  // always read in a double and convert later if necessary
      
      fscanf(fid, " %d %d %lf \n", &ROW, &COL, &VAL);   
      if( VAL == 0 ) 
        csr_compressor=1;
      coo_row[i] = (magma_int_t) ROW - 1; 
      coo_col[i] = (magma_int_t) COL - 1;
      coo_val[i] = MAGMA_Z_MAKE( VAL, 0.);
    }
  } else {
    printf("Unrecognized data type\n");
    exit(1);
  }
  
  fclose(fid);
  printf(" done\n");
  
  

  if(mm_is_symmetric(matcode)) { //duplicate off diagonal entries
  printf("detected symmetric case\n");
    magma_int_t off_diagonals = 0;
    for(magma_int_t i = 0; i < A->nnz; ++i){
      if(coo_row[i] != coo_col[i])
        ++off_diagonals;
    }
    magma_int_t true_nonzeros = 2 * off_diagonals + (A->nnz - off_diagonals);
      
    magmaDoubleComplex *new_val;
    magma_int_t* new_row;
    magma_int_t* new_col;
    magma_zmalloc_cpu( &new_val, true_nonzeros );
    magma_imalloc_cpu( &new_row, true_nonzeros );
    magma_imalloc_cpu( &new_col, true_nonzeros );

    magma_int_t ptr = 0;
    for(magma_int_t i = 0; i < A->nnz; ++i) {
        if(coo_row[i] != coo_col[i]) {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
        new_col[ptr] = coo_row[i];  
        new_row[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;  
      } else 
      {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
      }
    }      
    
    free (coo_row);
    free (coo_col);
    free (coo_val);

    coo_row = new_row;  
    coo_col = new_col; 
    coo_val = new_val;     
    A->nnz = true_nonzeros;
    printf("total number of nonzeros: %d\n",A->nnz);    

  } //end symmetric case
  
  magmaDoubleComplex tv;
  magma_int_t ti;
  
  
  //If matrix is not in standard format, sorting is necessary
  /*
  
    std::cout << "Sorting the cols...." << std::endl;
  // bubble sort (by cols)
  for (int i=0; i<A->nnz-1; ++i)
    for (int j=0; j<A->nnz-i-1; ++j)
      if (coo_col[j] > coo_col[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }

  std::cout << "Sorting the rows...." << std::endl;
  // bubble sort (by rows)
  for (int i=0; i<A->nnz-1; ++i)
    for (int j=0; j<A->nnz-i-1; ++j)
      if ( coo_row[j] > coo_row[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }
  std::cout << "Sorting: done" << std::endl;
  
  */
  
  
  magma_zmalloc_cpu( &A->val, A->nnz );
  assert((A->val) != NULL);
  
  magma_imalloc_cpu( &A->col, A->nnz );
  assert((A->col) != NULL);
  
  magma_imalloc_cpu( &A->row, A->num_rows+1 );
  assert((A->row) != NULL);
  

  
  
  // original code from  Nathan Bell and Michael Garland
  // the output CSR structure is NOT sorted!

  for (magma_int_t i = 0; i < num_rows; i++)
    (A->row)[i] = 0;
  
  for (magma_int_t i = 0; i < A->nnz; i++)
    (A->row)[coo_row[i]]++;
  
  
  //cumsum the nnz per row to get Bp[]
  for(magma_int_t i = 0, cumsum = 0; i < num_rows; i++){     
    magma_int_t temp = (A->row)[i];
    (A->row)[i] = cumsum;
    cumsum += temp;
  }
  (A->row)[num_rows] = A->nnz;
  
  //write Aj,Ax into Bj,Bx
  for(magma_int_t i = 0; i < A->nnz; i++){
    magma_int_t row_  = coo_row[i];
    magma_int_t dest = (A->row)[row_];
    
    (A->col)[dest] = coo_col[i];
    
    (A->val)[dest] = coo_val[i];
    
    (A->row)[row_]++;
  }
  free (coo_row);
  free (coo_col);
  free (coo_val);
  
  for(int i = 0, last = 0; i <= num_rows; i++){
    int temp = (A->row)[i];
    (A->row)[i]  = last;
    last   = temp;
  }
  
  (A->row)[A->num_rows]=A->nnz;
     

  for (magma_int_t k=0; k<A->num_rows; ++k)
    for (magma_int_t i=(A->row)[k]; i<(A->row)[k+1]-1; ++i) 
      for (magma_int_t j=(A->row)[k]; j<(A->row)[k+1]-1; ++j) 

      if ( (A->col)[j] > (A->col)[j+1] ){

        ti = (A->col)[j];
        (A->col)[j] = (A->col)[j+1];
        (A->col)[j+1] = ti;

        tv = (A->val)[j];
        (A->val)[j] = (A->val)[j+1];
        (A->val)[j+1] = tv;

      }
  if( csr_compressor > 0){ // run the CSR compressor to remove zeros
      printf("removing zeros: ");
      magma_z_sparse_matrix B;
      magma_z_mtransfer( *A, &B, Magma_CPU, Magma_CPU ); 
      magma_z_csr_compressor(&(A->val), 
                        &(A->row),
                         &(A->col), 
                       &B.val, &B.row, &B.col, &B.num_rows); 
      B.nnz = B.row[num_rows];
      printf(" remaining nonzeros:%d ", B.nnz); 
      magma_free_cpu( A->val ); 
      magma_free_cpu( A->row ); 
      magma_free_cpu( A->col ); 
      magma_z_mtransfer( B, A, Magma_CPU, Magma_CPU ); 
      magma_z_mfree( &B ); 
      printf("done.\n");
  }
  return MAGMA_SUCCESS;
}




