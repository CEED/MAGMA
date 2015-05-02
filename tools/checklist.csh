#!/bin/tcsh
#
# Looks for things that need to be fixed before a release, e.g.,
# using {malloc,free} instead of magma_{malloc,free}_cpu
# Run at the top level:
#
#     ./tools/checklist.py > checklist.txt
#
# @author Mark Gates

svn st -vq | perl -pi -e 's/^.{41}//' | sort \
    | egrep -v '^\.$$|obsolete|Makefile\.|deprecated|^exp|contrib|\.png' \
    | egrep 'Makefile|\w\.\w' >! files.txt

setenv FILES   "`cat files.txt`"
setenv HEADERS "`egrep '\.h' files.txt`"

echo "============================================================ required fixes"

# fixed except sparse
echo "========== C malloc, instead of magma_*malloc_cpu                          *** required fix ***"
egrep '\bmalloc *\('                   $FILES | egrep -v 'quark|alloc.cpp'
echo

# fixed except sparse
echo "========== C free, instead of magma_free_cpu                               *** required fix ****"
egrep '^ *free *\('                    $FILES | egrep -v 'quark|alloc.cpp'
echo

# fixed except trevc3; needs rewrite to fix
echo "========== C++ new, instead of magma_*malloc_cpu                           *** required fix (currently only thread_queue & trevc3_mt use it) ***"
egrep '= *new\b'                       $FILES
egrep '\bnew +\w+\('                   $FILES
echo

# fixed except trevc3; needs rewrite to fix
echo "========== C++ delete,  instead of magma_free_cpu                          *** required fix (currently only thread_queue uses it) ***"
egrep '^ *delete\b'                    $FILES
echo

# fixed
echo "========== C++ streams                                                     *** required fix ***"
egrep '\b(fstream|iostream|sstream|cin|cout)\b' $FILES
echo

# fixed
echo "========== define PRECISION_{s, d, c, z} outside s, d, c, z files, resp.   *** required fix ***"
egrep '^ *# *define +PRECISION_s' */[dcz]*.{h,c,cu,cpp} sparse-iter/*/[dcz]*.{h,c,cu,cpp} sparse-iter/*/magma_[dcz]*.{h,c,cu,cpp} -l
egrep '^ *# *define +PRECISION_d' */[scz]*.{h,c,cu,cpp} sparse-iter/*/[scz]*.{h,c,cu,cpp} sparse-iter/*/magma_[scz]*.{h,c,cu,cpp} -l
egrep '^ *# *define +PRECISION_c' */[sdz]*.{h,c,cu,cpp} sparse-iter/*/[sdz]*.{h,c,cu,cpp} sparse-iter/*/magma_[sdz]*.{h,c,cu,cpp} -l | egrep -v 'scnrm2.cu'
egrep '^ *# *define +PRECISION_z' */[sdc]*.{h,c,cu,cpp} sparse-iter/*/[sdc]*.{h,c,cu,cpp} sparse-iter/*/magma_[sdc]*.{h,c,cu,cpp} -l | egrep -v 'dznrm2.cu'
echo

# fixed
echo "========== define {REAL,COMPLEX} outside {[sd],[cz]} files, resp.          *** required fix ***"
egrep '^ *# *define +REAL'    */[cz]*.{h,c,cu,cpp} sparse-iter/*/[cz]*.{h,c,cu,cpp} sparse-iter/*/magma_[cz]*.{h,c,cu,cpp} -l | egrep -v 'cblas_[sd].cpp'
egrep '^ *# *define +COMPLEX' */[sd]*.{h,c,cu,cpp} sparse-iter/*/[sd]*.{h,c,cu,cpp} sparse-iter/*/magma_[sd]*.{h,c,cu,cpp} -l
echo

# fixed
echo "========== using PRECISION_z, REAL, COMPLEX without define                 *** required fix ***"
egrep 'defined\( *PRECISION_\w *\)'    $FILES -l | xargs egrep '^ *#define PRECISION_\w'   -L
egrep 'ifdef +PRECISION_\w'            $FILES -l | xargs egrep '^ *#define PRECISION_\w'   -L
egrep 'defined\( *(REAL|COMPLEX) *\)'  $FILES -l | xargs egrep '^ *#define (REAL|COMPLEX)' -L
egrep 'ifdef +(REAL|COMPLEX)'          $FILES -l | xargs egrep '^ *#define (REAL|COMPLEX)' -L | egrep -v 'gemm_stencil_defs.h'
echo

# fixed
echo "========== define PRECISION_z, REAL, COMPLEX without undef in headers      *** required fix ***"
egrep '^ *#define +PRECISION_\w'       $HEADERS  -l | xargs egrep '^ *#undef PRECISION_\w' -L
egrep '^ *#define +REAL'               $HEADERS  -l | xargs egrep '^ *#undef REAL'         -L
egrep '^ *#define +COMPLEX'            $HEADERS  -l | xargs egrep '^ *#undef COMPLEX'      -L
echo

# fixed
echo "========== using defunct GPUSHMEM                                          *** required fix ***"
egrep 'GPUSHMEM'                       $FILES
echo

# fixed
echo "========== cuda.h (MathWorks cannot use CUDA driver routines, cu[A-Z]*)    *** required fix ***"
egrep 'cuda\.h'                        $FILES | egrep -v 'interface_cuda/error.h|fortran2.cpp|for CUDA_VERSION'
echo

# fixed except 2-stage and sparse
echo "========== exit, instead of returning error code                           *** required fix ***"
egrep '\bexit *\( *-?\w+ *\)'          $FILES | egrep -v 'trace.cpp|quark|magma_util.cpp|testings.h|\.py'
echo
echo
echo



echo "============================================================ should be fixed"

echo "========== cuda, cublas functions                                          *** should be fixed ***"
egrep '(cuda|cublas)[a-zA-Z]\w+ *\('   $FILES \
    | egrep -v 'interface_cuda|testing_blas_z.cpp|cudaFuncSetCacheConfig|cudaCreateChannelDesc|cudaCreateTextureObject|cudaDeviceSetSharedMemConfig|cudaDestroyTextureObject|cublasSetAtomicsMode|cudaGetErrorString|cudaMemcpyToSymbol|cudaMemcpyFromSymbol|cudaDeviceSetCacheConfig|cudaBindTexture|cudaUnbindTexture|cudaEventElapsedTime|cudaEventCreateWithFlags|cudaMemGetInfo|cublas\w+Batched|cublasHandle|cublasCreate|cublasDestroy|fortran2.cpp'
echo

echo "========== system includes                                                 *** should be fixed ***"
egrep 'include *<' $FILES | egrep -v 'testing/|\.h: *#include|(plasma|core_blas|cublas_v2|omp)\.h' | egrep -v 'quark'
echo
echo
echo



echo "============================================================ informational"

echo "========== define PRECISION_z, REAL, COMPLEX without use                   (NOT errors, just curious)"
egrep '#define +(PRECISION_\w|REAL|COMPLEX)' $FILES -l \
    | xargs egrep 'defined\(PRECISION_\w|REAL|COMPLEX\)|ifdef (PRECISION_\w|REAL|COMPLEX)' -L
echo
echo
echo



echo "============================================================ stylistic"

# fixed
echo "========== single-line for"
egrep '^ *for *\(.*?;.*?;.*?\).*;'    $FILES
echo

# fixed except sparse
echo "========== space before semicolons"
egrep '\S +;'                         $FILES
echo

# fixed except sparse
# exclude "strings;" and scripts
echo "========== missing space after semicolons"
egrep ';\S'                           $FILES | egrep -v '".*;.*"|\.(sh|py|pl)'
echo

# too many
#echo "========== single-line if, without { } braces"
#egrep '^ *if *\([^{]*;'               $FILES | egrep -v fortran2.cpp
#echo

# way too many exceptions like A(i,j), min(m,n), etc.
#echo "========== no space after comma"
#egrep ',\S'                           $FILES | egrep -v '(min|max)\(\w+,\w+\)|ISEED'
#echo

# 213 files, 4305 lines need fix
#echo "========== trailing spaces"
#egrep '\S +$'               $FILES | egrep -v 'fortran2.cpp' | wc
#echo

# not sure what these do
#echo "========== ngpu and info on own line"
#egrep 'magma_int_t ngpu, '             $FILES
#egrep ', +magma_int_t *\* *info'       $FILES
#echo

# not sure what these do
#echo "========== lwork, lda, etc. starting line (A & lda, work & lwork, should be on same line)"
#egrep '^ +magma_int_t +l\w+[^;]+$'     $FILES
#echo
