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
    | egrep -v '^\.$$|obsolete|Makefile\.|deprecated|^exp|contrib|\.png|results/v' \
    | egrep 'Makefile|\w\.\w' >! files.txt

setenv FILES   "`cat files.txt`"
setenv HEADERS "`egrep '\.h' files.txt`"

echo "============================================================ required fixes"

# aasen needs fix
echo "========== no prototypes in cpp files; put in headers                      *** required fix ***"
perl -n0777e 'if ( m/extern "C"\s+\w+\s+(\w+)\([^)]*\) *;/ ) { print "$ARGV: $1\n"; }' $FILES
echo

# fixed except magmablas/*gemm*.h -- some of these DO get multiply included!
echo "========== headers not protecting against multiple inclusion               *** required fix ***"
egrep '^#ifndef \w+_([hH]_?|HPP)\b' -L $HEADERS
egrep '^#define \w+_([hH]_?|HPP)\b' -L $HEADERS
echo

# fixed except sparse mmio
echo "========== C malloc, instead of magma_*malloc_cpu                          *** required fix ***"
egrep '\b(malloc|calloc|realloc|reallocf|valloc|strdup) *\(' $FILES | egrep -v 'quark|alloc.cpp'
echo

# fixed except sparse mmio
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
egrep '\b(fstream|iostream|sstream|cin|cout)\b' $FILES | egrep -v 'checklist.csh'
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
egrep 'GPUSHMEM'                       $FILES | egrep -v 'checklist.csh'
echo

# fixed
echo "========== CUDA driver routines (cu[A-Z]*; MathWorks cannot use these)"
grep 'cu[A-Z]\w+ *\(' `cat files.txt` $FILES \
    | egrep -v 'make_cuComplex|make_cuDoubleComplex|make_cuFloatComplex|cuComplexDoubleToFloat|cuComplexFloatToDouble|cuGetErrorString|cuConj|cuC(real|imag|add|sub|mul|div|addf|subf|mulf|divf|fmaf|abs|fma)'
echo

# fixed
echo "========== cuda.h (mostly only needed for CUDA driver routines, cu[A-Z]*)  *** required fix ***"
egrep 'cuda\.h'                        $FILES | egrep -v 'checklist.csh|interface_cuda/error.h|fortran2.cpp|for CUDA_VERSION'
echo

# fixed except 2-stage
echo "========== exit, instead of returning error code                           *** required fix ***"
egrep '\bexit *\( *-?\w+ *\)'          $FILES | egrep -v 'trace.cpp|quark|magma_util.cpp|testings.h|\.py'
echo

# fixed except sparse
echo "========== old formulas for ceildiv & roundup                              *** required fix ***"
./tools/checklist_ceildiv.pl           $FILES | egrep -v 'checklist_ceildiv.pl|documentation.txt|magma\.h|\.f:|\.F90:'
echo



echo
echo
echo "============================================================ should be fixed"

echo "========== cuda, cublas functions                                          *** should be fixed ***"
egrep '(cuda|cublas)[a-zA-Z]\w+ *\('   $FILES \
    | egrep -v '\.f:|interface_cuda|testing_blas_z.cpp|cudaFuncSetCacheConfig|cudaCreateChannelDesc|cudaCreateTextureObject|cudaDeviceSetSharedMemConfig|cudaDestroyTextureObject|cublasSetAtomicsMode|cudaGetErrorString|cudaMemcpyToSymbol|cudaMemcpyFromSymbol|cudaDeviceSetCacheConfig|cudaBindTexture|cudaUnbindTexture|cudaEventElapsedTime|cudaEventCreateWithFlags|cudaMemGetInfo|cublas\w+Batched|cublasHandle|cublasCreate|cublasDestroy|fortran2.cpp' \
    | egrep -v 'testing_z([a-z]+)(_\w+)?\.cpp:.*cublasZ\1'
echo

echo "========== system includes                                                 *** should be fixed ***"
egrep 'include *<' $FILES | egrep -v 'testing/|\.h: *#include|(plasma|core_blas|cublas_v2|omp)\.h' | egrep -v 'quark'
echo

# fixed
echo "========== device sync, instead of queue sync                              *** should be fixed ***"
egrep '(magma_device_sync|cudaDeviceSynchronize)' $FILES | egrep -v 'interface.cpp'
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

# mostly fixed; some exceptions
echo "========== extraneous newlines after {"
perl -n0777e 'if ( m/^(.*\{) *\n *\n/m ) { print "$ARGV: $1\n"; }' $FILES
echo

# mostly fixed; some exceptions
echo "========== extraneous newlines before }"
perl -n0777e 'if ( m/\n *\n( *\}.*)/ ) { print "$ARGV: $1\n"; }' $FILES
echo

# fixed
echo "========== extraneous newlines at EOF"
perl -n0777e 'if ( m/\n\n+$/     ) { print "$ARGV\n"; }' $FILES
echo

# fixed
echo "========== single-line for"
egrep '^ *for *\(.*?;.*?;.*?\).*;'    $FILES
echo

# fixed
echo "========== space before semicolons"
egrep '\S +;'                         $FILES | egrep -v 'quark'
echo

# fixed
# exclude "strings;" and scripts
echo "========== missing space after semicolons"
egrep ';\S'                           $FILES | egrep -v '".*;.*"|\.(csh|sh|py|pl)|quark'
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
