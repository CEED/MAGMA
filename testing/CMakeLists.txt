###
#
# @file      : CMakeLists.txt
#
# @description   : Project MAGMA
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 23-01-2012
# @last modified : mar. 17 avril 2012 15:53:16 CEST
#
###

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE(RulesPrecisions)

# Add include and link directories
# --------------------------------
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)

# Check for the subdirectories 
# ----------------------------
ADD_SUBDIRECTORY(matgen)
ADD_SUBDIRECTORY(lin)

# Generate the magma testing sources for all possible precisions
# --------------------------------------------------------------
SET(MAGMA_TEST_GENERATED "")
SET(ZSRC 
# ----------
# BLAS
        testing_zgemm.cpp
        testing_zgemv.cpp
        testing_zhemv.cpp
        testing_zsymv.cpp
        #testing_zhemm_mgpu.cpp
        #testing_zher2k_mgpu.cpp
        testing_auxiliary.cpp
        testing_zblas.cpp
        testing_zgeadd.cpp
        testing_zlacpy.cpp
        testing_zsymmetrize.cpp
        testing_zswap.cpp
        testing_zhemv_mgpu.cpp

# ----------
# Cholesky, GPU interface
        testing_zcposv_gpu.cpp
        testing_zposv_gpu.cpp
        testing_zpotrf_gpu.cpp
        testing_zpotri_gpu.cpp
        testing_zpotrf_mgpu.cpp

# Cholesky, CPU interface
        testing_zposv.cpp
        testing_zpotrf.cpp
        testing_zpotri.cpp

# ----------
# LU, GPU interface
        testing_zcgesv_gpu.cpp
        testing_zgesv_gpu.cpp
        testing_zgetrf_gpu.cpp
        testing_zgetri_gpu.cpp
        testing_zgetrf_mgpu.cpp

# LU, CPU interface
        testing_zgesv.cpp
        testing_zgetrf.cpp

# ----------
# QR and least squares, GPU interface
        testing_zcgeqrsv_gpu.cpp
        testing_zgelqf_gpu.cpp
        testing_zgeqrf_gpu.cpp
        testing_zgeqr2_gpu.cpp
        testing_zgeqrs3_gpu.cpp
        testing_zgeqrs_gpu.cpp
        testing_zlarfb_gpu.cpp
        testing_zungqr_gpu.cpp
        testing_zunmqr_gpu.cpp
        testing_zgeqrf_mgpu.cpp

# QR, CPU interface
        testing_zgelqf.cpp
        testing_zgeqlf.cpp
        testing_zgeqrf.cpp
        testing_zungqr.cpp
        testing_zunmqr.cpp
        testing_zgeqp3.cpp

# ----------
# symmetric eigenvalues, GPU interface
        testing_dsyevd_gpu.cpp
        testing_zheevd_gpu.cpp
        testing_zhetrd_gpu.cpp

# symmetric eigenvalues, CPU interface
        testing_dsyevd.cpp
        testing_zheevd.cpp
        testing_zhetrd.cpp

# symmetric eigenvalues, 2-stage
        testing_zhetrd_he2hb.cpp
        #testing_zhetrd_he2hb_mgpu.cpp
        testing_zhegvdx_2stage.cpp

# generalized symmetric eigenvalues
        testing_dsygvd.cpp
        testing_zhegvd.cpp
        testing_zhegvdx.cpp

# ----------
# non-symmetric eigenvalues
        testing_dgeev.cpp
        testing_zgeev.cpp
        testing_zgehrd.cpp

# ----------
# SVD
        testing_zgesvd.cpp
        testing_zgebrd.cpp
   )

IF(HAVE_PGI)
    SET(CUDA_INTERFACE
        ${CUDA_TOOLKIT_ROOT_DIR}/src/fortran_thunking.c
       )
    SET(ZSRC
        ${ZSRC}
        testing_zgetrf_gpu_f.cuf
       )
ELSE()
    IF(MORSE_CUDA_WITH_FORTRAN)
        SET(CUDA_INTERFACE
            ${CUDA_TOOLKIT_ROOT_DIR}/src/fortran.c
           )
        SET(ZSRC
            ${ZSRC}
            testing_zgetrf_gpu_f.F90
            testing_zgetrf_f.f90
           )
    ENDIF()
ENDIF()

precisions_rules_py(MAGMA_TEST_GENERATED "${ZSRC}"
                    PRECISIONS "${MAGMA_PRECISION}"
                    DICTIONARY "MAGMA")

# Define the list of common sources
# ---------------------------------
SET(MAGMA_COMMON_GENERATED "")
SET(ZSRC
    testing_zutil.cpp
   )

precisions_rules_py(MAGMA_COMMON_GENERATED "${ZSRC}"
                    PRECISIONS "${MAGMA_PRECISION}"
                    DICTIONARY "MAGMA")

SET(MAGMA_COMMON
    ${CUDA_INTERFACE}
    ${MAGMA_COMMON_GENERATED}
    testing_util.cpp
   )

# Force generation of common sources files
# ----------------------------------------
ADD_CUSTOM_TARGET(magma_common_testing ALL SOURCES ${MAGMA_COMMON})

# Create libmagma_testing.a
# --------------------------
ADD_LIBRARY(magma_testing STATIC ${MAGMA_COMMON})
SET_PROPERTY(TARGET magma_testing PROPERTY LINKER_LANGUAGE Fortran)
ADD_DEPENDENCIES(magma_testing magma_common_testing)

# Add tests (C/CPP)
# -----------------
UNSET(libs_for_tests)
LIST(APPEND libs_for_tests "magma")
LIST(APPEND libs_for_tests "magmablas")
LIST(APPEND libs_for_tests "magma")
LIST(APPEND libs_for_tests "magma_testing")
LIST(APPEND libs_for_tests "magma_lapacktest")
LIST(APPEND libs_for_tests "${CUDA_LIST_LDFLAGS}")
LIST(APPEND libs_for_tests "tmglib")
FOREACH(_dep PLASMA LAPACKE LAPACK CBLAS BLAS TMG)
    IF(HAVE_${_dep})
        UNSET(${_dep}_path_flags)
        FOREACH(_path ${${_dep}_LIBRARY_PATH})
            LIST(APPEND ${_dep}_path_flags "-L${_path}")
        ENDFOREACH()
        LIST(APPEND libs_for_tests ${${_dep}_path_flags})
        LIST(APPEND libs_for_tests ${${_dep}_LIBRARY})
    ENDIF()
ENDFOREACH()
FOREACH(_dep DL)
    IF(HAVE_${_dep})
        LIST(APPEND libs_for_tests ${${_dep}_LIST_LDFLAGS})
    ENDIF()
ENDFOREACH()
LIST(APPEND libs_for_tests ${CMAKE_Fortran_FLAGS})
LIST(APPEND libs_for_tests ${CMAKE_Fortran_LDFLAGS})
LIST(APPEND libs_for_tests ${MAGMA_EXTRA_LIBS}) 

FOREACH(_test_src ${MAGMA_TEST_GENERATED})
    GET_FILENAME_COMPONENT(_name_exe ${_test_src} NAME_WE)
    ADD_EXECUTABLE(${_name_exe} ${_test_src})
    SET_PROPERTY(TARGET ${_name_exe} PROPERTY LINKER_LANGUAGE Fortran)
    TARGET_LINK_LIBRARIES(${_name_exe} ${libs_for_tests})
    ADD_TEST(NAME ${_name_exe} 
             COMMAND ${_name_exe} -M 1000 -N 500)
    INSTALL(TARGETS ${_name_exe}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/testing)
ENDFOREACH()

###
### END CMakeLists.txt
###
