###
#
# @file      : CMakeLists.txt
#
# @description   : Project MAGMA
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 20-01-2012
# @last modified : mer. 16 mai 2012 11:37:18 CEST
#
###


CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE(RulesPrecisions)

# Generate the magma sources for all possible precisions
# ------------------------------------------------------
SET(MAGMA_CTRL_GENERATED "")
SET(MAGMA_CTRLF_GENERATED "")
SET(ZSRCF "")
SET(ZSRC
         ../control/zprint.cpp
         ../control/zpanel_to_q.cpp
         ../interface_cuda/blas_z.cpp
         ../interface_cuda/zset_get.cpp
         )
IF(HAVE_PGI)
    SET(ZSRC ${ZSRC}
             ../control/magma_zf77pgi.cpp
             )
ELSE()
    SET(ZSRCF ${ZSRCF}
              ../control/magma_zfortran.F90
              )
    SET(ZSRC ${ZSRC}
             ../control/magma_zf77.cpp
             )

    SET(MAGMA_CTRLF
             ../control/magma_param.f90
             ../control/magma.f90
             )
ENDIF()

precisions_rules_py(MAGMA_CTRL_GENERATED
                    "${ZSRC}"
                    PRECISIONS ${MAGMA_PRECISION})
precisions_rules_py(MAGMA_CTRLF_GENERATED
                    "${ZSRCF}"
                    PRECISIONS ${MAGMA_PRECISION})

# Define the list of sources
# --------------------------
SET(MAGMA_CTRLF ${MAGMA_CTRLF_GENERATED}
                ${MAGMA_CTRLF}
                )

SET(MAGMA_CTRL ${MAGMA_CTRLF}
               ${MAGMA_CTRL_GENERATED}
               ../control/auxiliary.cpp
               ../control/xerbla.cpp
               ../control/get_nb_tesla.cpp
               ../control/get_nb_fermi.cpp
               ../control/timer.cpp
               ../control/trace.cpp
               ../control/strlcpy.cpp
               ../control/bulge_auxiliary.cpp
               ../interface_cuda/alloc.cpp
               ../interface_cuda/error.cpp
               ../interface_cuda/set_get.cpp
               ../interface_cuda/interface.cpp
               )

# Generate the magma sources for all possible precisions
# ------------------------------------------------------
SET(MAGMA_SRCS "")
SET(MAGMA_SRCS_GENERATED "")

# ------------------------------------------------------------
# Loosely, the order is alphabetic in each section, but mixed precision,
# driver, and multi-GPU routines are separated where it made sense.
# Double precision "d" files that generate "s" files must also be listed
# in ZSRC, despite the name.
SET(ZSRC
### Cholesky, GPU interface
    zcposv_gpu.cpp
    zposv_gpu.cpp
    zpotrf_gpu.cpp
    zpotri_gpu.cpp
    zpotrs_gpu.cpp
    zlauum_gpu.cpp
    ztrtri_gpu.cpp
    zpotrf_mgpu.cpp

### Cholesky, CPU interface
    zposv.cpp
    zpotrf.cpp
    zpotrf2_ooc.cpp
    zpotrf3_mgpu.cpp
    zpotri.cpp
    zlauum.cpp
    ztrtri.cpp

### LU, GPU interface
    zcgesv_gpu.cpp
    zcgetrs_gpu.cpp
    zgesv_gpu.cpp
    zgetrf_gpu.cpp
    zgetrf_nopiv_gpu.cpp
    zgetri_gpu.cpp
    zgetrs_gpu.cpp
    zgetrf_mgpu.cpp

### LU, CPU interface
    zgesv.cpp
    zgetrf.cpp
    zgetrf3_ooc.cpp
    zgetrf1_mgpu.cpp
    zgetf2_nopiv.cpp
    zgetrf_nopiv.cpp

### QR and least squares, GPU interface
    zcgeqrsv_gpu.cpp
    zgelqf_gpu.cpp
    zgels3_gpu.cpp
    zgels_gpu.cpp
    zgeqrf2_gpu.cpp
    zgeqrf3_gpu.cpp
    zgeqrf_gpu.cpp
    zgeqrs3_gpu.cpp
    zgeqrs_gpu.cpp
    zlarfb_gpu.cpp
    zungqr_gpu.cpp
    zunmql2_gpu.cpp
    zunmqr2_gpu.cpp
    zunmqr_gpu.cpp
    zgeqrf_mgpu.cpp

# QR, CPU interface
    zgelqf.cpp
    zgeqlf.cpp
    zgeqrf-v4.cpp
    zgeqrf.cpp
    zgeqrf_ooc.cpp
    zungqr.cpp
    zunmql.cpp
    zunmqr.cpp
    zunmqr_m.cpp
    zgeqp3.cpp
    zlaqps.cpp

# symmetric eigenvalue, GPU interface
    dsyevd_gpu.cpp
    zheevd_gpu.cpp
    zheevdx_gpu.cpp
    zheevr_gpu.cpp
    zheevx_gpu.cpp
    zhetrd2_gpu.cpp
    zhetrd_gpu.cpp
    zunmtr_gpu.cpp

# symmetric eigenvalues, CPU interface
    dsyevd.cpp
    dsyevdx.cpp
    zheevd.cpp
    zheevdx.cpp
    zheevr.cpp
    zheevx.cpp
    dlaex0.cpp
    dlaex1.cpp
    dlaex3.cpp
    dstedx.cpp
    zhetrd.cpp
    zlatrd.cpp
    zlatrd2.cpp
    zstedx.cpp
    zungtr.cpp
    zunmtr.cpp

# symmetric eigenvalues, multi-GPU (not yet released)
#    zheevd_m.cpp
#    dlaex0_m.cpp
#    dlaex1_m.cpp
#    dlaex3_m.cpp
#    dstedx_m.cpp
#    zstedx_m.cpp
#    zunmtr_m.cpp

# symmetric eigenvalues 2-stage
    dbulge_aux.cpp
    zbulge_applyQ.cpp
    zbulge_applyQ_v2.cpp
    zbulge_aux.cpp
    zbulge_kernel.cpp
    zbulge_kernel_v2.cpp
    zhetrd_he2hb.cpp
    zhetrd_he2hb_mgpu.cpp
    zhetrd_hb2st.cpp
    zbulge_back.cpp
    zhetrd_bhe2trc_v3.cpp
    zhetrd_bhe2trc_v5.cpp
    zungqr_2stage_gpu.cpp
    zunmqr_gpu_2stage.cpp
    zheevdx_2stage.cpp
    zhegvdx_2stage.cpp

# generalized symmetric eigenvalues
    dsygvd.cpp
    zhegst.cpp
    zhegvd.cpp
    zhegvdx.cpp
    zhegvr.cpp
    zhegvx.cpp
    zhegst_gpu.cpp
    zhegst_m.cpp
    zhegvd_m.cpp
    ztrsm_m.cpp

# non-symmetric eigenvalues
    dgeev.cpp
    zgeev.cpp
    zgehrd.cpp
    zgehrd2.cpp
    zlahr2.cpp
    zlahru.cpp
    zunghr.cpp

# SVD
    dgesvd.cpp
    zgebrd.cpp
    zgesvd.cpp
    zlabrd_gpu.cpp
     )

IF(MAGMA_USE_PLASMA)
    SET(ZSRC ${ZSRC}
             zgessm_gpu.cpp
             zssssm_gpu.cpp
             zgetrl_gpu.cpp
             ztstrf_gpu.cpp
             )
ENDIF()
precisions_rules_py(MAGMA_SRCS_GENERATED
                    "${ZSRC}"
                    PRECISIONS ${MAGMA_PRECISION})

# Define the list of sources
# --------------------------
SET(MAGMA_SRCS 
               ${MAGMA_CTRL}
               ${MAGMA_SRCS_GENERATED}
               dlaed2.f
               slaed2.f
               )

# Add dependencies on the header files
# ------------------------------------
SET_SOURCE_FILES_PROPERTIES(${MAGMA_SRCS} 
    PROPERTIES OBJECTS_DEPENDS magma_include
    )

# Compile step
# ------------
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/control)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/interface_cuda)
ADD_LIBRARY(magma ${MAGMA_SRCS})
SET_PROPERTY(TARGET magma PROPERTY LINKER_LANGUAGE Fortran)

FOREACH(_external_package "blas" "lapack" "plasma")
    STRING(TOUPPER "${_external_package}" _NAMEVAR)
    IF(DEFINED ${_NAMEVAR}_BUILD_MODE)
        ADD_DEPENDENCIES(magma ${_external_package}_install)
    ENDIF()
ENDFOREACH()

# Install step
# ------------
INSTALL(TARGETS magma
        DESTINATION lib)

FOREACH(_src ${MAGMA_CTRLF})
    GET_FILENAME_COMPONENT(_namesrc ${_src} NAME)
    STRING(REGEX REPLACE ".[Ff]90" ".mod"  _namemod "${_namesrc}")
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${_namemod}
            DESTINATION include)
ENDFOREACH()

###
### END CMakeLists.txt
###
