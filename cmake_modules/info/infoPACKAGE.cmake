###
#
#  @file infoPACKAGE.cmake
#
#  @project MORSE
#  MORSE is a software package provided by:
#     Inria Bordeaux - Sud-Ouest,
#     Univ. of Tennessee,
#     Univ. of California Berkeley,
#     Univ. of Colorado Denver.
#
#  @version 0.1.0
#  @author Cedric Castagnede
#  @date 13-07-2012
#
###

###
#
# INFO_INSTALL_PACKAGE: Macro to define test functions
#
###
MACRO(INFO_DEPS_PACKAGE _NAME)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _UPIDP)

    # Looking for info for MORSE_ALL_PACKAGES
    # ---------------------------------------
    IF("^BLAS$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoBLAS)
        BLAS_INFO_DEPS()

    ELSEIF("^REFBLAS$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoREFBLAS)
        REFBLAS_INFO_DEPS()

    ELSEIF("^EIGEN$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoEIGEN)
        EIGEN_INFO_DEPS()

    ELSEIF("^TMG$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoTMG)
        TMG_INFO_DEPS()

    ELSEIF("^LAPACK$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoLAPACK)
        LAPACK_INFO_DEPS()

    ELSEIF("^CBLAS$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoCBLAS)
        CBLAS_INFO_DEPS()

    ELSEIF("^LAPACKE$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoLAPACKE)
        LAPACKE_INFO_DEPS()

    ELSEIF("^PLASMA$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoPLASMA)
        PLASMA_INFO_DEPS()

    ELSEIF("^MPI$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoOPENMPI)
        MPI_INFO_DEPS()

    ELSEIF("^HWLOC$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoHWLOC)
        HWLOC_INFO_DEPS()

    ELSEIF("^FXT$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoFXT)
        FXT_INFO_DEPS()

    ELSEIF("^STARPU$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoSTARPU)
        STARPU_INFO_DEPS()

    ELSEIF("^QUARK$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoQUARK)
        QUARK_INFO_DEPS()

    ELSEIF("^CUDA$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoCUDA)
        CUDA_INFO_DEPS()

    ELSEIF("^OPENCL$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoOPENCL)
        OPENCL_INFO_DEPS()

    ELSEIF("^APPML$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoAPPML)
        APPML_INFO_DEPS()

    ELSEIF("^MAGMA$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoMAGMA)
        MAGMA_INFO_DEPS()

    ELSEIF("^MAGMA_MORSE$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoMAGMA_MORSE)
        MAGMA_MORSE_INFO_DEPS()

    ELSEIF("^MORSE$" STREQUAL "^${_UPIDP}$")
        INCLUDE(infoMORSE)
        MORSE_INFO_DEPS()

    ELSE()
        MESSAGE(FATAL_ERROR "DEV_ERROR - Macro INFO_DEPS_PACKAGE: ${_UPIDP}")

    ENDIF()

ENDMACRO(INFO_DEPS_PACKAGE)

###
#
# INFO_INSTALL_PACKAGE: Macro to define test functions
#
###
MACRO(INFO_INSTALL_PACKAGE _NAME)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _UPIIP)

    # Looking for info for MORSE_ALL_PACKAGES
    # ---------------------------------------
    IF("^BLAS$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoBLAS)
        BLAS_INFO_INSTALL()

    ELSEIF("^REFBLAS$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoREFBLAS)
        REFBLAS_INFO_INSTALL()

    ELSEIF("^EIGEN$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoEIGEN)
        EIGEN_INFO_INSTALL()

    ELSEIF("^TMG$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoTMG)
        TMG_INFO_INSTALL()

    ELSEIF("^LAPACK$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoLAPACK)
        LAPACK_INFO_INSTALL()

    ELSEIF("^CBLAS$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoCBLAS)
        CBLAS_INFO_INSTALL()

    ELSEIF("^LAPACKE$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoLAPACKE)
        LAPACKE_INFO_INSTALL()

    ELSEIF("^PLASMA$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoPLASMA)
        PLASMA_INFO_INSTALL()

    ELSEIF("^CUDA$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoCUDA)
        CUDA_INFO_INSTALL()

    ELSEIF("^MPI$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoOPENMPI)
        MPI_INFO_INSTALL()

    ELSEIF("^HWLOC$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoHWLOC)
        HWLOC_INFO_INSTALL()

    ELSEIF("^FXT$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoFXT)
        FXT_INFO_INSTALL()

    ELSEIF("^STARPU$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoSTARPU)
        STARPU_INFO_INSTALL()

    ELSEIF("^QUARK$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoQUARK)
        QUARK_INFO_INSTALL()

    # Looking for info for MORSE_ALL_LIBRARIES
    # ----------------------------------------
    ELSEIF("^MAGMA$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoMAGMA)
        MAGMA_INFO_INSTALL()

    ELSEIF("^MAGMA_MORSE$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoMAGMA_MORSE)
        MAGMA_MORSE_INFO_INSTALL()

    ELSEIF("^SCALFMM$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoSCALFMM)
        SCALFMM_INFO_INSTALL()

    ELSEIF("^SCALFMM_MORSE$" STREQUAL "^${_UPIIP}$")
        INCLUDE(infoSCALFMM_MORSE)
        SCALFMM_MORSE_INFO_INSTALL()

    ELSE()
        MESSAGE(FATAL_ERROR "DEV_ERROR - Macro INFO_INSTALL_PACKAGE: ${_UPIIP}")

    ENDIF()

ENDMACRO(INFO_INSTALL_PACKAGE)

###
#
# INFO_FIND_PACKAGE: Macro to define infos
#
###
MACRO(INFO_FIND_PACKAGE _NAME)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _UPIFP)

   # Looking for info for MORSE_ALL_PACKAGES
    # --------------------------------------
    IF("^BLAS$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoBLAS)
        BLAS_INFO_FIND()

    ELSEIF("^REFBLAS$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoREFBLAS)
        REFBLAS_INFO_FIND()

    ELSEIF("^EIGEN$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoEIGEN)
        EIGEN_INFO_FIND()

    ELSEIF("^TMG$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoTMG)
        TMG_INFO_FIND()

    ELSEIF("^LAPACK$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoLAPACK)
        LAPACK_INFO_FIND()

    ELSEIF("^CBLAS$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoCBLAS)
        CBLAS_INFO_FIND()

    ELSEIF("^LAPACKE$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoLAPACKE)
        LAPACKE_INFO_FIND()

    ELSEIF("^PLASMA$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoPLASMA)
        PLASMA_INFO_FIND()

    ELSEIF("^MPI$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoOPENMPI)
        MPI_INFO_FIND()

    ELSEIF("^HWLOC$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoHWLOC)
        HWLOC_INFO_FIND()

    ELSEIF("^FXT$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoFXT)
        FXT_INFO_FIND()

    ELSEIF("^STARPU$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoSTARPU)
        STARPU_INFO_FIND()

    ELSEIF("^QUARK$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoQUARK)
        QUARK_INFO_FIND()

    ELSEIF("^OPENCL$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoOPENCL)
        OPENCL_INFO_FIND()

    ELSEIF("^APPML$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoAPPML)
        APPML_INFO_FIND()

    ELSEIF("^METIS$" STREQUAL "^${_UPIFP}$")
        INCLUDE(infoMETIS)
        METIS_INFO_FIND()

    ELSE()
        MESSAGE(FATAL_ERROR "DEV_ERROR - Macro INFO_FIND_PACKAGE: ${_UPIFP}")

    ENDIF()

ENDMACRO(INFO_FIND_PACKAGE)

##
## @end file infoPACKAGE.cmake
##
