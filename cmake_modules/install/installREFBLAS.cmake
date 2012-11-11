###
#
#  @file installREFBLAS.cmake
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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE(installExternalPACKAGE)

MACRO(INSTALL_REFBLAS _MODE)

    # Message about refblas
    # ---------------------
    MESSAGE(STATUS "Installing BLAS - refblas version")
    MESSAGE(STATUS "Installing BLAS - do not expect high performance from this reference library!")

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(BLAS_PATH ${CMAKE_INSTALL_PREFIX}/blas)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(BLAS_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Create make.inc
    # ---------------
    IF(EXISTS ${CMAKE_BINARY_DIR}/blas_make.inc)
        FILE(REMOVE ${CMAKE_BINARY_DIR}/blas_make.inc)
    ENDIF()
    STRING(TOUPPER "${CMAKE_BUILD_TYPE}" TYPE)
    FILE(APPEND ${CMAKE_BINARY_DIR}/blas_make.inc
"
SHELL      = /bin/sh
FORTRAN    = ${CMAKE_Fortran_COMPILER}
OPTS       = ${CMAKE_Fortran_FLAGS_${TYPE}}
OPTS      += ${CMAKE_Fortran_FLAGS}
DRVOPTS    = $(OPTS)
LOADER     = ${CMAKE_Fortran_COMPILER}
LOADER    += ${CMAKE_Fortran_LDFLAGS}
ARCH       = ${CMAKE_AR}
ARCHFLAGS  = rc
RANLIB     = ${CMAKE_RANLIB}
BLASLIB    = librefblas.${MORSE_LIBRARY_EXTENSION}
")
    IF(BUILD_SHARED_LIBS)
        FILE(APPEND ${CMAKE_BINARY_DIR}/blas_make.inc
"
OPTS      += -fPIC
DRVOPTS   += $(OPTS)
ARCHFLAGS  = rcs
")
    ENDIF(BUILD_SHARED_LIBS)

    # Define steps of installation
    # ----------------------------
    SET(BLAS_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/blas)
    SET(BLAS_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/blas)
    SET(BLAS_CONFIG_CMD  ${CMAKE_COMMAND} -E copy
                           ${CMAKE_BINARY_DIR}/blas_make.inc
                           ${BLAS_BUILD_PATH}/make.inc)
    SET(BLAS_BUILD_CMD   ${CMAKE_MAKE_PROGRAM})
    SET(BLAS_INSTALL_CMD ${CMAKE_COMMAND} -E make_directory ${BLAS_PATH})

    # Define additional step
    # ----------------------
    UNSET(BLAS_ADD_INSTALL_STEP)
    FOREACH(_task lib)
        LIST(APPEND BLAS_ADD_INSTALL_STEP blas_create_${_task}_path)
        SET(blas_create_${_task}_path_CMD ${CMAKE_COMMAND} -E make_directory ${BLAS_PATH}/${_task})
        SET(blas_create_${_task}_path_DIR ${CMAKE_INSTALL_PREFIX})
    ENDFOREACH()
    FOREACH(_task librefblas)
        LIST(APPEND BLAS_ADD_INSTALL_STEP blas_copy_${_task})
        SET(blas_copy_${_task}_CMD ${CMAKE_COMMAND} -E copy
                                     ${BLAS_BUILD_PATH}/${_task}.${MORSE_LIBRARY_EXTENSION} .)
        SET(blas_copy_${_task}_DIR ${BLAS_PATH}/lib)
    ENDFOREACH()

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("blas" "${BLAS_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(BLAS_VENDOR       "refblas")
    SET(BLAS_LIBRARY_PATH "${BLAS_BUILD_PATH}")
    SET(BLAS_LIBRARY      "${BLAS_LIBRARY_PATH}/librefblas.${MORSE_LIBRARY_EXTENSION}")
    SET(BLAS_LDFLAGS      "-L${BLAS_LIBRARY_PATH} -lrefblas")
    SET(BLAS_LIBRARIES    "refblas")

ENDMACRO(INSTALL_REFBLAS)

##
## @end file installREFBLAS.cmake
##
