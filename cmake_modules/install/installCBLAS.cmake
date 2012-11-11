###
#
#  @file installCBLAS.cmake
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

MACRO(INSTALL_CBLAS _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(CBLAS_PATH ${CMAKE_INSTALL_PREFIX}/cblas)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(CBLAS_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Create make.inc
    # ---------------
    IF(EXISTS ${CMAKE_BINARY_DIR}/cblas_make.inc)
        FILE(REMOVE ${CMAKE_BINARY_DIR}/cblas_make.inc)
    ENDIF()
    STRING(TOUPPER "${CMAKE_BUILD_TYPE}" TYPE)
    FILE(APPEND ${CMAKE_BINARY_DIR}/cblas_make.inc
"
SHELL      = /bin/sh
CC         = ${CMAKE_C_COMPILER}
CFLAGS     = ${CMAKE_C_FLAGS_${TYPE}}
CFLAGS    += ${CMAKE_C_FLAGS}
CFLAGS    += ${FORTRAN_MANGLING_DETECTED}

FC         = ${CMAKE_Fortran_COMPILER}
FFLAGS     = ${CMAKE_Fortran_FLAGS_${TYPE}}
FFLAGS    += ${CMAKE_Fortran_FLAGS}
FFLAGS    += ${CMAKE_Fortran_LDFLAGS} 

LOADER     = ${CMAKE_Fortran_COMPILER}
LOADER    += ${CMAKE_Fortran_LDFLAGS}
ARCH       = ${CMAKE_AR}
ARCHFLAGS  = cr
RANLIB     = ${CMAKE_RANLIB}
BLLIB      = ${BLAS_LDFLAGS}
CBLIB      = libcblas.${MORSE_LIBRARY_EXTENSION}
")
    IF(BUILD_SHARED_LIBS)
        FILE(APPEND ${CMAKE_BINARY_DIR}/cblas_make.inc "
CFLAGS    += -fPIC
FFLAGS    += -fPIC
ARCHFLAGS  = rcs
")
    ENDIF(BUILD_SHARED_LIBS)

    # Define steps of installation
    # ----------------------------
    SET(CBLAS_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/cblas)
    SET(CBLAS_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/cblas)
    SET(CBLAS_CONFIG_CMD  ${CMAKE_COMMAND} -E copy
                            ${CMAKE_BINARY_DIR}/cblas_make.inc
                            ${CBLAS_BUILD_PATH}/Makefile.in)
    SET(CBLAS_BUILD_CMD   ${CMAKE_MAKE_PROGRAM} alllib)
    SET(CBLAS_INSTALL_CMD ${CMAKE_COMMAND} -E make_directory ${CBLAS_PATH})

    # Define additional step (warning - the order in very important)
    # --------------------------------------------------------------
    UNSET(CBLAS_ADD_INSTALL_STEP)
    FOREACH(_task lib include)
        LIST(APPEND CBLAS_ADD_INSTALL_STEP cblas_create_${_task}_path)
        SET(cblas_create_${_task}_path_CMD ${CMAKE_COMMAND} -E make_directory ${CBLAS_PATH}/${_task})
        SET(cblas_create_${_task}_path_DIR ${CMAKE_INSTALL_PREFIX})
    ENDFOREACH()
    FOREACH(_task libcblas)
        LIST(APPEND CBLAS_ADD_INSTALL_STEP cblas_copy_${_task})
        SET(cblas_copy_${_task}_CMD ${CMAKE_COMMAND} -E copy
                                    ${CBLAS_BUILD_PATH}/src/${_task}.${MORSE_LIBRARY_EXTENSION} .)
        SET(cblas_copy_${_task}_DIR ${CBLAS_PATH}/lib)
    ENDFOREACH()
    FOREACH(_task cblas)
        LIST(APPEND CBLAS_ADD_INSTALL_STEP cblas_copy_${_task}_h)
        SET(cblas_copy_${_task}_h_CMD ${CMAKE_COMMAND} -E copy ${CBLAS_BUILD_PATH}/include/${_task}.h .)
        SET(cblas_copy_${_task}_h_DIR ${CBLAS_PATH}/include)
    ENDFOREACH()

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("cblas" "${CBLAS_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(CBLAS_LIBRARY_PATH "${CBLAS_BUILD_PATH}/src")
    SET(CBLAS_INCLUDE_PATH "${CBLAS_BUILD_PATH}/include")
    SET(CBLAS_LIBRARY      "${CBLAS_LIBRARY_PATH}/libcblas.${MORSE_LIBRARY_EXTENSION}")
    SET(CBLAS_LDFLAGS      "-L${CBLAS_LIBRARY_PATH} -lcblas")
    SET(CBLAS_LIBRARIES    "cblas")

ENDMACRO(INSTALL_CBLAS)

##
## @end file installCBLAS.cmake
##
