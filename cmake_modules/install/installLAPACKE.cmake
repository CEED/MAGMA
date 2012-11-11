###
#
#  @file installLAPACKE.cmake
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

MACRO(INSTALL_LAPACKE _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(LAPACKE_PATH ${CMAKE_INSTALL_PREFIX}/lapacke)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(LAPACKE_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Create make.inc
    # ---------------
    IF(EXISTS ${CMAKE_BINARY_DIR}/lapacke_make.inc)
        FILE(REMOVE ${CMAKE_BINARY_DIR}/lapacke_make.inc)
    ENDIF()
    STRING(TOUPPER "${CMAKE_BUILD_TYPE}" TYPE)
    FILE(APPEND ${CMAKE_BINARY_DIR}/lapacke_make.inc "
SHELL      = /bin/sh

FORTRAN    = ${CMAKE_Fortran_COMPILER}
OPTS       = ${CMAKE_Fortran_FLAGS_${TYPE}}
OPTS      += ${CMAKE_Fortran_FLAGS}
DRVOPTS    = $(OPTS)
NOOPT      = ${CMAKE_Fortran_FLAGS_${TYPE}}
NOOPT      = ${CMAKE_Fortran_FLAGS}
LOADER     = ${CMAKE_Fortran_COMPILER}
LOADOPTS   = ${CMAKE_Fortran_LDFLAGS}

CC         = ${CMAKE_C_COMPILER} 
CFLAGS     = ${CMAKE_C_FLAGS_${TYPE}}
CFLAGS    += ${CMAKE_C_FLAGS}

ARCH       = ${CMAKE_AR}
ARCHFLAGS  = cr
RANLIB     = ${CMAKE_RANLIB}

BLASLIB    = ${BLAS_LDFLAGS}
LAPACKLIB  = ${LAPACK_LDFLAGS}
TMGLIB     = libtmg.${MORSE_LIBRARY_EXTENSION}
EIGSRCLIB  = libeigsrc.${MORSE_LIBRARY_EXTENSION}
LINSRCLIB  = liblinsrc.${MORSE_LIBRARY_EXTENSION}
LAPACKELIB = liblapacke.${MORSE_LIBRARY_EXTENSION}
")
    IF(BUILD_SHARED_LIBS)
        FILE(APPEND ${CMAKE_BINARY_DIR}/lapacke_make.inc "
CFLAGS    += -fPIC
OPTS      += -fPIC
NOOPT     += -fPIC
DRVOPTS    = $(OPTS)
LOADOPTS  += -fPIC
ARCHFLAGS  = rcs
")
    ENDIF(BUILD_SHARED_LIBS)

    # Define steps of installation
    # ----------------------------
    SET(LAPACKE_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/lapacke)
    SET(LAPACKE_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/lapacke/lapacke)
    SET(LAPACKE_CONFIG_CMD  ${CMAKE_COMMAND} -E copy
                              ${CMAKE_BINARY_DIR}/lapacke_make.inc
                              ${LAPACKE_SOURCE_PATH}/make.inc)
    SET(LAPACKE_BUILD_CMD   ${CMAKE_MAKE_PROGRAM} lapacke)
    SET(LAPACKE_INSTALL_CMD ${CMAKE_COMMAND} -E make_directory ${LAPACKE_PATH})

    # Define additional step
    # ----------------------
    UNSET(LAPACKE_ADD_INSTALL_STEP)
    FOREACH(_task lib include)
        LIST(APPEND LAPACKE_ADD_INSTALL_STEP lapacke_create_${_task}_path)
        SET(lapacke_create_${_task}_path_CMD ${CMAKE_COMMAND} -E make_directory ${LAPACKE_PATH}/${_task})
        SET(lapacke_create_${_task}_path_DIR ${CMAKE_INSTALL_PREFIX})
    ENDFOREACH()
    FOREACH(_task liblapacke)
        LIST(APPEND LAPACKE_ADD_INSTALL_STEP lapacke_copy_${_task})
        SET(lapacke_copy_${_task}_CMD ${CMAKE_COMMAND} -E copy ${LAPACKE_SOURCE_PATH}/${_task}.${MORSE_LIBRARY_EXTENSION} .)
        SET(lapacke_copy_${_task}_DIR ${LAPACKE_PATH}/lib)
    ENDFOREACH()
    FOREACH(_task lapacke lapacke_config lapacke_utils lapacke_mangling lapacke_mangling_with_flags)
        LIST(APPEND LAPACKE_ADD_INSTALL_STEP lapacke_copy_${_task}_h)
        SET(lapacke_copy_${_task}_h_CMD ${CMAKE_COMMAND} -E copy ${LAPACKE_BUILD_PATH}/include/${_task}.h .)
        SET(lapacke_copy_${_task}_h_DIR ${LAPACKE_PATH}/include)
    ENDFOREACH()

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("lapacke" "${LAPACKE_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(LAPACKE_LIBRARY_PATH "${LAPACKE_SOURCE_PATH}")
    SET(LAPACKE_INCLUDE_PATH "${LAPACKE_BUILD_PATH}/include")
    SET(LAPACKE_LIBRARY      "${LAPACKE_LIBRARY_PATH}/liblapacke.${MORSE_LIBRARY_EXTENSION}")
    SET(LAPACKE_LDFLAGS      "-L${LAPACKE_LIBRARY_PATH} -llapacke")
    SET(LAPACKE_LIBRARIES    "lapacke")

ENDMACRO(INSTALL_LAPACKE)

##
## @end file installLAPACKE.cmake
##
