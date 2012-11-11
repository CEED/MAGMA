###     
#
#  @file Summmary.cmake
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

MACRO(SUMMARY)
########################
#
# MORSE Summary
#
########################
    MESSAGE(STATUS "-----------------------------------------------------------------------------")
    MESSAGE(STATUS "Status of your MORSE project configuration:")
    MESSAGE(STATUS "-----------------------------------------------------------------------------")
    MESSAGE(STATUS "MORSE modules requested:")
    MESSAGE(STATUS "  - MAGMA                : ${MAGMA}")
    MESSAGE(STATUS "  - MAGMA_MORSE          : ${MAGMA_MORSE}")
    MESSAGE(STATUS "")
    MESSAGE(STATUS "  - MORSE_ENABLE_TESTING : ${MORSE_ENABLE_TESTING}")
    IF(MAGMA)
        MESSAGE(STATUS "")
        MESSAGE(STATUS "Configuration of the MAGMA project:")
        MESSAGE(STATUS "  - MAGMA_USE_FERMI         : ${MAGMA_USE_FERMI}")
        MESSAGE(STATUS "  - MAGMA_USE_PLASMA        : ${MAGMA_USE_PLASMA}")
    ENDIF(MAGMA)
    IF(MAGMA_MORSE)
        MESSAGE(STATUS "")
        MESSAGE(STATUS "Configuration of the MORSE module:")
        #MESSAGE(STATUS "  - MORSE_USE_MULTICORE : ${MORSE_USE_MULTICORE}")
        #MESSAGE(STATUS "")
        IF(HAVE_CUDA)
            MESSAGE(STATUS "  - MORSE_USE_CUDA      : ${HAVE_CUDA}")
        ELSE(HAVE_CUDA)
            MESSAGE(STATUS "  - MORSE_USE_CUDA      : OFF")
        ENDIF(HAVE_CUDA)
        IF(HAVE_MPI)
            MESSAGE(STATUS "  - MORSE_USE_MPI       : ${HAVE_MPI}")
        ELSE(HAVE_MPI)
            MESSAGE(STATUS "  - MORSE_USE_MPI       : OFF")
        ENDIF(HAVE_MPI)
        MESSAGE(STATUS "")
        MESSAGE(STATUS "  - MORSE_SCHED_STARPU  : ${MORSE_SCHED_STARPU}")
        MESSAGE(STATUS "  - MORSE_SCHED_QUARK   : ${MORSE_SCHED_QUARK}")
    ENDIF(MAGMA_MORSE)

########################
#
# IMPLEMENTATION Summary
#
########################

    MESSAGE(STATUS "")
    MESSAGE(STATUS "Configuration of the external libraries: implementation")
    FOREACH(_lib CUDA OPENCL MPI)
        IF(HAVE_${_lib})
            MESSAGE(STATUS "  - ${_lib} :")
            MESSAGE(STATUS "    --> ${_lib}_MODE         : ${${_lib}_USED_MODE}")
            IF(NOT "^${_lib}$" STREQUAL "^MPI$")
            MESSAGE(STATUS "    --> ${_lib}_LIBRARY_PATH : ${${_lib}_LIBRARY_PATH}")
            ENDIF()
            MESSAGE(STATUS "    --> ${_lib}_INCLUDE_PATH : ${${_lib}_INCLUDE_PATH}")
            MESSAGE(STATUS "    --> ${_lib}_LIBRARIES    : ${${_lib}_LIBRARIES}") 
            IF(MORSE_DEBUG_CMAKE)
            MESSAGE(STATUS "    --> ${_lib}_LIBRARY      : ${${_lib}_LIBRARY}") 
            ENDIF(MORSE_DEBUG_CMAKE)
            MESSAGE(STATUS "")
        ENDIF(HAVE_${_lib})
    ENDFOREACH()

########################
#
# RUNTIME Summary
#
########################

    MESSAGE(STATUS "Configuration of the external libraries: runtime")
    FOREACH(_lib STARPU QUARK HWLOC)
        IF(HAVE_${_lib})
            MESSAGE(STATUS "  - ${_lib} :")
            MESSAGE(STATUS "    --> ${_lib}_MODE         : ${${_lib}_USED_MODE}")
            MESSAGE(STATUS "    --> ${_lib}_LIBRARY_PATH : ${${_lib}_LIBRARY_PATH}")
            MESSAGE(STATUS "    --> ${_lib}_INCLUDE_PATH : ${${_lib}_INCLUDE_PATH}")
            MESSAGE(STATUS "    --> ${_lib}_LIBRARIES    : ${${_lib}_LIBRARIES}")
            IF(MORSE_DEBUG_CMAKE)
                MESSAGE(STATUS "    --> ${_lib}_LIBRARY      : ${${_lib}_LIBRARY}")
            ENDIF(MORSE_DEBUG_CMAKE)
            MESSAGE(STATUS "")
        ENDIF(HAVE_${_lib})
    ENDFOREACH()

########################
#
# LINEAR ALGEBRA Summary
#
########################

    MESSAGE(STATUS "Configuration of the external libraries: linear algebra")
    FOREACH(_lib BLAS LAPACK TMG CBLAS LAPACKE PLASMA)
        IF(HAVE_${_lib})
            MESSAGE(STATUS "  - ${_lib} :")
            MESSAGE(STATUS "    --> ${_lib}_MODE         : ${${_lib}_USED_MODE}")
            IF("^${_lib}$" STREQUAL "^TMG$"   OR
               "^${_lib}$" STREQUAL "^BLAS$"  OR
               "^${_lib}$" STREQUAL "^CBLAS$" OR
               "^${_lib}$" STREQUAL "^LAPACK$"  )
            MESSAGE(STATUS "    --> ${_lib}_VENDOR       : ${${_lib}_VENDOR}")
            ENDIF()
            MESSAGE(STATUS "    --> ${_lib}_LIBRARY_PATH : ${${_lib}_LIBRARY_PATH}")
            IF(NOT "^${_lib}$" STREQUAL "^BLAS$" OR
               NOT "^${_lib}$" STREQUAL "^LAPACK$")
            MESSAGE(STATUS "    --> ${_lib}_INCLUDE_PATH : ${${_lib}_INCLUDE_PATH}")
            ENDIF()
            MESSAGE(STATUS "    --> ${_lib}_LIBRARIES    : ${${_lib}_LIBRARIES}")
            IF(MORSE_DEBUG_CMAKE)
            MESSAGE(STATUS "    --> ${_lib}_LIBRARY      : ${${_lib}_LIBRARY}")
            ENDIF(MORSE_DEBUG_CMAKE)
            MESSAGE(STATUS "")
        ENDIF(HAVE_${_lib})
    ENDFOREACH()

#######################
#
# TPL Summary
#
#######################
    MESSAGE(STATUS "Status of third-party packages:")
    FOREACH(_lib FXT) 
        IF(HAVE_${_lib})
            MESSAGE(STATUS "  - ${_lib} :")
            MESSAGE(STATUS "    --> ${_lib}_MODE         : ${${_lib}_USED_MODE}")
            IF("^${_lib}$" STREQUAL "^BLAS$" OR
               "^${_lib}$" STREQUAL "^LAPACK$")
            MESSAGE(STATUS "    --> ${_lib}_VENDOR       : ${${_lib}_VENDOR}")
            ENDIF()
            MESSAGE(STATUS "    --> ${_lib}_LIBRARY_PATH : ${${_lib}_LIBRARY_PATH}")
            IF(NOT "^${_lib}$" STREQUAL "^BLAS$" OR
               NOT "^${_lib}$" STREQUAL "^LAPACK$")
            MESSAGE(STATUS "    --> ${_lib}_INCLUDE_PATH : ${${_lib}_INCLUDE_PATH}")
            ENDIF()
            MESSAGE(STATUS "    --> ${_lib}_LIBRARIES    : ${${_lib}_LIBRARIES}")
            IF(MORSE_DEBUG_CMAKE)
            MESSAGE(STATUS "    --> ${_lib}_LIBRARY      : ${${_lib}_LIBRARY}")
            ENDIF(MORSE_DEBUG_CMAKE)
            MESSAGE(STATUS "")
        ENDIF(HAVE_${_lib})
    ENDFOREACH()

#######################
#
# Help 
#
#######################
    MESSAGE(STATUS "")
    MESSAGE(STATUS "--------------+--------------------------------------------------------------")
    MESSAGE(STATUS "Command       | Description")
    MESSAGE(STATUS "--------------+--------------------------------------------------------------")
    MESSAGE(STATUS "make          | Compile MORSE project in:")
    MESSAGE(STATUS "              |     ${CMAKE_BINARY_DIR}")
    MESSAGE(STATUS "              |")
    MESSAGE(STATUS "make test     | Build and run the unit-tests.")
    MESSAGE(STATUS "              |")
    MESSAGE(STATUS "make install  | Install MORSE projecs in:")
    MESSAGE(STATUS "              |     ${CMAKE_INSTALL_PREFIX}")
    MESSAGE(STATUS "              | To change that:")
    MESSAGE(STATUS "              |     cmake ../ -DCMAKE_INSTALL_PREFIX=yourpath")
    MESSAGE(STATUS "              |")
    MESSAGE(STATUS "--------------+--------------------------------------------------------------")

ENDMACRO(SUMMARY)

##
## @end file Summary.cmake
##
