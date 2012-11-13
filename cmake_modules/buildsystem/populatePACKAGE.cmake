###
#
#  @file populatePACKAGE.cmake 
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
# POPULATE_COMPILE_SYSTEM: Call INCLUDE_DIRECTORIES, LINK_DIRECTORIES, etc...
#
###
MACRO(POPULATE_COMPILE_SYSTEM _PCS_PACKAGE)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_PCS_PACKAGE}" _UPPERPCS_PACKAGE)
    STRING(TOLOWER "${_PCS_PACKAGE}" _LOWERPCS_PACKAGE)
    IF("^${_UPPERPCS_PACKAGE}$" STREQUAL "^CUDA$")
        MESSAGE(STATUS "Looking for CUDA - found")

        # Define all flags needed
        IF(CUDA_VERSION VERSION_LESS "4.0")
            SET(CUDA_HOST_COMPILATION_CPP OFF)
        ENDIF(CUDA_VERSION VERSION_LESS "4.0")
        SET(CUDA_BUILD_EMULATION OFF)
        IF(${ARCH_X86_64})
            LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
            SET(CUDA_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
        ELSE()
            LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib)
            SET(CUDA_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib")
        ENDIF()
        SET(CUDA_LIBRARIES "cudart;cublas;cuda")
        SET(CUDA_LDFLAGS "-L${CUDA_LIBRARY_PATH} -lcudart -lcuda")
        SET(CUDA_LIBRARY "${CUDA_CUDART_LIBRARY};${CUDA_CUDA_LIBRARY}")
        SET(CUDA_INCLUDE_PATH "${CUDA_INCLUDE_DIRS}")
        INCLUDE_DIRECTORIES(${CUDA_INCLUDE_PATH})
        IF("^${MORSE_USE_CUDA}}$" STREQUAL "^$")
            SET(MORSE_USE_CUDA "ON" CACHE STRING
                "Enable/Disable CUDA dependency (ON/OFF/<not-defined>)" FORCE)
        ENDIF()
        SET(CUDA_USED_MODE "FIND")
        SET(HAVE_CUDA ON)
        SET(HAVE_CUBLAS ON)
        
        # Should move them to a magma_config.h.in 
        ADD_DEFINITIONS(-DHAVE_CUDA)
        ADD_DEFINITIONS(-DHAVE_CUBLAS)

        # Fortran mangling for cuda >= 5.0
        SET(MORSE_CUDA_WITH_FORTRAN ON)
        IF(CUDA_VERSION VERSION_GREATER "4.9")
            IF(FORTRAN_MANGLING_DETECTED)
                IF("${FORTRAN_MANGLING_DETECTED}" STREQUAL "-DADD_")
                    ADD_DEFINITIONS(-DCUBLAS_GFORTRAN)
                ELSEIF("${FORTRAN_MANGLING_DETECTED}" STREQUAL "-DUPCASE")
                    ADD_DEFINITIONS(-DCUBLAS_INTEL_FORTRAN)
                ELSE()
                    SET(MORSE_CUDA_WITH_FORTRAN OFF)
                ENDIF()
            ENDIF()
        ENDIF()

    ELSEIF("^${_UPPERPCS_PACKAGE}$" STREQUAL "^MPI$")
        IF(MPI_EP)
            LINK_DIRECTORIES(${MPI_LIBRARY_PATH})
            INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})

        ELSE(MPI_EP)
            UNSET(MPI_INCLUDE_PATH)
            UNSET(MPI_LIBRARY_PATH)
            UNSET(MPI_LIBRARIES)
            UNSET(MPI_LIBRARY)
            UNSET(MPI_LDFLAGS)

            LIST(APPEND MPI_INCLUDE_PATH "${MPI_C_INCLUDE_PATH}")
            LIST(APPEND MPI_INCLUDE_PATH "${MPI_CXX_INCLUDE_PATH}")
            LIST(APPEND MPI_INCLUDE_PATH "${MPI_Fortran_INCLUDE_PATH}")
            LIST(REMOVE_DUPLICATES MPI_INCLUDE_PATH)
            INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})

            LIST(APPEND MPI_LIBRARY "${MPI_C_LIBRARIES}")
            LIST(APPEND MPI_LIBRARY "${MPI_CXX_LIBRARIES}")
            LIST(APPEND MPI_LIBRARY "${MPI_Fortran_LIBRARIES}")
            LIST(REMOVE_DUPLICATES MPI_LIBRARY)

            FOREACH(_lib ${MPI_C_LIBRARIES};${MPI_CXX_LIBRARIES};${MPI_Fortran_LIBRARIES})
                GET_FILENAME_COMPONENT(_lib_name ${_lib} NAME_WE)
                GET_FILENAME_COMPONENT(_lib_path ${_lib} PATH   )
                STRING(REGEX REPLACE "^lib(.*)$" "\\1" _lib_flag ${_lib_name})
                IF("^${_lib_flag}$" MATCHES "^(.*)mpi(.*)$")
                    LIST(INSERT MPI_LIBRARIES 0 ${_lib_flag})
                ELSE()
                    LIST(APPEND MPI_LIBRARIES ${_lib_flag})
                ENDIF()
                LIST(APPEND MPI_LIBRARY_PATH "${_lib_path}")
                LIST(REMOVE_DUPLICATES MPI_LIBRARIES)
                LIST(REMOVE_DUPLICATES MPI_LIBRARY_PATH)
            ENDFOREACH()
            LINK_DIRECTORIES(${MPI_LIBRARY_PATH})

            FOREACH(_path ${MPI_LIBRARY_PATH})
                SET(MPI_LDFLAGS      "${MPI_LDFLAGS} -L${_path}")
            ENDFOREACH()
            FOREACH(_lib ${MPI_LIBRARIES})
                SET(MPI_LDFLAGS      "${MPI_LDFLAGS} -l${_lib}")
            ENDFOREACH()
            SET(MPI_USED_MODE "FIND")

        ENDIF(MPI_EP)
    
        IF("^${MORSE_USE_MPI}$" STREQUAL "^$")
            SET(MORSE_USE_MPI "ON" CACHE STRING
                "Enable/Disable MPI dependency (ON/OFF/<not-defined>)" FORCE)
        ENDIF()
        SET(HAVE_MPI ON)
    
    ELSE()
        LINK_DIRECTORIES(${${_UPPERPCS_PACKAGE}_LIBRARY_PATH})
        INCLUDE_DIRECTORIES(${${_UPPERPCS_PACKAGE}_INCLUDE_PATH})
        IF("^${MORSE_USE_${_UPPERPCS_PACKAGE}}$" STREQUAL "^$")
            SET(MORSE_USE_${_UPPERPCS_PACKAGE} "ON" CACHE STRING
                "Enable/Disable ${_UPPERPCS_PACKAGE} dependency (ON/OFF/<not-defined>)" FORCE)
        ENDIF()
        SET(HAVE_${_UPPERPCS_PACKAGE} ON)

    ENDIF()

    # Fill the following variables
    #   - ${_UPPERPCS_PACKAGE}_LDFLAGS
    #   - ${_UPPERPCS_PACKAGE}_LIST_LDFLAGS
    # -------------------------------------
    #UNSET(${_UPPERPCS_PACKAGE}_LDFLAGS)
    UNSET(${_UPPERPCS_PACKAGE}_LIST_LDFLAGS)
    FOREACH(_path ${${_dep}_LIBRARY_PATH})
        LIST(APPEND ${_UPPERPCS_PACKAGE}_LIST_LDFLAGS "-L${_path}")
    ENDFOREACH()
    FOREACH(_lib ${${_dep}_LIBRARIES})
        LIST(APPEND ${_UPPERPCS_PACKAGE}_LIST_LDFLAGS "-l${_lib}")
    ENDFOREACH()

ENDMACRO(POPULATE_COMPILE_SYSTEM)

###
#
# FIND_AND_POPULATE_LIBRARY: Call FIND_PACKAGE then POPULATE_COMPILE_SYSTEM
#
###
MACRO(FIND_AND_POPULATE_LIBRARY _FPL_PACKAGE)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_FPL_PACKAGE}" _UPPERFPL_PACKAGE)

    # Discover the graph of dependencies
    # ----------------------------------
    DISCOVER_GRAPH_DEPENDENCIES("${_FPL_PACKAGE}" "${_FPL_PACKAGE}")

    # Looking for dependencies
    # ------------------------
    SET(${_UPPERFPL_PACKAGE}_FIND_DEPS "")
    FOREACH(_package ${${_UPPERFPL_PACKAGE}_ALL_REQUESTED})
        IF(NEED_${_package} AND NOT ${_package}_EP)
            # Debug message
            IF(MORSE_DEBUG_CMAKE)
                MESSAGE(STATUS "Looking for ${_UPPERFPL_PACKAGE} - ${_UPPERFPL_PACKAGE} requires ${_package}")
            ENDIF(MORSE_DEBUG_CMAKE)

            # Put _UPPERFPL_PACKAGE in FAPL_MEMORY for recursivity
            LIST(APPEND FAPL_MEMORY ${_UPPERFPL_PACKAGE})

            # Looking for _package
            FIND_PACKAGE(${_package} QUIET)

            # Get back _UPPERFPL_PACKAGE from FAPL_MEMORY
            LIST(LENGTH FAPL_MEMORY FAPL_SIZE)
            MATH(EXPR FAPL_ID "${FAPL_SIZE}-1")
            LIST(GET FAPL_MEMORY ${FAPL_ID} _UPPERFPL_PACKAGE)
            LIST(REMOVE_AT FAPL_MEMORY ${FAPL_ID})

            # Populate for _package
            IF(${_package}_FOUND)
                POPULATE_COMPILE_SYSTEM("${_package}")
                LIST(APPEND ${_UPPERFPL_PACKAGE}_FIND_DEPS ${_package})
            ENDIF()

        ENDIF()
    ENDFOREACH()

ENDMACRO(FIND_AND_POPULATE_LIBRARY)

##
## @end file populatePACKAGE.cmake 
##
