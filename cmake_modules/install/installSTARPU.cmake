###
#
# @file      : installSTARPU.cmake
#
# @description   : Project MORSE (http://hiepacs.bordeaux.inria.fr/eagullo/morse)
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 19-01-2012
# @last modified : Tue 10 Jul 2012 09:46:10 AM CEST
#
###

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE(installExternalPACKAGE)
INCLUDE(downloadPACKAGE)
INCLUDE(update_env)
INCLUDE(infoSTARPU)

MACRO(INSTALL_STARPU _MODE)

    # Get info for this package
    # -------------------------
    STARPU_INFO_INSTALL()

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(STARPU_PATH ${CMAKE_INSTALL_PREFIX}/starpu)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(STARPU_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Define options
    # --------------
    SET(_starpu_list_flags "")
    SET(STARPU_OPTIONS --prefix=${STARPU_PATH})
    IF(HAVE_MPI)
        SET(STARPU_OPTIONS ${STARPU_OPTIONS} --with-mpicc=${MPI_C_COMPILER})
        LIST(APPEND _starpu_list_flags "starpumpi")
        LIST(APPEND _starpu_list_flags "starpu")
        SET(_starpu_string_flags "-lstarpumpi -lstarpu")
    ELSE(HAVE_MPI)
        SET(STARPU_OPTIONS ${STARPU_OPTIONS} --without-mpicc)
        LIST(APPEND _starpu_list_flags "starpu")
        SET(_starpu_string_flags "-lstarpu")
    ENDIF(HAVE_MPI)

    IF(HAVE_CUDA)
        SET(STARPU_OPTIONS ${STARPU_OPTIONS} --with-cuda-dir=${CUDA_TOOLKIT_ROOT_DIR})
        IF(${ARCH_X86_64})
            SET(STARPU_OPTIONS ${STARPU_OPTIONS} --with-cuda-lib-dir=${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        ELSE(${ARCH_X86_64})
            SET(STARPU_OPTIONS ${STARPU_OPTIONS} --with-cuda-lib-dir=${CUDA_TOOLKIT_ROOT_DIR}/lib)
        ENDIF(${ARCH_X86_64})
        
        IF(${HAVE_STDCPP})
            LIST(APPEND _starpu_list_flags "cudart" "cublas" "cuda" "stdc++")
            SET(_starpu_string_flags "${_starpu_string_flags} -lcudart -lcublas -lcuda -lstdc++")
        ELSE(${HAVE_STDCPP})
            LIST(APPEND _starpu_list_flags "cudart" "cublas" "cuda")
            SET(_starpu_string_flags "${_starpu_string_flags} -lcudart -lcublas -lcuda")
        ENDIF(${HAVE_STDCPP})

    ELSE(HAVE_CUDA)
        SET(STARPU_OPTIONS ${STARPU_OPTIONS} --disable-cuda)
    ENDIF(HAVE_CUDA)

    IF(HAVE_FXT)
        UPDATE_ENV(PKG_CONFIG_PATH ${FXT_LIBRARY_PATH}/pkgconfig)
        SET(STARPU_OPTIONS ${STARPU_OPTIONS} --with-fxt=${FXT_PATH})
    ENDIF(HAVE_FXT)

    IF(HAVE_HWLOC)
        UPDATE_ENV(PKG_CONFIG_PATH ${HWLOC_LIBRARY_PATH}/pkgconfig)
        SET(STARPU_OPTIONS ${STARPU_OPTIONS} --with-hwloc=${HWLOC_PATH})
        LIST(APPEND _starpu_list_flags "hwloc")
        SET(_starpu_string_flags "${_starpu_string_flags} -lhwloc")
    ENDIF(HAVE_HWLOC)

    SET(_ADD_PARAMETERS "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}")

    # Define steps of installation
    # ----------------------------
    SET(STARPU_CONFIG_CMD ${_ADD_PARAMETERS} ./configure)
    SET(STARPU_MAKE_CMD ${CMAKE_MAKE_PROGRAM})
    SET(STARPU_MAKEINSTALL_CMD ${CMAKE_MAKE_PROGRAM} install)

    # Install the external package
    # ----------------------------
    DEFINE_DOWNLOAD_PACKAGE("starpu" "${_MODE}")
    INSTALL_EXTERNAL_PACKAGE("starpu" "${STARPU_BUILD_MODE}")

    # Info of STARPU version:
    # -----------------------
    STRING(REGEX REPLACE "starpu-([0-9].[0-9].[0-9]).*.tar.gz$" "\\1" STARPU_VERSION "${STARPU_USED_FILE}")
    STRING(REGEX REPLACE "starpu-([0-9].[0-9]).*.tar.gz$" "\\1" STARPU_VERSION_PATH "${STARPU_USED_FILE}")
    MARK_AS_ADVANCED(STARPU_VERSION_PATH)

    # Set linker flags
    # ----------------
    SET(STARPU_BINARY_PATH ${STARPU_PATH}/bin)
    SET(STARPU_LIBRARY_PATH ${STARPU_PATH}/lib)
    IF("0.9.9" LESS "${STARPU_VERSION}")
        STRING(REGEX REPLACE "starpumpi" "starpumpi-${STARPU_VERSION_PATH}"
               _starpu_string_flags "${_starpu_string_flags}")
        STRING(REGEX REPLACE "starpumpi" "starpumpi-${STARPU_VERSION_PATH}"
               _starpu_list_flags "${_starpu_list_flags}")
        STRING(REGEX REPLACE "starpu([^m])" "starpu-${STARPU_VERSION_PATH}\\1"
               _starpu_string_flags "${_starpu_string_flags}")
        STRING(REGEX REPLACE "starpu([^m])" "starpu-${STARPU_VERSION_PATH}\\1"
               _starpu_list_flags "${_starpu_list_flags}")
        SET(STARPU_INCLUDE_PATH ${STARPU_PATH}/include/starpu/${STARPU_VERSION_PATH})
    ELSE()
        SET(STARPU_INCLUDE_PATH ${STARPU_PATH}/include)
    ENDIF()
    SET(STARPU_LDFLAGS "-L${STARPU_LIBRARY_PATH} ${_starpu_string_flags}")
    SET(STARPU_LIBRARIES "${_starpu_list_flags}")

    # Force to use deprecated API
    # ---------------------------
    #IF("0.9.9" LESS "${STARPU_VERSION}")
    #    SET(STARPU_NEED_DEPRECATED_API ON)
    #ELSE()
    #    SET(STARPU_NEED_DEPRECATED_API OFF)
    #ENDIF()

ENDMACRO(INSTALL_STARPU)

###
### END installSTARPU.cmake
###

