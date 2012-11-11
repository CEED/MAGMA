###
#
#  @file installSTARPU.cmake
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
INCLUDE(BuildSystemTools)
INCLUDE(installExternalPACKAGE)

MACRO(INSTALL_STARPU _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(STARPU_PATH ${CMAKE_INSTALL_PREFIX}/starpu)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(STARPU_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Define options
    # --------------
    UNSET(STARPU_POST_OPTS)
    UNSET(STARPU_PRE_OPTS)

    LIST(APPEND STARPU_POST_OPTS --prefix=${STARPU_PATH})
    SET(STARPU_PRE_OPTS "${STARPU_PRE_OPTS} CC=\"${CMAKE_C_COMPILER}\"")
    SET(STARPU_PRE_OPTS "${STARPU_PRE_OPTS} CXX=\"${CMAKE_CXX_COMPILER}\"")

    IF(HAVE_MPI)
        LIST(APPEND STARPU_POST_OPTS --with-mpicc=${MPI_C_COMPILER})
    ELSE(HAVE_MPI)
        LIST(APPEND STARPU_POST_OPTS --without-mpicc)
    ENDIF(HAVE_MPI)

    IF(HAVE_CUDA)
        LIST(APPEND STARPU_POST_OPTS --with-cuda-dir=${CUDA_TOOLKIT_ROOT_DIR})
        IF(${ARCH_X86_64})
            LIST(APPEND STARPU_POST_OPTS --with-cuda-lib-dir=${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        ELSE(${ARCH_X86_64})
            LIST(APPEND STARPU_POST_OPTS --with-cuda-lib-dir=${CUDA_TOOLKIT_ROOT_DIR}/lib)
        ENDIF(${ARCH_X86_64})
    ELSE(HAVE_CUDA)
        LIST(APPEND STARPU_POST_OPTS --disable-cuda)
    ENDIF(HAVE_CUDA)

    IF(HAVE_OPENCL)
        #LIST(APPEND STARPU_POST_OPTS --with-opencl-dir=${OPENCL_PATH})
        LIST(APPEND STARPU_POST_OPTS --with-opencl-lib-dir=${OPENCL_LIBRARY_PATH})
        LIST(APPEND STARPU_POST_OPTS --with-opencl-include-dir=${OPENCL_INCLUDE_PATH})
    ELSE(HAVE_OPENCL)
        LIST(APPEND STARPU_POST_OPTS --disable-opencl)
    ENDIF(HAVE_OPENCL)

    IF(HAVE_FXT)
        IF(FXT_FOUND AND PKG_CONFIG_EXECUTABLE)
            UPDATE_ENV(PKG_CONFIG_PATH ${FXT_LIBRARY_PATH}/pkgconfig)
            LIST(APPEND STARPU_POST_OPTS --with-fxt=${FXT_PATH})

        ELSE()
            LIST(APPEND STARPU_POST_OPTS --with-fxt)
            SET(include_path "")
            FOREACH(_path ${FXT_INCLUDE_PATH})
                 SET(include_path "${include_path}-I${_path} ")
             ENDFOREACH()
            SET(STARPU_PRE_OPTS "${STARPU_PRE_OPTS} FXT_CFLAGS=\"${include_path}\"")
            SET(STARPU_PRE_OPTS "${STARPU_PRE_OPTS} FXT_LIBS=\"${FXT_LDFLAGS}\"")

        ENDIF()
    ENDIF(HAVE_FXT)

    IF(HAVE_HWLOC)
        IF(HWLOC_FOUND AND PKG_CONFIG_EXECUTABLE)
            UPDATE_ENV(PKG_CONFIG_PATH ${HWLOC_LIBRARY_PATH}/pkgconfig)
            LIST(APPEND STARPU_POST_OPTS --with-hwloc=${HWLOC_PATH})

        ELSE()
            LIST(APPEND STARPU_POST_OPTS --with-hwloc)
            SET(include_path "")
            FOREACH(_path ${HWLOC_INCLUDE_PATH})
                SET(include_path "${include_path}-I${_path} ")
            ENDFOREACH()
            SET(STARPU_PRE_OPTS "${STARPU_PRE_OPTS} HWLOC_CFLAGS=\"${include_path}\"")
            SET(STARPU_PRE_OPTS "${STARPU_PRE_OPTS} HWLOC_LIBS=\"${HWLOC_LDFLAGS}\"")

        ENDIF()
    ENDIF(HAVE_HWLOC)

    IF(SCALFMM_MORSE)
        LIST(APPEND STARPU_POST_OPTS --enable-maxbuffers=9)
        LIST(APPEND STARPU_POST_OPTS --disable-cuda-memcpy-peer)
    ENDIF(SCALFMM_MORSE)

    IF(HAVE_DL AND HAVE_CUDA)
        SET(STARPU_PRE_OPTS "${STARPU_PRE_OPTS} LIBS=\"-ldl\"")
    ENDIF(HAVE_DL AND HAVE_CUDA)

    IF(PKG_CONFIG_EXECUTABLE)
        SET(STARPU_PRE_OPTS "${STARPU_PRE_OPTS} PKG_CONFIG_PATH=\"$ENV{PKG_CONFIG_PATH}\"")
    ENDIF(PKG_CONFIG_EXECUTABLE)

    # Define steps of installation
    # ----------------------------
    SET(STARPU_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/starpu)
    SET(STARPU_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/starpu)
    SET(STARPU_PATCH_CMD   patch -p1 -i ${CMAKE_SOURCE_DIR}/patch/starpu104_nvcc-options.patch)
    SET(STARPU_CONFIG_CMD  ./configure_starpu)
    SET(STARPU_BUILD_CMD   ${CMAKE_MAKE_PROGRAM})
    SET(STARPU_INSTALL_CMD ${CMAKE_MAKE_PROGRAM} install)

    # Write a file named configure_starpu to launch configure
    # -------------------------------------------------------
    SET(STARPU_CFG_FILE "${STARPU_PRE_OPTS} ./configure")
    FOREACH(_arg ${STARPU_POST_OPTS})
        SET(STARPU_CFG_FILE "${STARPU_CFG_FILE} ${_arg}")
    ENDFOREACH()

    # Define additional step
    # ----------------------
    UNSET(STARPU_ADD_BUILD_STEP)
    LIST(APPEND STARPU_ADD_BUILD_STEP starpu_generate_preconfigure) 
    SET(starpu_generate_preconfigure_CMD ${CMAKE_COMMAND} -E echo "${STARPU_CFG_FILE}" > ${STARPU_BUILD_PATH}/configure_starpu)
    SET(starpu_generate_preconfigure_DIR ${STARPU_BUILD_PATH}) 
    SET(starpu_generate_preconfigure_DEP_BEFORE download)
    SET(starpu_generate_preconfigure_DEP_AFTER  configure)

    LIST(APPEND STARPU_ADD_BUILD_STEP starpu_chmod_preconfigure)
    SET(starpu_chmod_preconfigure_CMD chmod +x ${STARPU_BUILD_PATH}/configure_starpu)
    SET(starpu_chmod_preconfigure_DIR ${STARPU_BUILD_PATH})
    SET(starpu_chmod_preconfigure_DEP_BEFORE starpu_generate_preconfigure)
    SET(starpu_chmod_preconfigure_DEP_AFTER  configure)

    LIST(APPEND STARPU_ADD_BUILD_STEP starpu_print_preconfigure)
    SET(starpu_print_preconfigure_CMD cat ${STARPU_BUILD_PATH}/configure_starpu)
    SET(starpu_print_preconfigure_DIR ${STARPU_BUILD_PATH})
    SET(starpu_print_preconfigure_DEP_BEFORE starpu_generate_preconfigure)
    SET(starpu_print_preconfigure_DEP_AFTER  configure)

    # Install the external package
    # ----------------------------
    DEFINE_DOWNLOAD_PACKAGE("starpu" "${_MODE}")
    INSTALL_EXTERNAL_PACKAGE("starpu" "${STARPU_USED_MODE}")

    # Info of STARPU version:
    # -----------------------
    STRING(REGEX REPLACE "starpu-([0-9].[0-9].[0-9]).*.tar.gz$" "\\1" STARPU_VERSION "${STARPU_USED_FILE}")
    STRING(REGEX REPLACE "starpu-([0-9].[0-9]).*.tar.gz$" "\\1" STARPU_VERSION_PATH "${STARPU_USED_FILE}")
    MARK_AS_ADVANCED(STARPU_VERSION_PATH)

    # Set linker flags
    # ----------------
    UNSET(STARPU_LDFLAGS)
    UNSET(STARPU_LIBRARY)
    UNSET(STARPU_LIBRARIES)
    UNSET(STARPU_BINARY_PATH)
    UNSET(STARPU_LIBRARY_PATH)
    UNSET(STARPU_INCLUDE_PATH)

    LIST(APPEND STARPU_BINARY_PATH  "${STARPU_BUILD_PATH}/tools/.libs")
    LIST(APPEND STARPU_LIBRARY_PATH "${STARPU_BUILD_PATH}/src/.libs")
    LIST(APPEND STARPU_INCLUDE_PATH "${STARPU_BUILD_PATH}/include")

    IF(HAVE_MPI)
        LIST(APPEND STARPU_LIBRARY_PATH "${STARPU_BUILD_PATH}/mpi/.libs")
        LIST(APPEND STARPU_INCLUDE_PATH "${STARPU_BUILD_PATH}/mpi")
        LIST(APPEND STARPU_LIBRARY      "${STARPU_BUILD_PATH}/mpi/.libs/libstarpumpi-${STARPU_VERSION_PATH}.${MORSE_LIBRARY_EXTENSION}")
        LIST(APPEND STARPU_LIBRARIES    "starpumpi-${STARPU_VERSION_PATH}"                                                             )
        SET(STARPU_LDFLAGS              "-L${STARPU_BUILD_PATH}/mpi/.libs -lstarpumpi-${STARPU_VERSION_PATH}"                          )

    ELSE(HAVE_MPI)
        LIST(APPEND STARPU_LIBRARY      "${STARPU_BUILD_PATH}/src/.libs/libstarpu-${STARPU_VERSION_PATH}.${MORSE_LIBRARY_EXTENSION}")
        LIST(APPEND STARPU_LDFLAGS      "-L${STARPU_BUILD_PATH}/src/.libs -lstarpu-${STARPU_VERSION_PATH}"                          )
        SET(STARPU_LIBRARIES            "starpu-${STARPU_VERSION_PATH}"                                                             )

    ENDIF(HAVE_MPI)

    IF(HAVE_CUDA)
        LIST(APPEND STARPU_LIBRARY   "${CUDA_CUDART_LIBRARY}"    )
        LIST(APPEND STARPU_LIBRARY   "${CUDA_CUBLAS_LIBRARIES}"  )
        LIST(APPEND STARPU_LIBRARY   "${CUDA_CUDA_LIBRARY}"      )
        LIST(APPEND STARPU_LIBRARIES "cudart"                    )
        LIST(APPEND STARPU_LIBRARIES "cublas"                    )
        LIST(APPEND STARPU_LIBRARIES "cuda"                      )
        SET(STARPU_LDFLAGS           "${STARPU_LDFLAGS} -lcudart")
        SET(STARPU_LDFLAGS           "${STARPU_LDFLAGS} -lcublas")
        SET(STARPU_LDFLAGS           "${STARPU_LDFLAGS} -lcuda"  )

    ENDIF(HAVE_CUDA)

    FOREACH(_dep OPENCL STDCPP FXT HWLOC)
        IF(HAVE_${_dep})
            LIST(APPEND STARPU_LIBRARY   "${${_dep}_LIBRARY}")
            LIST(APPEND STARPU_LIBRARIES "${${_dep}_LIBRARIES}")
            SET(STARPU_LDFLAGS           "${STARPU_LDFLAGS} ${${_dep}_LDFLAGS}")
        ENDIF(HAVE_${_dep})
    ENDFOREACH()

ENDMACRO(INSTALL_STARPU)

##
## @end file installSTARPU.cmake
##

