###
#
#  @file installEIGEN.cmake
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

MACRO(INSTALL_EIGEN _MODE)

    # Message about refblas
    # ---------------------
    MESSAGE(STATUS "Installing BLAS - eigen version")

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(BLAS_PATH ${CMAKE_INSTALL_PREFIX}/eigen)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(BLAS_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Define options
    # --------------
    UNSET(BLAS_CONFIG_OPTS)
    LIST(APPEND BLAS_CONFIG_OPTS -DCMAKE_INSTALL_PREFIX=${BLAS_PATH})
    LIST(APPEND BLAS_CONFIG_OPTS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
    LIST(APPEND BLAS_CONFIG_OPTS -DEIGEN_BUILD_PKGCONFIG=OFF)
    IF(NOT BUILD_SHARED_LIBS)
        LIST(APPEND BLAS_CONFIG_OPTS -DBUILD_SHARED_LIBS=OFF)
    ENDIF()
    
    # Define steps of installation
    # ----------------------------
    SET(BLAS_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/blas)
    SET(BLAS_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/blas/build)
    SET(BLAS_CONFIG_CMD  ${CMAKE_COMMAND} ../)
    SET(BLAS_BUILD_CMD   ${CMAKE_MAKE_PROGRAM} blas lapack)
    SET(BLAS_INSTALL_CMD ${CMAKE_COMMAND} -E make_directory ${BLAS_PATH})

    # Define what library we will use
    # -------------------------------
    IF(BUILD_SHARED_LIBS)
        SET(eigen_lib_to_install "libeigen_blas;libeigen_lapack")
    ELSE(BUILD_SHARED_LIBS)
        SET(eigen_lib_to_install "libeigen_blas_static;libeigen_lapack_static")
    ENDIF(BUILD_SHARED_LIBS)

    # Define additional step (build)
    # ------------------------------
    UNSET(BLAS_ADD_BUILD_STEP)
    FOREACH(_task build)
        LIST(APPEND BLAS_ADD_BUILD_STEP blas_create_${_task}_path)
        SET(blas_create_${_task}_path_CMD ${CMAKE_COMMAND} -E make_directory ${_task})
        SET(blas_create_${_task}_path_DIR ${CMAKE_BINARY_DIR}/externals/blas)
        SET(blas_create_${_task}_path_DEP_BEFORE download)
        SET(blas_create_${_task}_path_DEP_AFTER  configure)
    ENDFOREACH()

    # Define additional step (install)
    # --------------------------------
    UNSET(BLAS_ADD_INSTALL_STEP)
    FOREACH(_task lib)
        LIST(APPEND BLAS_ADD_INSTALL_STEP blas_create_${_task}_path)
        SET(blas_create_${_task}_path_CMD ${CMAKE_COMMAND} -E make_directory ${BLAS_PATH}/${_task})
        SET(blas_create_${_task}_path_DIR ${CMAKE_INSTALL_PREFIX})
    ENDFOREACH()
    FOREACH(_task ${eigen_lib_to_install})
        STRING(REPLACE "libeigen_" "" _dir "${_task}")
        STRING(REPLACE "_static" "" _dir "${_dir}")
        LIST(APPEND BLAS_ADD_INSTALL_STEP blas_copy_${_task})
        SET(blas_copy_${_task}_CMD ${CMAKE_COMMAND} -E copy
                                     ${BLAS_BUILD_PATH}/${_dir}/${_task}.${MORSE_LIBRARY_EXTENSION} .)
        SET(blas_copy_${_task}_DIR ${BLAS_PATH}/lib)
    ENDFOREACH()

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("blas" "${BLAS_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(BLAS_VENDOR              "eigen"                    )
    SET(BLAS_LIBRARY_PATH        "${BLAS_BUILD_PATH}/blas"  )
    SET(EIGENLAPACK_LIBRARY_PATH "${BLAS_BUILD_PATH}/lapack")

    IF(BUILD_SHARED_LIBS)
        SET(BLAS_LIBRARY      "${BLAS_LIBRARY_PATH}/libeigen_blas.${MORSE_LIBRARY_EXTENSION}")
        SET(BLAS_LDFLAGS      "-L${BLAS_LIBRARY_PATH} -leigen_blas"                          )
        SET(BLAS_LIBRARIES    "eigen_blas"                                                   )

        SET(EIGENLAPACK_LIBRARY   "${EIGENLAPACK_LIBRARY_PATH}/libeigen_lapack.${MORSE_LIBRARY_EXTENSION}")
        SET(EIGENLAPACK_LDFLAGS   "-L${EIGENLAPACK_LIBRARY_PATH} -leigen_lapack"                          )
        SET(EIGENLAPACK_LIBRARIES "eigen_lapack"                                                          )

    ELSE()
        SET(BLAS_LIBRARY   "${BLAS_LIBRARY_PATH}/libeigen_blas_static.${MORSE_LIBRARY_EXTENSION}")
        SET(BLAS_LDFLAGS   "-L${BLAS_LIBRARY_PATH} -leigen_blas_static"                          )
        SET(BLAS_LIBRARIES "eigen_blas_static"                                                   )

        SET(EIGENLAPACK_LIBRARY   "${EIGENLAPACK_LIBRARY_PATH}/libeigen_lapack_static.${MORSE_LIBRARY_EXTENSION}")
        SET(EIGENLAPACK_LDFLAGS   "-L${EIGENLAPACK_LIBRARY_PATH} -leigen_lapack_static"                          )
        SET(EIGENLAPACK_LIBRARIES "eigen_lapack_static"                                                          )

    ENDIF()

    IF(HAVE_STDCPP)
        SET(BLAS_LDFLAGS           "${BLAS_LDFLAGS} -lstdc++")
        LIST(APPEND BLAS_LIBRARY   "${STDCPP_LIBRARY}"       )
        LIST(APPEND BLAS_LIBRARIES "stdc++"                  )

        SET(EIGENLAPACK_LDFLAGS           "${EIGENLAPACK_LDFLAGS} -lstdc++")
        LIST(APPEND EIGENLAPACK_LIBRARY   "${STDCPP_LIBRARY}"              )
        LIST(APPEND EIGENLAPACK_LIBRARIES "stdc++"                         )
     ENDIF(HAVE_STDCPP)

ENDMACRO(INSTALL_EIGEN)

##
## @end file installEIGEN.cmake
##
