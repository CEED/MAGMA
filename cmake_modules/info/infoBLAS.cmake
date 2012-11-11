###
#
#  @file infoBLAS.cmake
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
#
#
###
MACRO(BLAS_DEFINE_DEFAULT)

    STRING(TOUPPER "${MORSE_USE_BLAS}" VALUE_MORSE_USE_BLAS)
    IF("${VALUE_MORSE_USE_BLAS}" MATCHES "EIGEN")
        SET(MORSE_BLAS_DEFAULT_VALUE "EIGEN")

    ELSEIF("${VALUE_MORSE_USE_BLAS}" MATCHES "REFBLAS")
        SET(MORSE_BLAS_DEFAULT_VALUE "REFBLAS")

    ELSE()
        IF(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
            SET(MORSE_BLAS_DEFAULT_VALUE "REFBLAS")
        ELSE()
            IF(MAGMA_MORSE)
                SET(MORSE_BLAS_DEFAULT_VALUE "EIGEN")
            ELSE(MAGMA_MORSE)
                SET(MORSE_BLAS_DEFAULT_VALUE "REFBLAS")
            ENDIF(MAGMA_MORSE)
        ENDIF()

    ENDIF()

ENDMACRO(BLAS_DEFINE_DEFAULT)

###
#
#
#
###
MACRO(BLAS_INFO_DEPS)

    # Get the implementation of BLAS
    # ------------------------------
    BLAS_DEFINE_DEFAULT()
    INFO_DEPS_PACKAGE(${MORSE_BLAS_DEFAULT_VALUE})

    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(BLAS_DIRECT_DEPS "${${MORSE_BLAS_DEFAULT_VALUE}_DIRECT_DEPS}")
    FOREACH(_dep ${BLAS_DIRECT_DEPS})
        SET(BLAS_${_dep}_PRIORITY "${${MORSE_BLAS_DEFAULT_VALUE}_${_dep}_PRIORITY}")
    ENDFOREACH()

ENDMACRO(BLAS_INFO_DEPS)

###
#
#
#
###
MACRO(BLAS_INFO_INSTALL)

    # Get the implementation of BLAS
    # ------------------------------
    BLAS_DEFINE_DEFAULT()
    INFO_INSTALL_PACKAGE(${MORSE_BLAS_DEFAULT_VALUE})

    # Define web link of blas
    # -----------------------
    IF(NOT DEFINED BLAS_URL)
        SET(BLAS_URL    ${${MORSE_BLAS_DEFAULT_VALUE}_URL})
    ENDIF()

    # Define tarball of blas
    # ----------------------
    IF(NOT DEFINED BLAS_TARBALL)
        SET(BLAS_TARBALL ${${MORSE_BLAS_DEFAULT_VALUE}_TARBALL})
    ENDIF()

    # Define md5sum of blas
    # ---------------------
    IF(DEFINED BLAS_URL OR DEFINED BLAS_TARBALL)
        SET(BLAS_MD5SUM  ${${MORSE_BLAS_DEFAULT_VALUE}_MD5SUM})
    ENDIF()

    # Define repository of blas
    # -------------------------
    IF(NOT DEFINED BLAS_REPO_URL)
        SET(BLAS_REPO_MODE  ${${MORSE_BLAS_DEFAULT_VALUE}_REPO_MODE})
        SET(BLAS_REPO_URL   ${${MORSE_BLAS_DEFAULT_VALUE}_REPO_URL} )
        SET(BLAS_REPO_ID    ${${MORSE_BLAS_DEFAULT_VALUE}_REPO_ID}  )
        SET(BLAS_REPO_PWD   ${${MORSE_BLAS_DEFAULT_VALUE}_REPO_PWD} )
    ENDIF()

ENDMACRO(BLAS_INFO_INSTALL)

###
#
#
#
###
MACRO(BLAS_INFO_FIND)

    # Get the implementation of BLAS
    # ------------------------------
    BLAS_DEFINE_DEFAULT()
    INFO_FIND_PACKAGE(${MORSE_BLAS_DEFAULT_VALUE})

    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(BLAS_type_library        "${${MORSE_BLAS_DEFAULT_VALUE}_type_library}"       )
    SET(BLAS_name_library        "${${MORSE_BLAS_DEFAULT_VALUE}_name_library}"       )
    SET(BLAS_name_pkgconfig      "${${MORSE_BLAS_DEFAULT_VALUE}_name_pkgconfig}"     )
    SET(BLAS_name_include        "${${MORSE_BLAS_DEFAULT_VALUE}_name_include}"       )
    SET(BLAS_name_include_suffix "${${MORSE_BLAS_DEFAULT_VALUE}_name_include_suffix}")
    SET(BLAS_name_binary         "${${MORSE_BLAS_DEFAULT_VALUE}_name_binary}"        )
    SET(BLAS_name_fct_test       "${${MORSE_BLAS_DEFAULT_VALUE}_name_fct_test}"      )

ENDMACRO(BLAS_INFO_FIND)

##
## @end file infoBLAS.cmake
##
