###
#
#  @file infoPLASMA.cmake
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
MACRO(PLASMA_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(PLASMA_DIRECT_DEPS         "LAPACKE")
    LIST(APPEND PLASMA_DIRECT_DEPS "CBLAS"  )
    LIST(APPEND PLASMA_DIRECT_DEPS "QUARK"  )
    LIST(APPEND PLASMA_DIRECT_DEPS "HWLOC"  )

    # Define the priority of dependencies
    # -----------------------------------
    SET(PLASMA_LAPACKE_PRIORITY    "depends")
    SET(PLASMA_CBLAS_PRIORITY      "depends")
    SET(PLASMA_HWLOC_PRIORITY      "depends")
    SET(PLASMA_QUARK_PRIORITY      "depends")

ENDMACRO(PLASMA_INFO_DEPS)

###
#   
#   
#   
###     
MACRO(PLASMA_INFO_INSTALL)
    # Define web link of plasma
    # -------------------------
    IF(NOT DEFINED PLASMA_URL)
        SET(PLASMA_URL     "http://icl.cs.utk.edu/projectsfiles/plasma/pubs/plasma_2.4.6.tar.gz")
    ENDIF()

    # Define tarball of plasma
    # ------------------------
    IF(NOT DEFINED PLASMA_TARBALL)
        SET(PLASMA_TARBALL "plasma_2.4.6.tar.gz")
    ENDIF()

    # Define md5sum of plasma
    # -----------------------
    IF(DEFINED PLASMA_URL OR DEFINED PLASMA_TARBALL)
        SET(PLASMA_MD5SUM  "95c6e145636bbbdabf6b9c4ecb5ca2a7")
    ENDIF()

    # Define repository of plasma
    # ---------------------------
    IF(NOT DEFINED PLASMA_REPO_URL)
        SET(PLASMA_REPO_MODE "SVN")
        SET(PLASMA_REPO_URL   ""  )
        SET(PLASMA_REPO_ID    ""  )
        SET(PLASMA_REPO_PWD   ""  )
    ENDIF()

ENDMACRO(PLASMA_INFO_INSTALL)

###
#   
#   
#   
###     
MACRO(PLASMA_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(PLASMA_type_library        "C;Fortran"                          )
    SET(PLASMA_name_library        "plasma;coreblas;quark"              )
    SET(PLASMA_name_pkgconfig      "plasma"                             )
    SET(PLASMA_name_include        "plasma.h;core_blas.h;quark.h"       )
    SET(PLASMA_name_include_suffix "PLASMA_name_include_suffix-NOTFOUND")
    SET(PLASMA_name_binary         "PLASMA_name_binary-NOTFOUND"        )
    IF(BUILD_SINGLE)
        LIST(APPEND PLASMA_name_fct_test "PLASMA_sgetrf"                )
    ENDIF()
    IF(BUILD_DOUBLE)
        LIST(APPEND PLASMA_name_fct_test "PLASMA_dgetrf"                )
    ENDIF()
    IF(BUILD_COMPLEX)
        LIST(APPEND PLASMA_name_fct_test "PLASMA_cgetrf"                )
    ENDIF()
    IF(BUILD_COMPLEX16)
        LIST(APPEND PLASMA_name_fct_test "PLASMA_zgetrf"                )
    ENDIF()

ENDMACRO(PLASMA_INFO_FIND)

##
## @end file infoPLASMA.cmake
##
