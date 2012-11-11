###
#
#  @file infoTMG.cmake
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
MACRO(TMG_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(TMG_DIRECT_DEPS     "LAPACK" )

    # Define the priority of dependencies
    # -----------------------------------
    SET(TMG_LAPACK_PRIORITY "depends")
    
ENDMACRO(TMG_INFO_DEPS)

###
#   
#   
#   
###     
MACRO(TMG_INFO_INSTALL)
    # Define web link of lapacke
    # --------------------------
    IF(NOT DEFINED TMG_URL)
        SET(TMG_URL "http://www.netlib.org/lapack/lapack.tgz")
    ENDIF()

    # Define tarball of lapacke
    # -------------------------
    IF(NOT DEFINED TMG_TARBALL)
        SET(TMG_TARBALL "lapack.tgz")
    ENDIF()

    # Define md5sum of lapacke
    # ------------------------
    IF(DEFINED TMG_URL OR DEFINED TMG_TARBALL)
        SET(TMG_MD5SUM "61bf1a8a4469d4bdb7604f5897179478")
    ENDIF()

    # Define repository of lapacke
    # ----------------------------
    IF(NOT DEFINED TMG_REPO_URL)
        SET(TMG_REPO_MODE "SVN"                                               )
        SET(TMG_REPO_URL  "https://icl.cs.utk.edu/svn/lapack-dev/lapack/trunk")
        SET(TMG_REPO_ID   ""                                                  )
        SET(TMG_REPO_PWD  ""                                                  )
    ENDIF()

ENDMACRO(TMG_INFO_INSTALL)

###
#   
#   
#   
###     
MACRO(TMG_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    UNSET(TMG_name_fct_test)
    SET(TMG_type_library        "C;Fortran"                       )
    SET(TMG_name_library        "tmg"                             )
    SET(TMG_name_include        "TMG_name_include-NOTFOUND"       )
    SET(TMG_name_pkgconfig      "TMG_name_pkgconfig-NOTFOUND"     )
    SET(TMG_name_include_suffix "TMG_name_include_suffix-NOTFOUND")
    SET(TMG_name_binary         "TMG_name_binary-NOTFOUND"        )
    IF(BUILD_SINGLE)
        LIST(APPEND TMG_name_fct_test "slatms"                    )
    ENDIF()
    IF(BUILD_DOUBLE)
        LIST(APPEND TMG_name_fct_test "dlatms"                    )
    ENDIF()
    IF(BUILD_COMPLEX)
        LIST(APPEND TMG_name_fct_test "clatms"                    )
    ENDIF()
    IF(BUILD_COMPLEX16)
        LIST(APPEND TMG_name_fct_test "zlatms"                    )
    ENDIF()

ENDMACRO(TMG_INFO_FIND)

##
## @end file infoTMG.cmake
##
