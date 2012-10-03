###
#
# @file          : infoPLASMA.cmake
#
# @description   :
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 04-04-2012
# @last modified : mer. 16 mai 2012 10:20:16 CEST
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
    IF(NOT DEFINED PLASMA_SVN_REP)
        SET(PLASMA_REPO_MODE "SVN")
        SET(PLASMA_SVN_REP   ""   )
        SET(PLASMA_SVN_ID    ""   )
        SET(PLASMA_SVN_PWD   ""   )
    ENDIF()

   # Define dependencies
   # -------------------
   SET(PLASMA_DEPENDENCIES "hwloc;blas;lapack;cblas;lapacke")

ENDMACRO(PLASMA_INFO_INSTALL)

MACRO(PLASMA_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(PLASMA_type_library        "C;Fortran"                          )
    SET(PLASMA_name_library        "plasma;coreblas;quark"              )
    SET(PLASMA_name_pkgconfig      "plasma"                             )
    # Problem with core_blas.h ==> plasma.h need to be included in core_blas.h (type PLASMA_Complex64_t not defined)
    # SET(PLASMA_name_include        "plasma.h;core_blas.h;quark.h"       )
    SET(PLASMA_name_include        "plasma.h;quark.h"       )
    SET(PLASMA_name_include_suffix "PLASMA_name_include_suffix-NOTFOUND")
    SET(PLASMA_name_fct_test       "PLASMA_dgetrf"                      )
    SET(PLASMA_name_binary         "PLASMA_name_binary-NOTFOUND"        )

ENDMACRO(PLASMA_INFO_FIND)

###
### END infoPLASMA.cmake
###
