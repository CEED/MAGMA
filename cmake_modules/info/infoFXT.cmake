###
#
#  @file infoFXT.cmake
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
MACRO(FXT_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(FXT_DIRECT_DEPS "")

ENDMACRO(FXT_INFO_DEPS)

###
#
#
#
###
MACRO(FXT_INFO_INSTALL)

    # Define web link of fxt
    # ----------------------
    IF(NOT DEFINED FXT_URL)
        SET(FXT_URL     "http://download.savannah.gnu.org/releases/fkt/fxt-0.2.10.tar.gz")
    ENDIF()

    # Define tarball of fxt
    # ---------------------
    IF(NOT DEFINED FXT_TARBALL)
        SET(FXT_TARBALL "fxt-0.2.10.tar.gz")
    ENDIF()

    # Define md5sum of fxt
    # --------------------
    IF(DEFINED FXT_URL OR DEFINED FXT_TARBALL)
        SET(FXT_MD5SUM  "c23251dc829aa268c0da54a5cf051442")
    ENDIF()

    # Define repository of fxt
    # ------------------------
    IF(NOT DEFINED FXT_REPO_URL)
        SET(FXT_REPO_MODE "CVS"                                 )
        SET(FXT_CVS_REP   "cvs.savannah.nongnu.org:/sources/fkt")
        SET(FXT_CVS_MOD   "fkt"                                 )

    ENDIF()

ENDMACRO(FXT_INFO_INSTALL)

###
#
#
#
###
MACRO(FXT_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(FXT_type_library        "C"                               )
    SET(FXT_name_library        "fxt"                             )
    SET(FXT_name_pkgconfig      "fxt"                             )
    SET(FXT_name_include        "fxt.h"                           )
    SET(FXT_name_include_suffix "FXT_name_include_suffix-NOTFOUND")
    SET(FXT_name_fct_test       "fxt_infos"                       )
    SET(FXT_name_binary         "FXT_name_binary-NOTFOUND"        )
    #SET(FXT_name_binary        "fxt_print"                       )

ENDMACRO(FXT_INFO_FIND)

##
## @end file infoFXT.cmake
##
