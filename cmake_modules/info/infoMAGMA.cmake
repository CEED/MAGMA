###
#
#  @file infoMAGMA.cmake 
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
MACRO(MAGMA_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(MAGMA_DIRECT_DEPS         "CUDA"  )
    LIST(APPEND MAGMA_DIRECT_DEPS "LAPACK")
    LIST(APPEND MAGMA_DIRECT_DEPS "CBLAS" )
    LIST(APPEND MAGMA_DIRECT_DEPS "TMG"   )

    # Define the priority of dependencies
    # -----------------------------------
    INCLUDE(BuildSystemTools)
    DEFINE_PRIORITY("MAGMA" "LAPACK" "depends" "depends"  "depends")
    DEFINE_PRIORITY("MAGMA" "CBLAS"  "depends" "depends"  "depends")
    DEFINE_PRIORITY("MAGMA" "CUDA"   "depends" "depends"  "depends")

    IF(MORSE_ENABLE_TESTING)
        DEFINE_PRIORITY("MAGMA_MORSE" "TMG"   "depends" "depends"  "depends"   )
    ELSE(MORSE_ENABLE_TESTING)
        DEFINE_PRIORITY("MAGMA_MORSE" "TMG"   "depends" "depends"  "recommends")
    ENDIF(MORSE_ENABLE_TESTING)

ENDMACRO(MAGMA_INFO_DEPS)

###
#
#
#
###
MACRO(MAGMA_INFO_INSTALL)

    # Define web link of magma
    # ------------------------
    IF(NOT DEFINED MAGMA_URL)
        #SET(MAGMA_URL     "")
    ENDIF()

    # Define tarball of magma
    # -----------------------
    IF(NOT DEFINED MAGMA_TARBALL)
        #SET(MAGMA_TARBALL "")
    ENDIF()

    # Define md5sum of magma
    # ----------------------
    IF(DEFINED MAGMA_URL OR DEFINED MAGMA_TARBALL)
        #SET(MAGMA_MD5SUM  "")
    ENDIF()

    # Define repository of fxt
    # ------------------------
    IF(NOT DEFINED MAGMA_REPO_URL)
        SET(MAGMA_REPO_MODE "SVN")
        SET(MAGMA_REPO_URL  ""   )
        SET(MAGMA_REPO_ID   ""   )
        SET(MAGMA_REPO_PWD  ""   )
    ENDIF()

ENDMACRO(MAGMA_INFO_INSTALL)

##
## @end file infoMAGMA.cmake 
##
