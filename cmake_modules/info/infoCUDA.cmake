###
#
#  @file infoCUDA.cmake 
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
MACRO(CUDA_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(CUDA_DIRECT_DEPS "")

ENDMACRO(CUDA_INFO_DEPS)

###
#
#
#
###
MACRO(CUDA_INFO_INSTALL)

    # Define web link of cuda
    # -----------------------
    IF(NOT DEFINED CUDA_URL)
        #SET(CUDA_URL     "")
    ENDIF()

    # Define tarball of cuda
    # ----------------------
    IF(NOT DEFINED CUDA_TARBALL)
        #SET(CUDA_TARBALL "")
    ENDIF()

    # Define md5sum of cuda
    # ---------------------
    IF(DEFINED CUDA_URL OR DEFINED CUDA_TARBALL)
        #SET(CUDA_MD5SUM  "")
    ENDIF()

ENDMACRO(CUDA_INFO_INSTALL)

##
## @end file infoCUDA.cmake 
##
