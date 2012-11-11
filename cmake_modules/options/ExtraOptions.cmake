###     
#
#  @file ExtraOptions.cmake
#
#  @project MORSE
#  MORSE is a software package provided by:
#     Inria Bordeaux - Sud-Ouest,
#     Univ. of Tennessee,
#     Univ. of California Berkeley,
#     Univ. of Colorado Denver.
#
#  @version 0.2.0
#  @author Cedric Castagnede
#  @date 13-07-2012
#
###

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

MACRO(EXTRA_OPTIONS)

    # Extra options if MAGMA is ON
    # ----------------------------
    IF(MAGMA)
        INCLUDE(optionsMAGMA)
        OPTIONS_MAGMA()
    ENDIF(MAGMA)  

ENDMACRO(EXTRA_OPTIONS)

##
## @end file ExtraOptions.cmake
##
