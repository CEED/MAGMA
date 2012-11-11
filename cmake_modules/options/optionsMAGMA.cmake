###     
#
#  @file optionsMAGMA.cmake
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

###
#
# Extra options if MAGMA is ON
#
###
MACRO(OPTIONS_MAGMA)

    OPTION(MAGMA_USE_PLASMA 
           "Enable/Disable kernels using plasma-like kernels in MAGMA" OFF)
    OPTION(MAGMA_USE_FERMI   
           "Switch between Tesla (OFF) and Fermi (ON) cards" ON)
    #OPTION(MAGMA_MGPUS_STATIC
    #       "Enable/Disable compilation of MAGMA multi-GPUs static" OFF)

ENDMACRO(OPTIONS_MAGMA)

##
## @end file optionsMAGMA.cmake
##
