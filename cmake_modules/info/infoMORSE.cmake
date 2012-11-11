###
#
#  @file infoMORSE.cmake 
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
MACRO(MORSE_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    UNSET(MORSE_DIRECT_DEPS)
    IF(MAGMA_MORSE)
        LIST(APPEND MORSE_DIRECT_DEPS "MAGMA_MORSE")
    ENDIF(MAGMA_MORSE)
    IF(MAGMA)
        LIST(APPEND MORSE_DIRECT_DEPS "MAGMA")
    ENDIF(MAGMA)

    # Define the priority of dependencies
    # -----------------------------------
    SET(MORSE_MAGMA_PRIORITY         "depends")
    SET(MORSE_MAGMA_MORSE_PRIORITY   "depends")

ENDMACRO(MORSE_INFO_DEPS)


##
## @end file infoMORSE.cmake 
##
