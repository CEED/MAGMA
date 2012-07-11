###
#
# @file      : CMakeLists.txt
#
# @description   : Project DPLASMA
#
# @version       : 
# @created by    : Innovative Computing Laboratory
# @creation date : 19-01-2012
# @last modified : Tue 10 Jul 2012 09:40:05 AM CEST
#
###
#
# CMAKE_EXTRA_CFLAGS
# CMAKE_EXTRA_LDFLAGS_C
# CMAKE_EXTRA_FFLAGS
# CMAKE_EXTRA_LDFLAGS_F
# CMAKE_EXTRA_NOOPT
#
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE(CheckCCompilerFlag)
INCLUDE(CheckCSourceCompiles)
INCLUDE(CheckFunctionExists)
INCLUDE(CheckLibraryExists)
INCLUDE(CheckIncludeFiles)
INCLUDE(CheckStructHasMember)
INCLUDE(CheckTypeSize)

#########################################
#                                       #
#    Check the capabilities of the      #
#      system we are building for       #
#                                       #
#########################################

# Check the you have GNU Compiler
# -------------------------------
IF(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    SET(HAVE_GNU ON)
ELSE()
    SET(HAVE_GNU OFF)
ENDIF()

# Check the you have Intel Compiler
# -------------------------------
IF(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    SET(HAVE_INTEL ON)
ELSE()
    SET(HAVE_INTEL OFF)
ENDIF()

# Check the you have PGI Compiler
# -------------------------------
IF(CMAKE_Fortran_COMPILER_ID STREQUAL "PGI")
    SET(HAVE_PGI ON)
ELSE()
    SET(HAVE_PGI OFF)
ENDIF()

# Check if library stdc++ is available
# ------------------------------------
CHECK_C_COMPILER_FLAG("-lstdc++" HAVE_STDCPP)
IF(HAVE_STDCPP)
    MESSAGE(STATUS "Looking for libstdc++ - found")
ELSE(HAVE_STDCPP)
    MESSAGE(STATUS "Looking for libstdc++ - not found")
ENDIF(HAVE_STDCPP)

# Intel tricks
# ------------
STRING(REGEX MATCH ".*icc$" _match_icc ${CMAKE_C_COMPILER})
IF (_match_icc)
     MESSAGE(STATUS "Add -nofor_main to the Fortran linker.")
     SET(CMAKE_EXTRA_CFLAGS    "-diag-disable vec")
ENDIF (_match_icc)

STRING(REGEX MATCH ".*ifort$" _match_ifort ${CMAKE_Fortran_COMPILER})
IF (_match_ifort)
     MESSAGE(STATUS "Add -nofor_main to the Fortran linker.")
     SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} -nofor_main")
     SET(CMAKE_EXTRA_FFLAGS    "-diag-disable vec -fltconsistency -fp_port")
     SET(CMAKE_EXTRA_LDFLAGS_F "-nofor_main")
     SET(CMAKE_EXTRA_NOOPT     "-mp")
ENDIF (_match_ifort)

# Free Fortran Compiler tricks
# ----------------------------
STRING(REGEX MATCH ".*ftn$" _match_ftn ${CMAKE_Fortran_COMPILER})
IF (_match_ftn)
    MESSAGE(STATUS "Add -Mnomain to the Fortran linker.")
    SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} -Mnomain -Bstatic")
    SET(CMAKE_EXTRA_LDFLAGS_F "-Mnomain")
ENDIF (_match_ftn)

# IBM tricks
# ----------
STRING(REGEX MATCH ".*xlc$" _match_xlc ${CMAKE_C_COMPILER})
IF (_match_xlc)
     MESSAGE(ERROR "Please use the thread-safe version of the xlc compiler (xlc_r)")
ENDIF (_match_xlc)
STRING(REGEX MATCH ".*xlc_r$" _match_xlc ${CMAKE_C_COMPILER})
IF (_match_xlc)
     IF(BUILD_64bits)
         MESSAGE(STATUS "Add -q64 to the C compiler/linker.")
         SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -q64")
     ENDIF(BUILD_64bits)
     SET(CMAKE_EXTRA_CFLAGS "-qstrict -qthreaded")
ENDIF (_match_xlc)

STRING(REGEX MATCH ".*xlf$" _match_xlf ${CMAKE_Fortran_COMPILER})
IF (_match_xlf)
     MESSAGE(ERROR "Please use the thread-safe version of the xlf compiler (xlf_r)")
ENDIF (_match_xlf)
STRING(REGEX MATCH ".*xlf_r$" _match_xlf ${CMAKE_Fortran_COMPILER})
IF (_match_xlf)
  IF(BUILD_64bits)
     SET(arch_flags "-q64")
  ELSE(BUILD_64bits)
     SET(arch_flags "-q32")
  ENDIF(BUILD_64bits)
  MESSAGE(STATUS "Add ${arch_flags} and -nofor_main to the Fortran linker.")
  SET(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} ${arch_flags} -nofor_main")
  SET(CMAKE_EXTRA_FFLAGS "-qstrict -qthreaded")
ENDIF (_match_xlf)

# check for the CPU we build for
# ------------------------------
MESSAGE(STATUS "Building for target ${CMAKE_SYSTEM_PROCESSOR}")
STRING(REGEX MATCH "(i.86-*)|(athlon-*)|(pentium-*)" _mach_x86 ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_x86)
    MESSAGE(STATUS "Found target for X86")
    SET(ARCH_X86 1)
ENDIF (_mach_x86)

STRING(REGEX MATCH "(x86_64-*)|(X86_64-*)|(AMD64-*)|(amd64-*)" _mach_x86_64 ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_x86_64)
    MESSAGE(STATUS "Found target X86_64")
    SET(ARCH_X86_64 1)
ENDIF (_mach_x86_64)

STRING(REGEX MATCH "(ppc-*)|(powerpc-*)" _mach_ppc ${CMAKE_SYSTEM_PROCESSOR})
IF (_mach_ppc)
    MESSAGE(STATUS "Found target for PPC")
    SET(ARCH_PPC 1)
ENDIF (_mach_ppc)

# Fix the building system for 32 or 64 bits.
# ------------------------------------------
#
# On MAC OS X there is a easy solution, by setting the 
# CMAKE_OSX_ARCHITECTURES to a subset of the following values:
# ppc;ppc64;i386;x86_64.
# On Linux this is a little bit tricky. We have to check that the
# compiler supports the -m32/-m64 flags as well as the linker.
# Once this issue resolved the CMAKE_C_FLAGS and CMAKE_C_LDFLAGS
# have to be updated accordingly.
#
# TODO: Same trick for the Fortran compiler...
#       no idea how to correctly detect if the required/optional
#      libraries are in the correct format.
#
set(SAVE_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
if (BUILD_64bits)
  if( _match_xlc)
    set( ARCH_BUILD "-q64" )
  else (_match_xlc)
    set( ARCH_BUILD "-m64" )
  endif(_match_xlc)
else (BUILD_64bits)
  if( _match_xlc)
    set( ARCH_BUILD "-q32" )
  else (_match_xlc)
    set( ARCH_BUILD "-m32" )
  endif(_match_xlc)
endif (BUILD_64bits)

set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${ARCH_BUILD}")
CHECK_C_COMPILER_FLAG(${ARCH_BUILD} C_M32or64)

if (C_M32or64)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_BUILD}")
  set(CMAKE_C_LDFLAGS "${CMAKE_C_LDFLAGS} ${ARCH_BUILD}")
  set(LOCAL_FORTRAN_LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS} ${ARCH_BUILD}")
else (C_M32or64)
  set(CMAKE_REQUIRED_FLAGS "${SAVE_CMAKE_REQUIRED_FLAGS}")
endif (C_M32or64)
unset( SAVE_CMAKE_REQUIRED_FLAGS )

# Check compiler flags and capabilities
# -------------------------------------
IF( NOT _match_xlc )
  CHECK_C_COMPILER_FLAG( "-std=c99" HAVE_STD_C99)
  IF( HAVE_STD_C99 )
    SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99" )
  ENDIF( HAVE_STD_C99 )
ELSE( NOT _match_xlc )
  CHECK_C_COMPILER_FLAG( "-qlanglvl=extc99" HAVE_STD_C99)
  IF( HAVE_STD_C99 )
    SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qlanglvl=extc99" )
  ENDIF( HAVE_STD_C99 )
ENDIF( NOT _match_xlc )

# Set warnings for debug builds 
# -----------------------------
CHECK_C_COMPILER_FLAG( "-Wall" HAVE_WALL )
IF( HAVE_WALL )
    SET( C_WFLAGS "${C_WFLAGS} -Wall" )
ENDIF( HAVE_WALL )
CHECK_C_COMPILER_FLAG( "-Wextra" HAVE_WEXTRA )
IF( HAVE_WEXTRA )
    SET( C_WFLAGS "${C_WFLAGS} -Wextra" )
ENDIF( HAVE_WEXTRA )

# flags for the overly verbose icc
# --------------------------------
CHECK_C_COMPILER_FLAG( "-wd424" HAVE_WD )
IF( HAVE_WD )
    # 424: checks for duplicate ";"
    # 981: every volatile triggers a "unspecified evaluation order", obnoxious
    #      but might be useful for some debugging sessions. 
    # 1419: warning about extern functions being declared in .c
    #       files
    # 1572: cuda compares floats with 0.0f. 
    SET( C_WFLAGS "${C_WFLAGS} -wd424 -wd981 -wd1419 -wd1572" )
ENDIF( HAVE_WD )
SET( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${C_WFLAGS}" )
SET( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${C_WFLAGS}" )

# Threads
# -------
find_package(Threads)
if(Threads_FOUND)
  set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${CMAKE_THREAD_LIBS_INIT}")
  CHECK_FUNCTION_EXISTS(pthread_create HAVE_PTHREAD)
  if(HAVE_PTHREAD)
    set(EXTRA_LIBS "${EXTRA_LIBS};${CMAKE_THREAD_LIBS_INIT}")
  endif(HAVE_PTHREAD)
endif(Threads_FOUND)

CHECK_FUNCTION_EXISTS(sched_setaffinity HAVE_SCHED_SETAFFINITY)
if( NOT HAVE_SCHED_SETAFFINITY )
  CHECK_LIBRARY_EXISTS(rt sched_setaffinity "" HAVE_SCHED_SETAFFINITY)
endif( NOT HAVE_SCHED_SETAFFINITY )

# Timeval, timespec, realtime clocks, etc
# ---------------------------------------
CHECK_STRUCT_HAS_MEMBER("struct timespec" tv_nsec time.h HAVE_TIMESPEC_TV_NSEC)
if( NOT HAVE_TIMESPEC_TV_NSEC )
  add_definitions(-D_GNU_SOURCE)
  CHECK_STRUCT_HAS_MEMBER("struct timespec" tv_nsec time.h HAVE_TIMESPEC_TV_NSEC)
endif( NOT HAVE_TIMESPEC_TV_NSEC )
CHECK_LIBRARY_EXISTS(rt clock_gettime "" HAVE_CLOCK_GETTIME)
if( HAVE_CLOCK_GETTIME )
  set(EXTRA_LIBS "${EXTRA_LIBS};rt")
endif( HAVE_CLOCK_GETTIME )

# stdlib, stdio, string, getopt, etc
# ----------------------------------
CHECK_INCLUDE_FILES(stdarg.h HAVE_STDARG_H)

# va_copy is special as it is not required to be a function.
IF (HAVE_STDARG_H)
  CHECK_C_SOURCE_COMPILES("
      #include <stdarg.h>
      int main(void) {
      va_list a, b;
      va_copy(a, b);
      return 0;
      }"
      HAVE_VA_COPY
      )
 
  IF (NOT HAVE_VA_COPY)
    CHECK_C_SOURCE_COMPILES("
    #include <stdarg.h>
    int main(void) {
        va_list a, b;
        __va_copy(a, b);
        return 0;
    }"
    HAVE_UNDERSCORE_VA_COPY
    )
  endif (NOT HAVE_VA_COPY)
endif (HAVE_STDARG_H)

CHECK_FUNCTION_EXISTS(asprintf HAVE_ASPRINTF)
CHECK_FUNCTION_EXISTS(vasprintf HAVE_VASPRINTF)
CHECK_INCLUDE_FILES(getopt.h HAVE_GETOPT_H)
CHECK_INCLUDE_FILES(unistd.h HAVE_UNISTD_H)
CHECK_FUNCTION_EXISTS(getopt_long HAVE_GETOPT_LONG)
CHECK_INCLUDE_FILES(errno.h HAVE_ERRNO_H)
CHECK_INCLUDE_FILES(stddef.h HAVE_STDDEF_H)
CHECK_FUNCTION_EXISTS(getrusage HAVE_GETRUSAGE)
CHECK_INCLUDE_FILES(limits.h HAVE_LIMITS_H)
CHECK_INCLUDE_FILES(string.h HAVE_STRING_H)

# Looking for architecture parameters
# -----------------------------------
CHECK_TYPE_SIZE(void* SIZEOF_VOID_PTR)
