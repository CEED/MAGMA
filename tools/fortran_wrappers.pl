#!/usr/bin/perl
#
# Processes the include/magma_z.h header file
# to generate control/magma_zf77.cpp Fortran wrappers.
# TODO: need to also generate control/magma_zfortran.F90 prototypes.
#
# @author Mark Gates

use strict;
use Text::Balanced qw( extract_bracketed );

my( $pre, $return, $func, $funcf, $is_gpu,
	$text, $rest, $wrap, $call,
	$args, @args, $arg, $type, $var, $first_arg, $is_ptr );

my @ignore = qw(
	zlaqps
    zlatrd
    zlatrd2
    zlahr2
    zlahru
    zlabrd
    zlaex\d
    zlahr2_m
    zlahru_m
);
my $ignore = join( "|", @ignore );

# --------------------
# print header
print <<EOT;
#include <stdint.h>  // for uintptr_t

#include "magma.h"

/*
 * typedef comming from fortran.h file provided in CUDADIR/src directory
 * it will probably change with future release of CUDA when they use 64 bit addresses
 */
typedef size_t devptr_t;

#ifdef PGI_FORTRAN
#define DEVPTR(__ptr) ((cuDoubleComplex*)(__ptr))
#else
#define DEVPTR(__ptr) ((cuDoubleComplex*)(uintptr_t)(*(__ptr)))
#endif

#ifndef MAGMA_FORTRAN_NAME
#if defined(ADD_)
#define MAGMA_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname##_
#elif defined(NOCHANGE)
#define MAGMA_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname
#elif defined(UPCASE)
#define MAGMA_FORTRAN_NAME(lcname, UCNAME)  MAGMAF_##UCNAME
#endif
#endif

#ifndef MAGMA_GPU_FORTRAN_NAME
#if defined(ADD_)
#define MAGMA_GPU_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname##_gpu_
#elif defined(NOCHANGE)
#define MAGMA_GPU_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname##_gpu
#elif defined(UPCASE)
#define MAGMA_GPU_FORTRAN_NAME(lcname, UCNAME)  MAGMAF_##UCNAME##_GPU
#endif
#endif
EOT


# --------------------
undef $/;  # slurp whole file
while( <> ) {
	# strip out lines we don't want to copy
	s/#ifndef _MAGMA_Z_H_//;
	s/#define _MAGMA_Z_H_//;
	s/#include "magma_zgehrd_m.h"//;
	s/void zpanel_to_q.*//;
	s/void zq_to_panel.*//;
	s/#endif \/\* _MAGMA_Z_H_ \*\///;
	
	# search for each magma function
	while( m/(.*?)^(magma_int_t|int|void)\s+magma_(\w+?)(_gpu)?\s*(\(.*)/ms ) {
		$pre    = $1;
		$return = $2;
		$func   = $3;
		$is_gpu = $4;
		$text   = $5;
		
		($args, $rest) = extract_bracketed( $text, '()' );
		$args =~ s/\n/ /g;
		$args =~ s/^\( *//;
		$args =~ s/ *\)$//;
		
		$funcf = "MAGMAF_" . uc($func) . uc($is_gpu);
		if ( $is_gpu ) {
			$wrap = sprintf( "#define %s MAGMA_GPU_FORTRAN_NAME( %s, %s )\n",
				${funcf}, $func, uc($func) );
		}
		else {
			$wrap = sprintf( "#define %s MAGMA_FORTRAN_NAME( %s, %s )\n",
				${funcf}, $func, uc($func) );
		}
		
		if ( $func =~ m/^($ignore)$/ or $func =~ m/_mgpu/ ) {
			# ignore functions which the user has no need to call, and
			# multi-GPU functions, since we haven't dealt with passing
			# arrays of pointers in Fortran yet
			$wrap = "";
		}
		elsif ( $func =~ m/get_/ ) {
			# special case for get_nb functions
			# is returning an int safe? otherwise, we could make these take an output argument.
			$wrap .= "magma_int_t ${funcf}( magma_int_t *m )\n{\n    return magma_$func( *m );\n}\n\n";
		}
		else {
			# build up wrapper and the call inside the wrapper, argument by argument
			$wrap .= "void ${funcf}(\n    ";
			$call = "magma_$func$is_gpu(\n        ";
			
			$first_arg = 1;
			@args = split( /, */, $args );
			foreach $arg ( @args ) {
				($type, $var) = $arg =~ m/^((?:const +)?\w+(?: *\*+)?) *(\w+[\[\]0-9]*)$/;
				if ( not $type ) {
					print "\nFAILED: func $func, arg $arg\n";
				}
				$is_ptr = ($type =~ m/\*/);
				if ( $is_ptr ) {
					unless( $first_arg ) {
						$wrap .= ",\n    ";
						$call .= ",\n        ";
					}
					if ( ($is_gpu and $var =~ m/^d\w+/) or $var eq "dT" ) {
						# for _gpu interfaces assume ptrs that start with "d" are device pointers
						# Also CPU interface for geqrf, etc., passes dT as device pointer (weirdly)
						$wrap .= "devptr_t *$var";
						$call .= "DEVPTR($var)";
					}
					else {
						$wrap .= "$type$var";
						$call .= $var;
					}
				}
				else {
					unless( $first_arg ) {
						$wrap .= ", ";
						$call .= ", ";
					}
					# convert scalars to pointers for Fortran interface
					$wrap .= "$type *$var";
					$call .= "*$var";
				}
				$first_arg = 0;
			}
			$wrap .= " )\n{\n    $call );\n}\n\n";
		}
		
		print $pre, $wrap;
		$_ = $rest;
		s/^ *;\n+//;
	}
	print;
}
