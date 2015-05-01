#!/usr/bin/env perl -i.bak
#
# Initializes sparse structs (matrices, preconditioners, etc.) when declared,
# so pointers will be implicitly set to null,
# so cleanup code works whether or not the struct has been allocated.
#
# @author Mark Gates

use strict;

undef $/;

sub replace
{
	my( $type, $args, $val ) = @_;
	my @args = split( /,\s*/, $args );
	my @args2 = map { $_ . "=$val" } @args;
	my $args2 = join( ", ", @args2 );
	return "$type $args2;";
}


while( <> ) {
	s/(magma_z_matrix) ([^=(){}]*?);/replace($1, $2, "{Magma_CSR}")/ge;
	s/(magma_z_preconditioner) ([^=(){}]*?);/replace($1, $2, "{Magma_CG}")/ge;
	s/(magma_queue_t|magma_event_t) ([^=(){}]*?);/replace($1, $2, "NULL")/ge;
	s/(float|double|magmaDoubleComplex|magmaFloatComplex) (\*[^=(){}]*?);/replace($1, $2, "NULL")/ge;
	print;
}
