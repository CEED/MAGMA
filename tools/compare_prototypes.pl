#!/usr/bin/env perl
#
# Compares prototypes in headers to the definition in the file.
# This is fairly specific to MAGMA.
# Ignores some common differences, like compressing spaces and removing spaces inside ( ).
#
# Usage:  compare_prototypes.pl  include/*.h  file.cpp  file2.cu  ...
#
# Requires wdiff.pl (word diff) to show colorized diff between prototype and cpp definition.
#
# @author Mark Gates

use strict;

# add this script's directory to path, to find wdiff.pl in same directory.
my($dir, $script) = $0 =~ m|^(.*/)?([^/]+$)|;
print "dir $dir\n";
$ENV{PATH} = "$ENV{PATH}:$dir";

my $verbose = 0;
my %proto = ();

# --------------------
# $text_out = canon( $text_in );
# make a canonical version of text by compressing space, etc.
sub canon
{
	my( $p ) = @_;
	
	# strip out C++ comments
	$p =~ s|//.*||g;
	
	#  compress whitespace
	$p =~ s/\n/ /g;
	$p =~ s/  +/ /g;
	$p =~ s/\( +/(/;
	$p =~ s/ +\)/)/;
	
	# easier to start cleaning up with all lowercase (optional)
	# $p = lc($p);      
	
	return $p;
}


# -------------------- read header files
my @src = ();

undef $/;
for my $file ( @ARGV ) {
	if ( $file =~ m/\.(h|hpp)$/ ) {
		#print "header $file\n";
		open( FILE, $file ) or die( $! );
		while( <FILE> ) {
			while( m/^((?:magma_int_t|void|double)\s+(magma\w+)\(.*?\))/smg ) {
				my $p = canon( $1 );
				my $f = $2;
				if ( defined $proto{$f} and $proto{$f} ne $p ) {
					print "ERROR: duplicate prototype for $2\n";
				}
				$proto{$f} = $p;
			}
		}
		close( FILE );
	}
	else {
		#print "src $file\n";
		push @src, $file;
	}
}

#for my $f ( sort keys %proto ) {
#	printf( "%-16s %s\n", $f, $proto{$f} );
#}



# -------------------- read source (cpp) files
@ARGV = @src;
if ( $#ARGV < 0 ) {
	print STDERR "Note: no source files, reading from stdin.\n";
}

my $no_wdiff = 0;

while( <> ) {
	#print "src $ARGV\n";
	while( m/^(?:extern "C"\s+)?((?:magma_int_t|void|double)\s+(magma\w+)\(.*?\))/smg ) {
		my $d = $1;
		my $f = $2;
		if ( $d =~ m/^     +/m ) {
			printf( "%-30s: indented > 4 spaces: $f\n", $ARGV );
		}
		$d = canon( $d );
		if ( not defined $proto{ $f } ) {
			printf( "%-30s: no prototype found for $f\n", $ARGV );
		}
		elsif ( $d ne $proto{$f} ) {
			printf( "%-30s: prototype doesn't match definition:\n", $ARGV );
			open( T1, ">tmp1" );
			print T1 "prototype: $proto{$f}\n";
			close( T1 );
			
			open( T2, ">tmp2" );
			print T2 "definition: $d\n";
			close( T2 );
			
			if ( $no_wdiff or system( "wdiff.pl -a tmp1 tmp2" ) != 0 ) {
				if ( not $no_wdiff ) {
					warn( "Can't run wdiff.pl: $!; using regular diff.\n" );
					$no_wdiff = 1;
				}
				system( "diff tmp1 tmp2" );
			}
		}
		elsif ( $verbose ) {
			print "$ARGV: good $f\n";
		}
	}
}

unlink( "tmp1" );
unlink( "tmp2" );
