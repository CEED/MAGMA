#!/usr/bin/env perl
#
# Script to convert MAGMA-SPARSE documentation to doxygen formatted documentation.
# As the documentation format varies somewhat, you should verify all the output
# before committing it.
#
# Usage: doxygenify_sparse.pl files
#
# @author Mark Gates

use strict;

my $g_match;


#############################################
# Groups map routine names (minus precision and some suffixes)
# to doxygen modules, defined in docs/doxygen-modules.h
my $group;
my %groups = (
	'gmres'    => 'gesv',
	'bicgstab' => 'gesv',
	               
	'cg'       => 'posv',
	               
	'lobpcg'   => 'syev',
	'lobpcg'   => 'syev',
);

#print "groups", join( " ", keys( %groups )), "\n";

# variables to make lowercase
# this excludes matrices A, B, C, etc.
# my $vars = "D|E|M|N|K|NRHS|IPIV|LDA|LDB|LDC|LDZ|LIWORK|LRWORK|INFO|ABSTOL|ALPHA|BETA|COMPT|CUTPNT|DIAG|DIRECT|"
#          . "DWORK|HWORK|IB|IFAIL|IHI|IL|IU|ILO|INDXQ|ISUPPZ|ITER|ITYPE|IWORK|"
#          . "JOBU|JOBVL|JOBVR|JOBVT|JOBZ|JPVT|KB|LD[A-Z]+\d?|NRGPU|OFFSET|"
#          . "RANGE|RHO|RWORK|SIDE|STOREV|TAU|TAUP|TAUQ|UPLO|VBLKSIZ|WORK|TRANS|TRANSA|TRANSB|LWORK";


#############################################
# input:   comment block, in LAPACK format.
# returns: documentation in doxygen format.
# Comments that don't look like LAPACK format documentation are ignored,
# specifically, it must have "Purpose" or "Arguments".
sub docs
{
	my($_) = @_;
	if ( m/(Purpose|Arguments)/ ) {
		$g_match += 1;
		#print "docs\n";
		#print;
		
		# look for things like:
		#  type *variable  description
		s/^( +)(int|magma_\w+_t|magma_[scdz]_\w+|magmaDoubleComplex|magmaFloatComplex) +(\**) *(\w+)( +)(.*)/sprintf("    \@param\n    %-12s$2$3\n                $6\n",$4)/mge;
		
		
		# look for things like:
		#     variable  (input) integer
		# and change into:
		#     @param[in]
		#     variable    integer
		#s/^( +)(\w+ +)\((?:device )?input\) +(\w+.*)/$1\@param[in]\n$1$2$3/mg;
		#s/^( +)(\w+ +)\((?:device )?input or input.output\) +(\w+.*)/$1\@param[in,out]\n$1$2$3/mg;
		#s/^( +)(\w+ +)\((?:device )?output\) +(\w+.*)/$1\@param[out]\n$1$2$3/mg;
		#s/^( +)(\w+ +)\((?:device )?input.output\) +(\w+.*)/$1\@param[in,out]\n$1$2$3/mg;
		#s/^( +)(\w+ +)\((?:device )?workspace\) +(\w+.*)/$1\@param\n$1$2(workspace) $3/mg;
		#s/^( +)(\w+ +)\((?:device )?workspace.output\) +(\w+.*)/$1\@param[out]\n$1$2(workspace) $3/mg;
		#s/^( +)(\w+ +)\((?:device )?input.workspace\) +(\w+.*)/$1\@param[in]\n$1$2(workspace) $3/mg;
		#s/(param\[[^]]+\]\s+)($vars)\b/$1.lc($2)/eg;    # lowercase some names
		#s/(param\s+)($vars)\b/$1.lc($2)/eg;             # lowercase some names (no [in] etc.)
		
		# make option's values into bullet lists
		#while( s/((?:diag|direct|job\w*|range|side|storev|trans\w*|uplo) +CHARACTER[^@]*\n +) (     [=<>]+ ['0-9])/$1-$2/s ) {}
		
		# make info values into bullet lists
		#while( s/(info +INTEGER[^@]*\n +) (     [=<>]+ \d)/$1-$2/s ) {}
		
		#s/INTEGER/integer/g;
		#s/CHARACTER\*1/enum/g;
		#s/COMPLEX_16/double-complex/g;
		
		# fix /* to /**
		# fix ===== */ to ******/
		# remove extraneous MAGMA copyright (it should be in comment at top of file)
		s|/\*[ \n]|/**|;
		s|\n+( +)(=+)\s*\*/|\n\n$1\@ingroup $group\n$1********************************************************************/|m;
		s| *-- MAGMA .*\n( +Univ.*\n)* +\@date.*\n||;
		s|^( +)(===+) *$|$1 . ('-' x length($2))|meg;
		s|\.\.$|.|mg;
		
		# insert \n for paragraph breaks
		while( s|(\@param[^#]*\n) *\n( +\w[^#]*\@param)|$1    \\n\n$2| ) {}
	}
	return $_;
}


#############################################
# parse each file
for my $arg ( @ARGV ) {
	# strip off various prefix and suffix from filename to get base routine name
	my $f = $arg;
	$f =~ s/lapack_//;
	$f =~ s/magma_//;
	$f =~ s/\.(f|h|cu|cpp)//;
	
	$f =~ s/\d*$//;
	$f =~ s/_old//;
	$f =~ s/_merge//;
	$f =~ s/^(z|zc)p//;  # preconditioned?
	
	# $f =~ s/_gemm\d?//;
	# $f =~ s/_right//;
	# $f =~ s/_nopiv//;
	# $f =~ s/_incpiv//;
	# #$f =~ s/_2stage//;
	# #$f =~ s/_he2hb//;  $f =~ s/_hb2st//;
	# #$f =~ s/_sy2sb//;  $f =~ s/_sb2st//;
	# 
	# $f =~ s/_a_0//;
	# $f =~ s/_ab_0//;
	# $f =~ s/_defs//;
	# $f =~ s/_tesla//;
	# $f =~ s/_fermi//;
	# $f =~ s/_[NT]_[NT]//;
	# $f =~ s/_MLU//;  # whatever that is
	# $f =~ s/_old//;
	# $f =~ s/_work//;
	# $f =~ s/_kernels//;
	# $f =~ s/_reduce//;
	# $f =~ s/_special//;
	# $f =~ s/_spec//;
	# $f =~ s/_stencil//;
	# $f =~ s/_batched//;
	# $f =~ s/_batch//;
	# $f =~ s/[_-]v\d$//;
	# $f =~ s/_ooc$//;
	# $f =~ s/_m?gpu//;
	# $f =~ s/_1gpu//;
	# $f =~ s/_m$//;
	# $f =~ s/_mt$//;
	# $f =~ s/[_-]v\d$//;
	# $f =~ s/hemv_32/hemv/;
	# $f =~ s/_kernel//;
	my $p = '';
	if ( $f =~ s/^(z)c//     ) { $p = $1; }
	if ( $f =~ s/^([sdcz])// ) { $p = $1; }
	
	# find doxygen group for base routine name
	$group = $groups{$f} or warn( "No group for '$f'\n" );
	$group = "magmasparse_$p$group";
	$group =~ s/magmasparse_([zc])sy/magmasparse_$1he/;
	printf( "%-30s ==> %-30s\n", $arg, $group );
	
	# read in entire file.
	# process each /* ... */ comment block using docs().
	# output code and updated comments.
	#print "file $arg\n";
	open( IN, $arg );
	rename( $arg, "$arg.bak" );
	open( OUT, ">$arg" );
	$g_match = 0;
	undef $/;
	while( <IN> ) {
		while( m|\G(.*?)(/\*.*?\*/)|gcs ) {
			#print     "<<< one >>>\n", $1, "\n\n<<< TWO >>>\n", $2, "\n";
			#print OUT "<<< one >>>\n", $1, "\n\n<<< TWO >>>\n", $2, "\n";
			print OUT $1, docs($2);
		}
		m|\G(.*)|s;
		#print     "<<< three >>>\n", $1, "\n";
		#print OUT "<<< three >>>\n", $1, "\n";
		print OUT $1;
	}
	if ( not $g_match ) {
		print "  --  no documentation found\n";
	}
	else {
		print "\n";
	}
	close( IN );
	close( OUT );
}
