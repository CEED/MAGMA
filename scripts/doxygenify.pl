#!/usr/bin/env perl
#
# Script to convert LAPACK formatted documentation to doxygen formatted documentation.
# As the documentation format varies somewhat, you should verify all the output
# before committing it.
#
# Usage: doxygenify.pl files
#
# @author Mark Gates

use strict;

my $g_match;


#############################################
# Groups map routine names (minus precision and some suffixes)
# to doxygen modules, defined in docs/doxygen-modules.h
my $group;
my %groups = (
	'symv'                 => 'blas2',  'hemv'       => 'blas2',
	'gemv'                 => 'blas2',
	'trsv'                 => 'blas2',
	'trmv'                 => 'blas2',
	
	'trsm'                 => 'blas3',
	'gemm'                 => 'blas3',
	'symm'                 => 'blas3',  'hemm'       => 'blas3',
	'syrk'                 => 'blas3',  'herk'       => 'blas3',
	'syr2k'                => 'blas3',  'her2k'      => 'blas3',
	
	'laln2'                => 'aux0',
	
	'swap'                 => 'aux1',
	'larfg'                => 'aux1',
	'larfgx'               => 'aux1',
	'znrm2'                => 'aux1',
	'axpycp'               => 'aux1',
	
	'auxiliary'            => 'aux2',  # zlaset, etc.
	'transpose'            => 'aux2',
	'transpose_inplace'    => 'aux2',
	'lacpy'                => 'aux2',
	'lascl'                => 'aux2',
	'laset'                => 'aux2',
	'laswp'                => 'aux2',
	'lanhe'                => 'aux2',
	'lange'                => 'aux2',
	'lansy'                => 'aux2',
	'laset'                => 'aux2',
	'geadd'                => 'aux2',
	'symmetrize'           => 'aux2',
	'symmetrize_tiles'     => 'aux2',
	'swapblk'              => 'aux2',
	'swapdblk'             => 'aux2',
	'lat2c'                => 'aux2',
	'lag2z'                => 'aux2',
	'lag2c'                => 'aux2',
	'larf'                 => 'aux2',
	'larfx'                => 'aux2',
	
	'setmatrix'            => 'communication',
	'getmatrix'            => 'communication',
	'setmatrix_transpose'  => 'communication',
	'getmatrix_transpose'  => 'communication',
	'bcyclic'              => 'communication',
	
	'larfb'                => 'aux3',
	'larfbx'               => 'aux3',
	
	# --------------------
	'gesv'     => 'gesv_driver',
	
	'getrf'    => 'gesv_comp',
	'getrf2'   => 'gesv_comp',
	'getri'    => 'gesv_comp',
	'getrs'    => 'gesv_comp',
	
	'getf2'    => 'gesv_aux',
	'trtri'    => 'gesv_aux',
	
	'gessm'    => 'gesv_tile',
	'ssssm'    => 'gesv_tile',
	'tstrf'    => 'gesv_tile',
	
	# -----
	'posv'     => 'posv_driver',
	
	'potrf'    => 'posv_comp',
	'potrf2'   => 'posv_comp',
	'potrf3'   => 'posv_comp',
	'potri'    => 'posv_comp',
	'potrs'    => 'posv_comp',
	
	'potf2'    => 'posv_aux',
	'lauum'    => 'posv_aux',
	
	# -----
	'sysv'     => 'sysv_driver',
	
	'sytrf'    => 'sysv_comp',
	'sytri'    => 'sysv_comp',
	
	# -----
	'gels'     => 'gels_driver',
	'gels3'    => 'gels_driver',
	'gelss'    => 'gels_driver',
	
	'geqrsv'   => 'gels_comp',
	'geqrs'    => 'gels_comp',
	'geqrs3'   => 'gels_comp',
	
	# --------------------
	'geqrf'    => 'geqrf_comp',
	'geqr2x'   => 'geqrf_comp',
	'geqrf2'   => 'geqrf_comp',
	'geqrf3'   => 'geqrf_comp',
	'geqrf4'   => 'geqrf_comp',
	'unmqr'    => 'geqrf_comp',
	'unmqr2'   => 'geqrf_comp',
	'ungqr'    => 'geqrf_comp',
	'ungqr2'   => 'geqrf_comp',
	'gegqr'    => 'geqrf_comp',  # alternate
	
	'geqr2'    => 'geqrf_aux',
	
	'tsqrt'    => 'geqrf_tile',
	
	'geqp3'    => 'geqp3_comp',
	'geqp32'   => 'geqp3_comp',
	'laqps'    => 'geqp3_aux',
	'laqps2'   => 'geqp3_aux',
	'laqps3'   => 'geqp3_aux',
	
	# -----
	'gerqf'    => 'gerqf_comp',
	'unmrq'    => 'gerqf_comp',
	'ungrq'    => 'gerqf_comp',
	
	# -----
	'geqlf'    => 'geqlf_comp',
	'unmql'    => 'geqlf_comp',
	'unmql2'   => 'geqlf_comp',
	'ungql'    => 'geqlf_comp',
	
	# -----
	'gelqf'    => 'gelqf_comp',
	'unmlq'    => 'gelqf_comp',
	'unglq'    => 'gelqf_comp',
	
	# --------------------
	'geev'     => 'geev_driver',
	
	'gehrd'    => 'geev_comp',
	'gehrd2'   => 'geev_comp',
	'unmhr'    => 'geev_comp',
	'hseqr'    => 'geev_comp',
	'trevc'    => 'geev_comp',
	'trevc3'   => 'geev_comp',
	'unmhr'    => 'geev_comp',
	'unghr'    => 'geev_comp',
	
	'lahru'    => 'geev_aux',
	'lahr2'    => 'geev_aux',
	'laqtrsd'  => 'geev_aux',
	'latrsd'   => 'geev_aux',
	
	# --------------------
	'syev'     => 'syev_driver',    'heev'     => 'syev_driver',
	'syevx'    => 'syev_driver',    'heevx'    => 'syev_driver',
	
	'sygv'     => 'sygv_driver',    'hegv'     => 'sygv_driver',
	'sygvx'    => 'sygv_driver',    'hegvx'    => 'sygv_driver',
	
	'sytrd'    => 'syev_comp',      'hetrd'    => 'syev_comp',
	'sytrd2',  => 'syev_comp',      'hetrd2'   => 'syev_comp',
	'stedx'    => 'syev_comp',
	'unmtr'    => 'syev_comp',
	'ungtr'    => 'syev_comp',
	
	'latrd'    => 'syev_aux',
	'latrd2'   => 'syev_aux',
	
	# ----- D&C
	'syevd'    => 'syev_driver',    'heevd'    => 'syev_driver',
	'syevdx'   => 'syev_driver',    'heevdx'   => 'syev_driver',
	
	'sygvd'    => 'sygv_driver',    'hegvd'    => 'sygv_driver',
	'sygvdx'   => 'sygv_driver',    'hegvdx'   => 'sygv_driver',
	
	'hegst'    => 'syev_comp',
	
	'laex0'    => 'syev_aux',
	'laex1'    => 'syev_aux',
	'laex3'    => 'syev_aux',
	
	# ----- RRR
	
	'heevr'    => 'syev_driver',
	'hegvr'    => 'sygv_driver',
	
	# ----- 2-stage
	'syevdx_2stage' => 'syev_driver',    'heevdx_2stage' => 'syev_driver',
	
	'sygvdx_2stage' => 'sygv_driver',    'hegvdx_2stage' => 'sygv_driver',
	
	'ungqr_2stage' => 'syev_2stage',
	'unmqr_2stage' => 'syev_2stage',
	'hetrd_hb2st'  => 'syev_2stage',
	'hetrd_he2hb'  => 'syev_2stage',
	'move_eig'     => 'syev_2stage',
	'bulge_applyQ' => 'syev_2stage',
	'bulge_back'   => 'syev_2stage',
	'bulge_aux'    => 'syev_2stage',
	'bulge_kernel' => 'syev_2stage',
	
	# --------------------
	'gesvd'    => 'gesvd_driver',
	'gesdd'    => 'gesvd_driver',
	
	'gebrd'    => 'gesvd_comp',
	'unmbr'    => 'gesvd_comp',
	'ungbr'    => 'gesvd_comp',
	
	'labrd'    => 'gesvd_aux',
);

#print "groups", join( " ", keys( %groups )), "\n";

# variables to make lowercase
# this excludes matrices A, B, C, etc.
my $vars = "D|E|M|N|K|NRHS|IPIV|LDA|LDB|LDC|LDZ|LIWORK|LRWORK|INFO|ABSTOL|ALPHA|BETA|COMPT|CUTPNT|DIAG|DIRECT|"
         . "DWORK|HWORK|IB|IFAIL|IHI|IL|IU|ILO|INDXQ|ISUPPZ|ITER|ITYPE|IWORK|"
         . "JOBU|JOBVL|JOBVR|JOBVT|JOBZ|JPVT|KB|LD[A-Z]+\d?|NRGPU|OFFSET|"
         . "RANGE|RHO|RWORK|SIDE|STOREV|TAU|TAUP|TAUQ|UPLO|VBLKSIZ|WORK|TRANS|TRANSA|TRANSB|LWORK";


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
		#     variable  (input) integer
		# and change into:
		#     @param[in]
		#     variable    integer
		s/^( +)(\w+ +)\((?:device )?input\) +(\w+.*)/$1\@param[in]\n$1$2$3/mg;
		s/^( +)(\w+ +)\((?:device )?input or input.output\) +(\w+.*)/$1\@param[in,out]\n$1$2$3/mg;
		s/^( +)(\w+ +)\((?:device )?output\) +(\w+.*)/$1\@param[out]\n$1$2$3/mg;
		s/^( +)(\w+ +)\((?:device )?input.output\) +(\w+.*)/$1\@param[in,out]\n$1$2$3/mg;
		s/^( +)(\w+ +)\((?:device )?workspace\) +(\w+.*)/$1\@param\n$1$2(workspace) $3/mg;
		s/^( +)(\w+ +)\((?:device )?workspace.output\) +(\w+.*)/$1\@param[out]\n$1$2(workspace) $3/mg;
		s/^( +)(\w+ +)\((?:device )?input.workspace\) +(\w+.*)/$1\@param[in]\n$1$2(workspace) $3/mg;
		s/(param\[[^]]+\]\s+)($vars)\b/$1.lc($2)/eg;    # lowercase some names
		s/(param\s+)($vars)\b/$1.lc($2)/eg;             # lowercase some names (no [in] etc.)
		
		# make option's values into bullet lists
		while( s/((?:diag|direct|job\w*|range|side|storev|trans\w*|uplo) +CHARACTER[^@]*\n +) (     [=<>]+ ['0-9])/$1-$2/s ) {}
		
		# make info values into bullet lists
		while( s/(info +INTEGER[^@]*\n +) (     [=<>]+ \d)/$1-$2/s ) {}
		
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
	
	$f =~ s/_gemm\d?//;
	$f =~ s/_right//;
	$f =~ s/_nopiv//;
	$f =~ s/_incpiv//;
	#$f =~ s/_2stage//;
	#$f =~ s/_he2hb//;  $f =~ s/_hb2st//;
	#$f =~ s/_sy2sb//;  $f =~ s/_sb2st//;
	
	$f =~ s/_a_0//;
	$f =~ s/_ab_0//;
	$f =~ s/_defs//;
	$f =~ s/_tesla//;
	$f =~ s/_fermi//;
	$f =~ s/_[NT]_[NT]//;
	$f =~ s/_MLU//;  # whatever that is
	$f =~ s/_old//;
	$f =~ s/_work//;
	$f =~ s/_kernels//;
	$f =~ s/_reduce//;
	$f =~ s/_special//;
	$f =~ s/_spec//;
	$f =~ s/_stencil//;
	$f =~ s/_batched//;
	$f =~ s/_batch//;
	$f =~ s/[_-]v\d$//;
	$f =~ s/_ooc$//;
	$f =~ s/_m?gpu//;
	$f =~ s/_1gpu//;
	$f =~ s/_m$//;
	$f =~ s/_mt$//;
	$f =~ s/[_-]v\d$//;
	$f =~ s/hemv_32/hemv/;
	$f =~ s/_kernel//;
	my $p = '';
	if ( $f =~ s/^(z)c//     ) { $p = $1; }
	if ( $f =~ s/^([sdcz])// ) { $p = $1; }
	
	# find doxygen group for base routine name
	$group = $groups{$f} or warn( "No group for '$f'\n" );
	$group = "magma_$p$group";
	$group =~ s/magma_([zc])sy/magma_$1he/;
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
