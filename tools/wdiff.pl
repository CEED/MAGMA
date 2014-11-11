#!/usr/bin/perl
#
# Usage: wdiff.pl [-a|-h|-t] old_file new_file
#   -a  ANSI colors on terminal
#   -h  HTML output
#   -t  TEX  output (see also latexdiff command)
#
# Prints a word diff between two files, that is, ignoring changes in newlines
# and whitespace, just highlight words that changed.
#
# @author Mark Gates

use strict;
use File::Temp qw/tempfile/;
use Getopt::Std;

my %opts = ();
getopts( 'aht', \%opts );
my $ansi = $opts{'a'};
my $html = $opts{'h'};
my $tex  = $opts{'t'};

my( $oldfile, $newfile ) = @ARGV;

# escape characters for ANSI colors
# see http://en.wikipedia.org/wiki/ANSI_escape_code
my $esc     = chr(0x1B) . "[";
my $red     = '31m';
my $green   = '32m';
my $yellow  = '33m';
my $blue    = '34m';
my $magenta = '35m';
my $cyan    = '36m';
my $white   = '37m';
my $black   = '0m';

# -----
# define text to print before & after changes
my $before_old = '[-';
my $after_old  = '-]';
my $before_new = '[+';
my $after_new  = '+]';

if ( $ansi ) {
	$before_old = "$esc$red\[";
	$after_old  = "\]$esc$black";
	$before_new = "$esc$green\[";
	$after_new  = "\]$esc$black";
}
elsif ( $html ) {
	$before_old = '<span class="old">';
	$after_old  = '</span>';
	$before_new = '<span class="new">';
	$after_new  = '</span>';
}
elsif ( $tex ) {
	$before_old = '\old{';
	$after_old  = '}';
	$before_new = '\new{';
	$after_new  = '}';
}

undef $/;

# -----
open( OLD, $oldfile ) or die( "Can't open file '$oldfile': $!\n" );
my $old = <OLD>;
close( OLD );

# todo make punctuation seperate words
my $oldtmp = $old;
$oldtmp =~ s/\s+/\n/g;
my($OLDTMP, $oldtmpfile) = tempfile( '/tmp/wdiff-old-XXXXXX' );
print $OLDTMP $oldtmp;
close( $OLDTMP );

my @old = ();
while( $old =~ s/^(\S*\s+)// ) {
	push @old, $1;
}


# -----
open( NEW, $newfile ) or die( "Can't open file '$newfile': $!\n" );
my $new = <NEW>;
close( NEW );

my $newtmp = $new;
$newtmp =~ s/\s+/\n/g;
my($NEWTMP, $newtmpfile) = tempfile( '/tmp/wdiff-new-XXXXXX' );
print $NEWTMP $newtmp;
close( $NEWTMP );

my @new = ();
while( $new =~ s/^(\S*\s+)// ) {
	push @new, $1;
}


# -----
my $diff = `diff $oldtmpfile $newtmpfile`;
my @diff = split( "\n", $diff );

if ( $html ) {
print <<EOT;
<html>
<head>
  <style type="text/css">
    .old { color: #999999; text-decoration: line-through; }
    .new { color: #339933; }
  </style>
</head>
<body>
<pre>
EOT
}

# read diff output, using line numbers $old1, $old2, $new1, $new2
# to index into @old and @new arrays.
my $newindex = 1;
foreach my $line ( @diff ) {
	if ( my($old1, $junk1, $old2, $type, $new1, $junk2, $new2) =
	     $line =~ m/(\d+)(,(\d+))?([acd])(\d+)(,(\d+))?/ ) {
		#print STDERR "$line newindex $newindex\n";
		if ( not $old2 ) { $old2 = $old1; }
		if ( not $new2 ) { $new2 = $new1; }
		#print "\ndiff $old1,$old2  $new1,$new2 ($newindex) type $type\n";
		#printf STDERR "     \@new[ %3d .. %3d ]\n",   $newindex-1, $new1-2;
		#printf STDERR "     \$new[ %3d        ]\n",   $new1-1              if ( $type eq 'd' );
		#printf STDERR "   [ \@old[ %3d .. %3d ] ]\n", $old1-1,     $old2-1 if ( $type ne 'a' );
		#printf STDERR "   { \@new[ %3d .. %3d ] }\n", $new1-1,     $new2-1 if ( $type ne 'd' );
		
		print     join( '', @new[ $newindex-1 .. $new1-2 ] );
		print               $new[ $new1-1 ]                          if ( $type eq 'd' );
		# exclude space before & after text
		my $old = join( '', @old[ $old1-1     .. $old2-1 ] );
		my $new = join( '', @new[ $new1-1     .. $new2-1 ] );
		my( $old_mid, $old_post ) = $old =~ m/^(.*?)(\s*)$/;
		my( $new_mid, $new_post ) = $new =~ m/^(.*?)(\s*)$/;
		if ( $old_post eq $new_post ) { $old_post = ''; }
		print $before_old, $old_mid, $after_old, $old_post if ( $type ne 'a' );
		print $before_new, $new_mid, $after_new, $new_post if ( $type ne 'd' );
		$newindex = $new2+1;
	}
}

#print "final ($newindex)\n";
#printf STDERR "     \@new[ %3d .. %3d ]\n",   $newindex-1, $#new;
print join( '', @new[ $newindex-1 .. $#new ] );

if ( $html ) {
print <<EOT
</pre>
</body>
EOT
}
