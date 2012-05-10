#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Std;

my $major;
my $minor;
my $micro;

# default options
#my $svn     = "https://icl.cs.utk.edu/svn/magma/branches/sc_release";
#my $svninst = "https://icl.cs.utk.edu/svn/plasma/plasma-installer";
my $svn      = "https://icl.cs.utk.edu/svn/magma/trunk";
my $user     = "";
my $revision = "";
my $rc       = 0;  # release candidate

my @files2delete = qw(
    Makefile.gen
    tools
    quark
    docs
    include/Makefile
    make.inc.cumin
    make.inc.disco
    make.inc.ig
    make.inc.ig.pgi
    Release-ToDo.txt
    Release-ToDo-1.1.txt
    BugsToFix.txt
    src/zgeqrf_mc.cpp
    src/zgeqrf-v2.cpp
    src/zgeqrf-v3.cpp
    src/zgetrf_mc.cpp
    src/zpotrf_mc.cpp
    src/zhetrd_mgpu.cpp
    src/zlatrd_mgpu.cpp 
    src/obsolete
    magmablas/obsolete
    testing/fortran2.cpp
    testing/*.txt
    testing/testing_zgetrf_f.f
    testing/testing_zgetrf_gpu_f.cuf
    testing/testing_zgeqrf_mc.cpp
    testing/testing_zgeqrf-v2.cpp
    testing/testing_zpotrf_mc.cpp
    testing/testing_zgetrf_mc.cpp
    testing/testing_zswap.cpp
);
# Using qw() avoids need for "quotes", but comments aren't recognized inside qw()
#src/magma_zf77.cpp
#src/magma_zf77pgi.cpp


sub myCmd
{
    my ($cmd) = @_ ;
    print "---------------------------------------------------------------\n";
    print $cmd."\n";
    print "---------------------------------------------------------------\n";
    my $err = system($cmd);
    if ($err != 0) {
        print "Error during execution of the following command:\n$cmd\n";
        exit;
    }
}

sub MakeRelease
{
    my $numversion = "$major.$minor.$micro";
    my $cmd;

    if ( $rc > 0 ) {
        $numversion .= "-rc$rc";
    }

    my $RELEASE_PATH = $ENV{PWD}."/magma-$numversion";

    # Save current directory
    my $dir = `pwd`;
    chomp $dir;

    $cmd = "svn export --force $revision $user $svn $RELEASE_PATH";
    myCmd($cmd);

    chdir $RELEASE_PATH;

    # Change version in plasma.h
    #myCmd("perl -pi -e 's/PLASMA_VERSION_MAJOR +[0-9]+/PLASMA_VERSION_MAJOR $major/' include/plasma.h");
    #myCmd("perl -pi -e 's/PLASMA_VERSION_MINOR +[0-9]+/PLASMA_VERSION_MINOR $minor/' include/plasma.h");
    #myCmd("perl -pi -e 's/PLASMA_VERSION_MICRO +[0-9]+/PLASMA_VERSION_MICRO $micro/' include/plasma.h");

    # Change the version and date in comments
    # TODO make a generic date tag to search for, instead of November 2011.
    my($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime;
    my @months = (
        'January',   'February', 'March',    'April',
        'May',       'June',     'July',     'August',
        'September', 'October',  'November', 'December',
    );
    $year += 1900;
    my $date = "$months[$mon] $year";
    my $script = "s/MAGMA \\\(version [0-9.]+\\\)/MAGMA (version $numversion)/;";
    $script .= " s/November 2011/$date/;";
    myCmd("find . -type f -exec perl -pi -e '$script' {} \\;");
    
    # Change version in pkgconfig
    $script = "s/Version: [0-9.]+/Version: $numversion/;";
    myCmd("perl -pi -e '$script' lib/pkgconfig/magma.pc");
    
    # Precision Generation
    print "Generate the different precisions\n";
    myCmd("touch make.inc");
    myCmd("make generation");

    # Compile the documentation
    #print "Compile the documentation\n";
    #system("make -C ./docs");
    myCmd("rm -f make.inc");

    # Remove non-required files (e.g., Makefile.gen)
    foreach my $file (@files2delete) {
        myCmd("rm -rf $RELEASE_PATH/$file");
    }

    # Remove the lines relative to include directory in root Makefile
    myCmd("perl -ni -e 'print unless /cd include/' $RELEASE_PATH/Makefile");

    # Remove '.Makefile.gen files'
    myCmd("find $RELEASE_PATH -name .Makefile.gen -exec rm -f {} \\;");

    chdir $dir;

    # Save the InstallationGuide if we want to do a plasma-installer release
    #myCmd("cp $RELEASE_PATH/InstallationGuide README-installer");

    # Create tarball
    print "Create the tarball\n";
    my $DIRNAME  = `dirname $RELEASE_PATH`;
    my $BASENAME = `basename $RELEASE_PATH`;
    chomp $DIRNAME;
    chomp $BASENAME;
    myCmd("(cd $DIRNAME && tar cvzf ${BASENAME}.tar.gz $BASENAME)");
}

#sub MakeInstallerRelease {
#
#    my $numversion = "$major.$minor.$micro";
#    my $cmd;
#
#    $RELEASE_PATH = $ENV{ PWD}."/plasma-installer-$numversion";
#
#    # Sauvegarde du rep courant
#    my $dir = `pwd`; chomp $dir;
#
#    $cmd = "svn export --force $revision $user $svninst $RELEASE_PATH";
#    myCmd($cmd);
#
#    # Save the InstallationGuide if we want to do a plasma-installer release
#    myCmd("cp README-installer $RELEASE_PATH/README");
#
#    #Create tarball
#    print "Create the tarball\n";
#    my $DIRNAME  = `dirname $RELEASE_PATH`;
#    my $BASENAME = `basename $RELEASE_PATH`;
#    chomp $DIRNAME;
#    chomp $BASENAME;
#    myCmd("(cd $DIRNAME && tar cvzf ${BASENAME}.tar.gz $BASENAME)");
#}

sub Usage
{
    print "MakeRelease.pl [-h] [-u username] [-r revision] Major Minor Micro\n";
    print "   -h   Print this help\n";
    print "   -r   Choose svn release number\n";
    print "   -s   Choose magma directory for export\n";
    print "   -u   username\n";
}

my %opts;
getopts("hd:u:r:s:c:",\%opts);

if ( defined $opts{h}  ) {
    Usage();
    exit;
}
if ( defined $opts{u} ) {
    $user = "--username $opts{u}";
}
if ( defined $opts{r} ) {
    $revision = "-r $opts{r}";
}
if ( defined $opts{s} ) {
    $svn = $opts{s};
}
if ( defined $opts{c} ) {
    $rc = $opts{c};
}
if ( ($#ARGV + 1) < 3 ) {
    Usage();
    exit;
}

$major = $ARGV[0];
$minor = $ARGV[1];
$micro = $ARGV[2];

MakeRelease();
#MakeInstallerRelease();
