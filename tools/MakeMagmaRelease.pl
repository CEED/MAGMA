#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Std;

my $major;
my $minor;
my $micro;

# default options
my $svn      = "https://icl.cs.utk.edu/svn/magma/trunk";
my $user     = "";
my $revision = "";
my $rc       = 0;  # release candidate
my $beta     = 0;

# in alphabetic order
my @files2delete = qw(
    BugsToFix.txt
    Makefile.gen
    Release-ToDo-1.1.txt
    Release-ToDo.txt
    cmake_modules
    contrib
    control/sizeptr
    docs
    include/Makefile
    magmablas/obsolete
    magmablas/zhemm_1gpu.cpp
    magmablas/zhemm_1gpu_old.cpp
    make.inc.ig.pgi
    multi-gpu-dynamic-deprecated
    quark
    
    sparse-iter/blas/zmergeidr.cu
    sparse-iter/blas/zbcsrblockinfo.cu
    sparse-iter/blas/magma_z_mpksetup.cu
    sparse-iter/control/magma_z_mpksetup.cpp
    sparse-iter/python
    sparse-iter/src/zciterref.cpp
    sparse-iter/src/zcpbicgstab.cpp
    sparse-iter/src/zcpgmres.cpp
    sparse-iter/src/zcpir.cpp
    sparse-iter/src/zgmres_pipe.cpp
    sparse-iter/src/zilu.cpp
    sparse-iter/src/zlobpcg.cpp
    sparse-iter/src/zp1gmres.cpp
    sparse-iter/src/zpbicgstab.cpp
    sparse-iter/src/zpcg.cpp
    sparse-iter/src/zpgmres.cpp
    sparse-iter/testing/test_matrices
    sparse-iter/testing/testing_*.cpp
    
    src/obsolete
    testing/*.txt
    testing/fortran2.cpp
    testing/testing_zgetrf_f.f
    testing/testing_zgetrf_gpu_f.cuf
    tools
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
    my $stage = "";

    if ( $rc > 0 ) {
        $numversion .= "-rc$rc";
        $stage = "rc$rc";
    }
    if ( $beta > 0 ) {
        $numversion .= "-beta$beta";
        $stage = "beta$beta";
    }

    my $RELEASE_PATH = $ENV{PWD}."/magma-$numversion";
    if ( -e $RELEASE_PATH ) {
        die( "RELEASE_PATH $RELEASE_PATH already exists.\nPlease delete it or use different version.\n" );
    }

    # Save current directory
    my $dir = `pwd`;
    chomp $dir;

    $cmd = "svn export --force $revision $user $svn $RELEASE_PATH";
    myCmd($cmd);

    chdir $RELEASE_PATH;

    # Change version in magma.h
    myCmd("perl -pi -e 's/VERSION_MAJOR +[0-9]+/VERSION_MAJOR $major/' include/magma_types.h");
    myCmd("perl -pi -e 's/VERSION_MINOR +[0-9]+/VERSION_MINOR $minor/' include/magma_types.h");
    myCmd("perl -pi -e 's/VERSION_MICRO +[0-9]+/VERSION_MICRO $micro/' include/magma_types.h");
    myCmd("perl -pi -e 's/VERSION_STAGE +.+/VERSION_STAGE \"$stage\"/' include/magma_types.h");

    # Change the version and date in comments
    my($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime;
    my @months = (
        'January',   'February', 'March',    'April',
        'May',       'June',     'July',     'August',
        'September', 'October',  'November', 'December',
    );
    $year += 1900;
    my $date = "$months[$mon] $year";
    my $script = "s/MAGMA \\\(version [0-9.]+\\\)/MAGMA (version $numversion)/;";
    $script .= " s/\\\@date.*/\\\@date $date/;";
    myCmd("find . -type f -exec perl -pi -e '$script' {} \\;");
    
    # Change version in pkgconfig
    $script = "s/Version: [0-9.]+/Version: $numversion/;";
    myCmd("perl -pi -e '$script' lib/pkgconfig/magma.pc.in");
    
    # Precision Generation
    print "Generate the different precisions\n";
    myCmd("touch make.inc");
    myCmd("make -j generation");

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
#    my $dir = `pwd`;
#    chomp $dir;
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
    print "MakeRelease.pl [options] Major Minor Micro\n";
    print "   -h            Print this help\n";
    print "   -b beta       Beta version\n";
    print "   -c candidate  Release candidate number\n";
    print "   -r revision   Choose svn revision number\n";
    print "   -s url        Choose svn repository for export (default: $svn)\n";
    print "   -u username   SVN username\n";
}

my %opts;
getopts("hu:b:r:s:c:",\%opts);

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
if ( defined $opts{b} ) {
    $beta = $opts{b};
}
if ( ($#ARGV + 1) != 3 ) {
    Usage();
    exit;
}

$major = $ARGV[0];
$minor = $ARGV[1];
$micro = $ARGV[2];

MakeRelease();
#MakeInstallerRelease();
