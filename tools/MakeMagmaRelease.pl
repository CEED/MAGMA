#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Std;

my $version;
my $major;
my $minor;
my $micro;

# default options
my $svn      = "https://icl.cs.utk.edu/svn/magma/trunk";
my $user     = "";
my $revision = "";
my $rc       = 0;  # release candidate
my $beta     = 0;

# In alphabetic order
# Using qw() avoids need for "quotes", but comments aren't recognized inside qw()
my @files2delete = qw(
    ReleaseChecklist
    Makefile.gen
    contrib
    control/sizeptr
    
    docs/Doxyfile-fast
    docs/output_err
    
    include/Makefile
    magmablas/obsolete
    make.icl
    make.inc
    make.inc.ig.pgi
    multi-gpu-dynamic-deprecated
    quark
    scripts

    sparse-iter/python
    sparse-iter/testing/test_matrices

    src/obsolete
    testing/*.txt
    testing/fortran2.cpp
    testing/testing_zgetrf_f.f
    testing/testing_zgetrf_gpu_f.cuf
    tools
);


sub myCmd
{
    my ($cmd) = @_;
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
    my $cmd;
    my $stage = "";

    if ( $rc > 0 ) {
        $version .= "-rc$rc";
        $stage = "rc$rc";
    }
    if ( $beta > 0 ) {
        $version .= "-beta$beta";
        $stage = "beta$beta";
    }

    # Require recent doxygen, say >= 1.8.
    # ICL machines have ancient versions of doxygen (1.4 and 1.6);
    # the docs don't work at all.
    my $doxygen = `doxygen --version`;
    chomp $doxygen;
    my($v) = $doxygen =~ m/^(\d+\.\d+)\.\d+$/;
    my $doxygen_require = 1.8;
    if ( $v < $doxygen_require ) {
        print <<EOT;
=====================================================================
WARNING: MAGMA requires doxygen version $doxygen_require; installed version is $doxygen.
The documentation will not be generated correctly.
Look in /mnt/sw/doxygen for a recent release.
Enter 'ignore' to acknowledge, or control-C to go fix the problem.
=====================================================================
EOT
        do {
            $_ = <STDIN>;
        } while( not m/\bignore\b/ );
    }

    # svn export won't remove an existing directory,
    # causing old files to get mixed in with updated ones. Force user to fix it.
    my $RELEASE_PATH = $ENV{PWD}."/magma-$version";
    if ( -e $RELEASE_PATH ) {
        die( "RELEASE_PATH $RELEASE_PATH already exists.\nPlease delete it or use different version.\n" );
    }
    
    # Save current directory
    my $dir = `pwd`;
    chomp $dir;

    if ( not $rc and not $beta ) {
        print <<EOT;
Is this the final tar file for this release (that is, the release has been
thoroughly tested), and you want to update version in magma.h in SVN (yes/no)?
EOT
        $_ = <STDIN>;
        if ( m/\b(y|yes)\b/ ) {
            # If run as ./tools/MakeMagmaRelease.pl, no need to change dir;
            # if run as ./MakeMagmaRelease.pl, cd up to trunk, adjust includes, then cd back.
            if ( $dir =~ m|/tools$| ) {
                chdir "..";
            }
            $cmd = "perl -pi -e 's/VERSION_MAJOR +[0-9]+/VERSION_MAJOR $major/; "
                 .             " s/VERSION_MINOR +[0-9]+/VERSION_MINOR $minor/; "
                 .             " s/VERSION_MICRO +[0-9]+/VERSION_MICRO $micro/;' "
                 .             " include/magma_types.h";
            myCmd($cmd);
            myCmd("perl -pi -e 's/PROJECT_NUMBER +=.*/PROJECT_NUMBER         = $major.$minor.$micro/' docs/Doxyfile");
            myCmd("svn diff include/magma_types.h docs/Doxyfile");
            print "Commit these changes now (yes/no)?\n";
            $_ = <STDIN>;
            if ( m/\b(y|yes)\b/ ) {
                myCmd("svn commit -m 'version $major.$minor.$micro' include/magma_types.h");
            }
            
            print "Tag release in SVN (yes/no)?\n";
            $_ = <STDIN>;
            if ( m/\b(y|yes)\b/ ) {
                myCmd("svn copy -m 'version $major.$minor.$micro tag' ^/trunk ^/tags/release-$major.$minor.$micro");
            }
            
            # allow user to see that it went successfully
            print "Type enter to continue\n";
            $_ = <STDIN>;
            
            chdir $dir;
        }
    }

    # Export current SVN
    myCmd("svn export --force $revision $user $svn $RELEASE_PATH");

    print "cd $RELEASE_PATH\n";
    chdir $RELEASE_PATH;

    # Change version in magma.h (in export, not in the SVN itself)
    # See similar replacement above (minus stage)
    $cmd = "perl -pi -e 's/VERSION_MAJOR +[0-9]+/VERSION_MAJOR $major/; "
         .             " s/VERSION_MINOR +[0-9]+/VERSION_MINOR $minor/; "
         .             " s/VERSION_MICRO +[0-9]+/VERSION_MICRO $micro/; "
         .             " s/VERSION_STAGE +.+/VERSION_STAGE \"$stage\"/;' "
         .             " include/magma_types.h";
    myCmd($cmd);
    myCmd("perl -pi -e 's/PROJECT_NUMBER +=.*/PROJECT_NUMBER         = $major.$minor.$micro/' docs/Doxyfile");

    # Change the version and date in comments
    my($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime;
    my @months = (
        'January',   'February', 'March',    'April',
        'May',       'June',     'July',     'August',
        'September', 'October',  'November', 'December',
    );
    $year += 1900;
    my $date = "$months[$mon] $year";
    my $script = "s/MAGMA \\\(version [0-9.]+\\\)/MAGMA (version $version)/;";
    $script .= " s/\\\@date.*/\\\@date $date/;";
    myCmd("find . -type f -exec perl -pi -e '$script' {} \\;");
    
    # Change version in pkgconfig
    $script = "s/Version: [0-9.]+/Version: $version/;";
    myCmd("perl -pi -e '$script' lib/pkgconfig/magma.pc.in");
    
    # Precision Generation
    print "Generate the different precisions\n";
    myCmd("touch make.inc");
    myCmd("make -j generation");

    # Compile the documentation
    print "Compile the documentation\n";
    myCmd("make -C ./docs");
    myCmd("rm -f make.inc");

    # Remove non-required files (e.g., Makefile.gen)
    myCmd("rm -rf @files2delete");

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
#    my $version = "$major.$minor.$micro";
#    my $cmd;
#
#    $RELEASE_PATH = $ENV{ PWD}."/plasma-installer-$version";
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
    print "MakeRelease.pl [options] major.minor.micro\n";
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
if ( ($#ARGV + 1) != 1 ) {
    Usage();
    exit;
}

$version = shift;
($major, $minor, $micro) = $version =~ m/^(\d+)\.(\d+)\.(\d+)$/;

MakeRelease();
#MakeInstallerRelease();
