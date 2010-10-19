#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Std;

my $major; 
my $minor; 
my $micro; 

#Options par défaut
my $DIRNAME;
my $BASENAME;
my $svn    = "https://icl.cs.utk.edu/svn/plasma/trunk";
my $svninst= "https://icl.cs.utk.edu/svn/plasma/plasma-installer";
my $user   = "";

my @file2delete = (
    "Makefile.gen",
    "control/Makefile.gen",
    "compute/Makefile.gen",
    "core_blas/Makefile.gen",
    "examples/Makefile.gen",
    "testing/Makefile.gen",
    "timing/Makefile.gen",
    "docs/doxygen/Makefile.gen",
    "tools",
    "quark/examples",
    "quark/docs",
    "GenerateZDCS.cmake.optional",
    "CMakeBuildNotes",
    "CMakeLists.txt",
    "control/CMakeLists.txt",
    "include/CMakeLists.txt",
    "timing/CMakeLists.txt",
    "docs/asciidoc/CMakeBuildNotes.txt",
    "core_blas/CMakeLists.txt",
    "compute/CMakeLists.txt",
    "testing/lin/CMakeLists.txt",
    "testing/CMakeLists.txt",
    "examples/CMakeLists.txt",
    "makes/cmake32.bat",
    "makes/cmake64.bat"
    );

my $RELEASE_PATH;
my %opts;
my $NUMREL = "";

sub myCmd {
    my ($cmd) = @_ ;
    my $err = 0;

    $err = system($cmd);
    if ($err != 0) {
    	print "Error during execution of the following command:\n$cmd\n";
    	exit;
    }    
}
    
sub MakeRelease {

    my $numversion = $major.'.'.$minor.'.'.$micro;
    my $cmd;

    $RELEASE_PATH = $ENV{ PWD}."/plasma_".$numversion;

    # Sauvegarde du rep courant
    my $dir = `pwd`; chop $dir;

    $cmd = 'svn export --force '.$NUMREL.' '.$user.' '.$svn.' '.$RELEASE_PATH;
    myCmd($cmd);
    
    chdir $RELEASE_PATH;

    # Change version in plasma.h
    myCmd("sed -i 's/PLASMA_VERSION_MAJOR[ ]*[0-9]/PLASMA_VERSION_MAJOR $major/' include/plasma.h"); 
    myCmd("sed -i 's/PLASMA_VERSION_MINOR[ ]*[0-9]/PLASMA_VERSION_MINOR $minor/' include/plasma.h");
    myCmd("sed -i 's/PLASMA_VERSION_MICRO[ ]*[0-9]/PLASMA_VERSION_MICRO $micro/' include/plasma.h");

    # Change the version in comments 
    myCmd("find -type f -exec sed -i 's/\@version[ ]*[.0-9]*/\@version $numversion/' {} \\;");

    #Precision Generation
    print "Generate the different precision\n"; 
    myCmd("touch make.inc");
    myCmd("make generate");

    #Compile the documentation
    print "Compile the documentation\n"; 
    system("make -C ./docs");
    myCmd("rm -f make.inc");
    
    #Remove non required files (Makefile.gen)
    foreach my $file (@file2delete){
	print "Remove $file\n";
 	myCmd("rm -rf $RELEASE_PATH/$file");
    }
 
    # Remove 'include Makefile.gen from Makefile'
    myCmd("find $RELEASE_PATH -name Makefile -exec sed -i '/Makefile.gen/ d' {} \\;");

    chdir $dir;

    #Create tarball
    print "Create the tarball\n";
    $DIRNAME=`dirname $RELEASE_PATH`;
    $BASENAME=`basename $RELEASE_PATH`;
    chop $DIRNAME;
    chop $BASENAME;
    myCmd("(cd $DIRNAME && tar cvzf ${BASENAME}.tar.gz $BASENAME)");

}

sub MakeInstallerRelease {

    my $numversion = $major.'.'.$minor.'.'.$micro;
    my $cmd;

    $RELEASE_PATH = $ENV{ PWD}."/plasma-installer";

    # Sauvegarde du rep courant
    my $dir = `pwd`; chop $dir;

    $cmd = 'svn export --force '.$NUMREL.' '.$user.' '.$svninst.' '.$RELEASE_PATH;
    myCmd($cmd);
    
    #Create tarball
    print "Create the tarball\n";
    $DIRNAME=`dirname $RELEASE_PATH`;
    $BASENAME=`basename $RELEASE_PATH`;
    chop $DIRNAME;
    chop $BASENAME;
    myCmd("(cd $DIRNAME && tar cvzf ${BASENAME}.tar.gz $BASENAME)");
}

sub Usage {
    
    print "MakeRelease.pl [ -h ][ -d Directory ] [ -u username ] [ -r numrelease ] Major Minor Micro\n";
    print "   -h   Print this help\n";
    print "   -d   Choose directory for release\n";
    print "   -r   Choose svn release number\n";
    print "   -s   Choose plasma directory for export\n";
    print "   -u   username\n";

}

getopts("hd:u:r:s:",\%opts);

if ( defined $opts{h} ){
    Usage();
    exit;
}

if (defined $opts{d}){
    $RELEASE_PATH = $opts{d};
}
if (defined $opts{u}){
    $user = "--username $opts{u}";
}

if (defined $opts{r}){
    $NUMREL = "-r $opts{r}";
}
if (defined $opts{s}){
    $svn = $opts{s};
}

if ( ($#ARGV + 1) < 3 ) {
    Usage();
    exit;
}

$major = $ARGV[0];
$minor = $ARGV[1];
$micro = $ARGV[2];

MakeRelease();
MakeInstallerRelease();
