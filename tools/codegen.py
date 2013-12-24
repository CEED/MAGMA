#!/usr/bin/env python
"""@package Tools

This python script is responsible for precision generation replacements
as well as replacements of any kind in other files.  Different types of
replacements can be defined such that no two sets can conflict.  Multiple
types of replacements can, however, be specified for the same file.

@author Wesley Alvaro
@author Mark Gates
"""
__version__=2013.1224

import sys
import os
from os import path
from optparse import OptionParser, OptionGroup

from Conversion import EXTS, Conversion, check_gen, visible, valid_extension

def main():
    """Create option parser, set static variables of the converter and manage printing options/order."""
    global EXTS
    
    # Files found to be workable.
    work = []
    
    # Create the options parser for detecting options on the command line.    
    parser = OptionParser(usage="Usage: %prog [options]", version='%prog '+str(__version__))
    group = OptionGroup(parser, "Printing Options", "These options control the printing output.")
    group.add_option("-i", "--in-files",  help='Print input  filenames.',              action='store_true', dest='in_print',  default=False)
    group.add_option("-o", "--out-files", help='Print output filenames.',              action='store_true', dest='out_print', default=False)
    group.add_option("-m", "--make",      help='Print Makefile, i.e., .Makefile.gen',  action='store_true', dest='make',      default=False)
    group.add_option("-d", "--debug",     help='Print debugging messages.',            action='store_true', dest='debug',     default=False)
    parser.add_option_group(group)
    group = OptionGroup(parser, "Operating Mode Options", "These options alter the way the program operates on the input/output files.")
    group.add_option("-c", "--clean",     help='Remove files that are the product of generation.',     action='store_true', dest='out_clean', default=False)
    group.add_option("-T", "--test",      help="Don't actually do any work.",                          action='store_true', dest='test',      default=False)
    parser.add_option_group(group)
    group = OptionGroup(parser, "Settings", "These options specify how work should be done.")
    group.add_option("-P", "--prefix",     help='Output directory if different from input directory.', action='store', dest='prefix',     default=None)
    group.add_option("-f", "--file",       help='Files on which to operate [deprecated].',             action='store', dest='files',      type='string', default="")
    group.add_option("-p", "--precisions", help='Precisions to generate.',                             action='store', dest='precisions', type='string', default="")
    group.add_option("-e", "--extensions", help='File extensions on which to operate when recursing.', action='store', dest='exts',       type='string', default="")
    group.add_option("-R", "--recursive",  help='Recursively walk directories; convert files matching extensions.', action='store', dest='recursive',  default=False)
    parser.add_option_group(group)
    
    (options, args) = parser.parse_args()
    
    # deal with deprecated --files option
    if options.files:
        print >>sys.stderr, "Using deprecated codegen.py --file option; adding to args"
        args += options.files.split()
    
    # If file extensions are specified, override defaults.
    if options.exts:
        EXTS = options.exts.split()
    
    # Fill the 'work' array with files found to be operable.
    if ( options.recursive ):
        # recurse on current directory if none given
        if not args:
            args = ['.']
        for arg in args:
            for root, dirs, files in os.walk( arg ):
                dirs[:] = filter( visible, dirs )  # use in-place assignment to affect os.walk
                files   = filter( visible, files )
                files   = filter( valid_extension, files )
                for filename in files:
                    check_gen( path.join(root, filename), work )
        # end
    else:
        for arg in args:
            check_gen( arg, work )
    
    # Set static options for conversion.
    Conversion.debug  = options.debug
    Conversion.make   = options.make
    Conversion.prefix = options.prefix
    Conversion.required_precisions = options.precisions.split()
    if options.out_print or options.out_clean or options.in_print or options.make or options.test:
        Conversion.test = True
    
    if options.make:
        # print header for .Makefile.gen
        print '## Automatically generated Makefile'
        print 'PYTHON ?= python\n'
    
    c = Conversion(); # This initializes the variable for static member access.
    
    for w in work:
        # For each valid conversion file found.
        try:
            # Try creating and executing a converter.
            c = Conversion(w[0], w[1], w[2])
            c.run()
        except Exception, e:
            print >> sys.stderr, str(e)
            continue
    
    if options.make:
        # print footer for .Makefile.gen
        print
        print 'gen = ' + ' '.join(c.files_out) + '\n'
        print 'cleangen:'
        print '\trm -f $(gen)\n'
        print 'generate: $(gen)\n'
        print '.PHONY: cleangen generate'
    
    if options.in_print:
        # print input files
        print ' '.join(c.files_in)
    
    if options.out_print:
        # print output files
        print ' '.join(c.files_out)
    
    if options.out_clean:
        # clean generated files
        for filename in c.files_out:
            if path.exists(filename):
                os.remove(filename)
# end main

if __name__ == "__main__":
    main()
