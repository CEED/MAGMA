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
import re
import shutil
from os import path
from datetime import datetime
import traceback

# --------------------
from magmasubs import subs

# Fill in subs_search with same structure as subs, but containing None values.
# Later in substitute(), we'll cache compiled regexps in subs_search.
# We could pre-compile them here, but that would compile many unneeded ones.
#
# Fill in subs_replace with pre-processed version of subs, removing regexp escapes.
try:
    subs_search  = {}
    subs_replace = {}
    for key in subs.keys():
        nrow = len( subs[key]    )
        ncol = len( subs[key][0] )
        subs_search [key] = map( lambda x: [None]*ncol, xrange(nrow) )
        subs_replace[key] = map( lambda x: [None]*ncol, xrange(nrow) )
        for (i, row) in enumerate( subs[key] ):
            for (j, sub) in enumerate( row ):
                try:
                    sub = sub.replace( r'\b', ''  )
                    sub = sub.replace( r'\*', '*' )
                    sub = sub.replace( r'\(', '(' )
                    sub = sub.replace( r'\)', ')' )
                    sub = sub.replace( r'\.', '.' )
                except:
                    pass
                subs_replace[key][i][j] = sub
            # end
        # end
    # end
except Exception, e:
    print >> sys.stderr, 'error in subs:', e
    traceback.print_exc()


# --------------------
# Keyword used to signal replacement actions on a file
KEYWORD = '@precisions'

# Replace the above keyword with this one post-replacements
DONE_KEYWORD = '@generated'

# Regular expression for the replacement formatting
# re.M means ^ and $ match line breaks anywhere in the string (which is the whole file)
REGEX = re.compile( '^.*'+KEYWORD+'\s+((\w+,?)+)\s+(\w+)\s+->\s*((\s\w+)+).*$', flags=re.M )

# Default acceptable extensions for files during directory walking
EXTS = ['.c', '.cpp', '.h', '.hpp', '.f', '.jdf', '.f90', '.F90', '.f77', '.F77', '.cu', '.cuf', '.CUF']

def check_gen(filename, work):
    """Reads the file and determines if it needs generation.
    If so, appends it as tuple (filename, match-groups, text) to work array."""
    fd = open( filename, 'r')
    content = fd.read()
    fd.close()
    m = REGEX.search( content )
    if ( m ):
        work.append( (filename, m.groups(), content) )


def visible(filename):
    """Exclude hidden files"""
    return not filename.startswith('.')


def valid_extension(filename):
    """Exclude non-valid extensions"""
    global EXTS
    for ext in EXTS:
        if filename.endswith(ext):
            return True
    return False

# os.path.relpath not in python 2.4 (ancient, but still default on some machines)
def relpath(p):
    """Get the relative path of a file."""
    p = path.realpath(p)
    return p.replace(path.realpath('.')+'/', '')


class Conversion:
    """This class works on a single file to create generations"""

    # ---- static class variables
    # Is the conversion in debug mode? More verbose.
    debug = False

    # Is the conversion in test mode? No real work.
    test = False

    # Is the conversion in make mode? Output make commands.
    make = False

    # What (if any) prefix is specified for the output folder?
    # If None, use the file's resident folder.
    prefix = None
    required_precisions = []

    # A running list of files that are input.
    files_in = []

    # A running list of files that are output.
    files_out = []

    # A running list of files that are output.
    dependencies = []

    def __init__(self, file=None, match=None, content=None):
        """Constructor that takes a file, match, and content.
        @param file The file name of the input.
        @param match The regular expression matches
        @param content The ASCII content of the file.
        """
        if file is None: return
        self.content = content
        #file = path.realpath(file)
        rel = relpath(file)
        self.file = path.split(file)
        self.date = path.getmtime(file)
        if sys.platform!="win32" and path.samefile( path.join( self.file[0], self.file[1] ), sys.argv[0]):
            raise ValueError('Let\'s just forget codegen.py')
        try:
            # ['normal', 'all', 'mixed'] for example. These are the replacement types to be used.
            self.types = match[0].split(',')
            # 'z' for example. This is the current file's `type`.
            self.precision = match[2].lower()
            # ['c', 'd', 's'] for example. This is the current file's destination `types`.
            self.precisions = match[3].lower().split()
            if len(self.required_precisions):
                self.precstmp = []
                for prec in self.required_precisions:
                    if prec in self.precisions or prec == self.precision:
                        self.precstmp.append(prec)
                self.precisions = self.precstmp
        except:
            raise ValueError(path.join(self.file[0], self.file[1])+' : Invalid conversion string')
        self.files_in.append(rel)

    def run(self):
        """Does the appropriate work, if in test mode, this is limited to only converting names."""
        if self.convert_names() and not self.test:
            # If not in test mode, actually make changes and export to disk.
            self.convert_data()
            self.export_data()

    def convert_names(self):
        """Investigate file name and make appropriate changes."""
        self.names = []
        self.dates = []
        self.copy = []
        self.converted = []
        load = False
        if self.debug: print '|'.join(self.types), self.precision, relpath(path.join(self.file[0], self.file[1]))
        for precision in self.precisions:
            # For each destination precision, make the appropriate changes to the file name/data.
            new_file = self.convert(self.file[1], precision)
            if self.debug: print precision, ':',
            copy = False
            if new_file != self.file[1] or self.prefix is not None:
                if self.prefix is None:
                    # If no prefix is specified, use the file's current folder.
                    prefix = ''
                    makeprefix = ''
                else:
                    # If a prefix is specified, set it up.
                    prefix = self.prefix
                    makeprefix = '--prefix '+prefix
                    if new_file == self.file[1]:
                        copy = True
                # Where the destination file will reside.
                conversion = path.join(prefix, new_file)
                file_out = relpath(conversion)
                if self.make:
                    # If in GNU Make mode, write the rule to create the file.
                    file_in = relpath(path.join(self.file[0], self.file[1]))
                    print file_out+':', file_in
                    print "\t$(PYTHON)", sys.argv[0], makeprefix, '-p', precision, file_in
                self.names.append(new_file)
                self.files_out.append(file_out)
                self.dependencies.append( (path.join(self.file[0], self.file[1]), precision, file_out) )
                if self.debug: print relpath(conversion), ':',
                try:
                    # Try to emulate Make like time based dependencies.
                    date = path.getmtime(conversion)
                    diff = self.date - date
                    self.dates.append(diff)
                    if self.debug:
                        if diff > 0: print 'Old',
                        else: print 'Current',
                        print diff
                    if diff > 0: load = True
                except:
                    if self.debug: print 'Missing'
                    self.dates.append(None)
                    load = True
            elif precision != self.precision:
                # There was no change in the file's name, thus,
                # no work can be done without overwriting the original.
                if self.debug: print '<No Change>', ':'
                else: print >> sys.stderr, new_file, 'had no change for', precision
                self.names.append(None)
                self.dates.append(None)
            self.copy.append(copy)
        return load

    def export_data(self):
        """After all of the conversions are complete,
        this will write the output file contents to the disk."""
        for i in range(len(self.names)):
            name = self.names[i]
            data = self.converted[i]
            copy = self.copy[i]
            if copy:
                shutil.copy(self.files_in[i], self.files_out[i])
                continue
            if data is None or name is None: continue
            fd = open(self.files_out[i], 'w')
            fd.write(data)
            fd.close()

    def convert_data(self):
        """Convert the data in the files by making the
        appropriate replacements for each destination precision."""
        for i in range(len(self.precisions)):
            precision = self.precisions[i]
            name = self.names[i]
            date = self.dates[i]
            copy = self.copy[i]
            if name is not None and not copy and (date is None or date > 0):
                self.converted.append(self.convert(self.content, precision))
            else: self.converted.append(None)

    def substitute(self, sub_type, data, precision):
        """This operates on a single replacement type.
        @param sub_type The name of the replacement set.
        @param data The content subject for replacments.
        @param precision The target precision for replacements.
        """
        try:
            # Get requested substitution tables based on sub_type,
            # and columns based on precisions (row 0 is header),
            # then loop over all substitutions (excluding row 0).
            subs_o = subs[sub_type]
            subs_s = subs_search[sub_type]
            subs_r = subs_replace[sub_type]
            jto   = subs_r[0].index(precision)
            jfrom = subs_r[0].index(self.precision)
            for (orig, search, replace) in zip( subs_o[1:], subs_s[1:], subs_r[1:] ):
                # cache compiled regexp
                if search[jfrom] is None:
                    search[jfrom] = re.compile( orig[jfrom] )
                data = re.sub( search[jfrom], replace[jto], data )
        except Exception, e:
            print >>sys.stderr, 'substitution failed:', e, sub_type, self.precision, '->', precision, subs_r[0]
        return data

    def convert(self, data, precision):
        """Select appropriate replacements for the current file.
        @param data The content subject for the replacements.
        @param precision The target precision for generation.
        """
        global KEYWORD, DONE_KEYWORD
        for sub_type in self.types:
            # For all the conversion types for the current file,
            # make the correct replacements.
            data = self.substitute(sub_type, data, precision)
        # Replace the replacement keywork with one that signifies this is an output file,
        # to prevent multiple replacement issues if run again.
        data = re.sub( KEYWORD+' '+','.join(self.types)+'.*',
                       DONE_KEYWORD+' from '+self.file[1]+' '+','.join(self.types)+' '+self.precision+' -> '+precision+', '+datetime.now().ctime(), data );
        return data
# end class Conversion
