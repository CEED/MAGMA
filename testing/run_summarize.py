#!/usr/bin/env python
#
# MAGMA (version 2.0) --
# Univ. of Tennessee, Knoxville
# Univ. of California, Berkeley
# Univ. of Colorado, Denver
# @date

## @file run_summarize.py
#  @author Mark Gates
#
# Usage:
# First run tests, saving output in tests.txt:
#     ./run_tests.py [options] > tests.txt
#     or
#     ./testing_xyz  [options] > tests.txt
#
# Then parse their output:
#     ./run_summarize.py test.txt
#
# Parses the output of MAGMA testers and sorts tests into categories:
#     ok:              test passed
#     suspect:         test failed, but  error < tol2*eps, so it's probably ok
#     failed:          test failed, with error > tol2*eps
#     error:           segfault, etc.
#     known failures:  commands that we already know have issues
#     ignore:          ignore issues like using --ngpu 2 with only 1 gpu.
#
# tol  is tolerance given when the tester was run, by default tol = 30.
# tol2 is specified here, using --tol, by default tol2 = 100.
#
# For each suspect or failed command, prints the command and suspect or failed
# tests. Also adds the ratio {error/eps} in braces after each error.
# Tests that passed are not output by default.
#
# This is helpful to re-parse and summarize output from run-tests.py, and
# to apply a second tolerence to separate true failures from borderline cases.

import re
import sys

from optparse import OptionParser

parser = OptionParser()
parser.add_option( '--tol',  action='store',      dest='tol',  help='set tolerance (tol2)', default='100' )
parser.add_option( '--okay', action='store_true', dest='okay', help='print okay tests',     default=False )

(opts, args) = parser.parse_args()


# --------------------
tol2     = int(opts.tol)

seps     = 5.96e-08
deps     = 1.11e-16

stol_30  = 30.0   * seps
dtol_30  = 30.0   * deps

stol_100 = 100.0  * seps
dtol_100 = 100.0  * deps

stol_1k  = 1000.0 * seps
dtol_1k  = 1000.0 * deps

print 'single epsilon %.2e,  tol %.0f,  tol*eps %.2e,  30*eps %.2e,  100*eps %.2e,  1000*eps %.2e' % (seps, tol2, tol2*seps, 30*seps, 100*seps, 1000*seps)
print 'double epsilon %.2e,  tol %.0f,  tol*eps %.2e,  30*eps %.2e,  100*eps %.2e,  1000*eps %.2e' % (deps, tol2, tol2*deps, 30*deps, 100*deps, 1000*deps)

epsilons = {
	's': seps,
	'c': seps,
	'd': deps,
	'z': deps,
}


# --------------------
# hash of cmd: row, where each row is an array of fields.
# Most fields are an array of text lines.
# The known field is a boolean flag.
data = {}

# fields in each row
OKAY    = 0
SUSPECT = 1
FAILED  = 2
ERROR   = 3
IGNORE  = 4
OTHER   = 5
KNOWN   = 6


# --------------------
# input: re match object of floating point number
# returns number + (number/eps)
def ratio( match ):
	global g_fail, eps
	s     = match.group(1)
	error = float( s )
	ratio = error / eps
	g_fail |= (ratio >= tol2)
	return s + ' {%7.1f}' % (ratio)
# end


# --------------------
size = None
for filename in args:
	cmd = 'unknown'
	# fields:     [ okay, susp, fail, err,  ignr, othr, known ]
	data[ cmd ] = [ [],   [],   [],   [],   [],   [],   False ]
	
	eps = epsilons['d']  # double by default
	
	fopen = open( filename )
	for line in fopen:
		line = line.rstrip()
		
		# -----
		# look for command
		# we only want up to first -c, --range, or -N size
		m = re.search( r'^(?:cuda-memcheck +)?(./testing.*?) (-c|--range|-N)', line )
		if ( m ):
			if ( size ):
				data[ cmd ][ OTHER ].append( size )
			size = None
			cmd = m.group(1)
			# gemv|hemv_mgpu|symv_mgpu|hetrd_mgpu|sytrd_mgpu|cblas_[cz]
			known = (re.search( 'geqr2x_gpu.*--version +[24]', cmd ) != None)
			if ( not data.has_key( cmd )):
				# fields:     [ okay, susp, fail, err,  ignr, othr, known ]
				data[ cmd ] = [ [],   [],   [],   [],   [],   [],   known ]
			m = re.search( 'testing_([scdz])', cmd )
			if ( m ):
				eps = epsilons[ m.group(1) ]
		
		# -----
		# look for line with a couple optional words and a size, e.g.:
		# "1234 ..."
		# "upper  1234 ..."
		# "vector  upper  1234 ..."
		# this will be prepended to subsequent non-size lines containing failures
		if ( re.search( '^ *([a-zA-Z]\w* +){0,2}\d+ ', line )):
			if ( size ):
				data[ cmd ][ OTHER ].append( size )
			size = line
		
		# -----
		# look for errors (segfaults, etc.)
		if   ( re.search( 'exit|memory leak|memory mapping error|CUDA runtime error|illegal value|ERROR SUMMARY: [1-9]', line )):
			if ( size and size != line ):
				line = size + '\n' + line
			size = None
			data[ cmd ][ ERROR ].append( line )
		
		# ignore if no multi-GPUs
		elif ( re.search( '--ngpu \d+ exceeds number of CUDA devices', line )):
			if ( size ):
				data[ cmd ][ OTHER ].append( size )
			size = None
			data[ cmd ][ IGNORE ].append( line )
		
		# look for accuracy failures
		elif ( re.search( 'failed', line )
		       and not re.search( r'\*\* \d+ tests failed', line )
		       and not re.search( r'^ *\d+ tests failed accuracy test', line )):
			if ( size and size != line ):
				line = size + '\n' + line
			size = None
			g_fail = False
			line = re.sub( ' (\d\.\d+e[+-]\d+)', ratio, line )
			if ( g_fail ):
				data[ cmd ][ FAILED  ].append( line )
			else:
				line = re.sub( 'failed', 'suspect', line )
				data[ cmd ][ SUSPECT ].append( line )
		
		# look for "ok"
		elif ( re.search( r' ok *$', line )):
			if ( size and size != line ):
				line = size + '\n' + line
			size = None
			data[ cmd ][ OKAY ].append( line )
		# end
# end


# labels for each field
labels = [
	'okay tests',
	'suspicious tests (tol2*eps > error > tol*eps)',
	'failed tests (error > tol2*eps)',
	'errors (segfault, etc.)',
	'ignored (e.g., ngpu unavailable)',
	'other (lines that did not get matched)',
	'known failures',
]


# ------------------------------------------------------------
# Processes commands that have tests in the given field.
# If output is true, prints commands and tests in given field,
# otherwise just prints count of the commands and tests.
# If field is KNOWN, prints tests that are suspect, failed, or error.
def output( field, output ):
	cmds  = 0
	tests = 0
	result = ''
	for cmd in sorted( data.keys()):
		row = data[cmd]
		if ( field == KNOWN ):
			if ( row[KNOWN] ):
				result += cmd + '\n'
				num = len(row[SUSPECT]) + len(row[FAILED]) + len(row[ERROR])
				if ( num == 0 ):
					result += 'no failures (has ' + cmd + ' been fixed?)\n\n'
				elif ( output ):
					#result += cmd + '\n'
					if ( opts.okay and len(row[OKAY]) > 0 ):
						result += labels[OKAY]    + ':\n' + '\n'.join( row[OKAY]    ) + '\n'
					if ( len(row[SUSPECT]) > 0 ):
						result += labels[SUSPECT] + ':\n' + '\n'.join( row[SUSPECT] ) + '\n'
					if ( len(row[FAILED]) > 0 ):
						result += labels[FAILED]  + ':\n' + '\n'.join( row[FAILED]  ) + '\n'
					if ( len(row[ERROR]) > 0 ):
						result += labels[ERROR]   + ':\n' + '\n'.join( row[ERROR]   ) + '\n'
					result += '\n'
				# nd
				cmds  += 1
				tests += num
		else:
			num = len( row[field] )
			if ( num > 0 and not row[KNOWN] ):
				if ( output ):
					result += cmd + '\n'
					result += '\n'.join( row[field] ) + '\n\n'
				# end
				cmds  += 1
				tests += num
			# end
		# end
	# end
	print '#' * 120
	print '%-50s %3d commands, %6d tests' % (labels[field]+':', cmds, tests )
	print result
	print
# end


output( OKAY,    opts.okay )
output( ERROR,   True   )
output( FAILED,  True   )
output( SUSPECT, True   )
output( KNOWN,   True   )
output( IGNORE,  True   )
#output( OTHER,   True   )  # tests that didn't have "ok" or "failed"
