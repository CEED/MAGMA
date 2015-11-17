#!/usr/bin/env python
#
# Parses output from testers into Python numpy arrays or Matlab arrays.
# Replaces words and "---" with NAN.
# Attempts to name arrays based on the filename, MAGMA tester, and options.
#
# @author Mark Gates

import sys
import os
import re
import numpy
import optparse


# --------------------
# parse command line options

parser = optparse.OptionParser()
parser.add_option( '-m', '--matlab', action='store_true', help='output Matlab' )
parser.add_option( '-p', '--python', action='store_true', help='output Python (default)', default=True )
(opts, args) = parser.parse_args()

if ( opts.matlab ):
	opts.python = False
	comment = '%'
else:
	comment = '#'


# --------------------
# print header

if ( opts.python ):
	print 'import numpy'
	print 'from numpy import array, nan, inf'


# --------------------
# dump one table of data

def dump( name ):
	global rows, maxwidths, filename
	
	if ( not name ):
		return
	
	if ( rows ):
		formats = map( lambda x: '%%%ds' % x, maxwidths )
		if ( opts.python ):
			format = '\t[ ' + ',  '.join( formats ) + ' ],'
		else:
			format = '\t' + ',  '.join( formats )
		
		# differentiate data from multiple files
		if ( len(args) > 1 ):
			(base,ext) = os.path.splitext( filename )
			base = re.sub( '[^\w]', '_', base )
			name = base + '_' + name
			#name += '_' + base
		# end
		
		print
		if ( cmd ):
			print comment, cmd
		if ( opts.python ):
			print name, '= array(['
		else:
			print name, '= ['
		for row in rows:
			try:
				print format % tuple(row)
			except Exception, e:
				print >>sys.stderr, e, "format:", format, "row:", row
		if ( opts.python ):
			print '])'
		else:
			print ']'
		
		rows = []
		maxwidths = []
	# end
# end


# --------------------
# process files

# if no arguments, or special argmument '-', read from stdin
if ( len(args) == 0 ):
	args.append( '-' )

cmd     = None
ngpu    = None
threads = None
for filename in args:
	rows = []
	maxwidths = []
	name = 'unknown'
	#lastname = None
	
	if ( filename == '-' ):
		infile = sys.stdin
	else:
		infile = open( filename )
	for line in infile:
		m = re.search( '^(numactl|\./testing)', line )
		if ( m ):
			dump( name )  # output previous results
			cmd = line.rstrip()
			continue
		# end
		
		m = re.search( '--ngpu +(\d+)', line )
		if ( m ):
			ngpu = m.group(1)
		
		m = re.search( 'MAGMA_NUM_GPUS +(\d+)', line )
		if ( m ):
			ngpu = m.group(1)
		
		m = re.search( 'MKL_NUM_THREADS +(\d+)', line )
		if ( m ):
			threads = m.group(1)
		
		m = re.search( r'^(?:\% *)?Usage: ./testing_(\w+)', line )  # for magma
		if ( m ):
			dump( name )  # output previous results
			
			name = m.group(1)
			if ( threads ):
				name += '_t' + threads
			
			if ( ngpu ):
				name += '_ngpu' + ngpu
			
			m = re.search( '-m (\d+)', line )
			if ( m ): name += '_m' + m.group(1)
			
			m = re.search( '-n (\d+)', line )
			if ( m ): name += '_n' + m.group(1)
			
			m = re.search( '-U', line )
			if ( m ): name += '_upper'
			
			m = re.search( '-SR', line )
			if ( m ): name += '_right'
			
			m = re.search( '-DU', line )
			if ( m ): name += '_unit'
			
			m = re.search( '-TT', line )
			if ( m ): name += '_trans'
			continue
		# end
		
		m = re.search( 'time_(\w+)', line )          # for plasma
		if ( m ):
			name = m.group(1)
			continue
		
		m = re.search( 'Nb threads: +(\d+)', line )  # for plasma
		if ( m ):
			name += '_t' + m.group(1)
			continue
		
		m = re.search( 'NB: +(\d+)', line )          # for plasma
		if ( m ):
			name += '_nb' + m.group(1)
			continue
		
		m = re.search( '\d+ +\d+|---', line )        # data, for plasma & magma
		if ( m ):
			line2 = re.sub( r'\b[a-zA-Z]+\b', ' nan ', line )
			line2 = re.sub( r'\s---\s', ' nan ', line2 )
			line2 = re.sub( '[()]', ' ', line2 ).strip()
			fields = re.split( ' +', line2 )
			
			# verify that everything is numeric
			# this ignores things like date (Thu Apr 30 19:14:36 EDT 2015)
			try:
				map( float, fields )
			except:
				#print >>sys.stderr, 'ignoring:', line.strip()
				continue
			
			rows.append( fields )
			widths = map( len, fields )
			for i in xrange( len( widths )):
				if ( i > len( maxwidths )):
					print 'error'
				elif ( i == len( maxwidths )):
					maxwidths.append( widths[i] )
				else:
					maxwidths[i] = max( maxwidths[i], widths[i] )
				# end
			# end
			continue
		# end
		
		#print 'unknown', line
	# end
	
	# output last data rows
	dump( name )
# end
