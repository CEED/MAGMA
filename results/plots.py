from __future__ import print_function

import matplotlib.pyplot as pp
import matplotlib
import numpy

import pyutil

# ----------------------------------------------------------------------
# import versions and setup color for each
import v150_cuda70_k40c
import v160_cuda70_k40c
import v161_cuda70_k40c

versions = [
	v150_cuda70_k40c,
	v160_cuda70_k40c,
	v161_cuda70_k40c,
]

# get nice distribution of colors from purple (old versions) to red (new versions)
x = numpy.linspace( 0, 1, len(versions) )
rainbow = matplotlib.cm.get_cmap('rainbow')
colors = rainbow(x)

for i in xrange( len(versions) ):
	versions[i].color = colors[i,:]
# end

# ----------------------------------------------------------------------
# defaults
pp.rcParams['legend.fontsize'] = 10

figsize = [9, 7]

# ----------------------------------------------------------------------
# column indices
getrf_m         = 0
getrf_n         = 1
getrf_cpu_flops = 2
getrf_cpu_time  = 3
getrf_gpu_flops = 4
getrf_gpu_time  = 5
getrf_error     = 6

potrf_n         = 0
potrf_cpu_flops = 1
potrf_cpu_time  = 2
potrf_gpu_flops = 3
potrf_gpu_time  = 4
potrf_error     = 5

geqrf_m         = 0
geqrf_n         = 1
geqrf_cpu_flops = 2
geqrf_cpu_time  = 3
geqrf_gpu_flops = 4
geqrf_gpu_time  = 5
geqrf_error     = 6

geev_n          = 0
geev_cpu_time   = 1
geev_gpu_time   = 2
geev_error      = 3

syev_n          = 0
syev_cpu_time   = 1
syev_gpu_time   = 2
syev_error      = 3

# note: testing_.gesvd outputs jobu & jobv; we throw out jobv in python data file
# to match testing_.gesdd which has only job.
svd_job         = 0
svd_m           = 1
svd_n           = 2
svd_cpu_time    = 3
svd_gpu_time    = 4
svd_error       = 5

symv_n             = 0
symv_gpu_flops     = 1
symv_gpu_time      = 2
symv_atomics_flops = 3
symv_atomics_time  = 4
symv_cublas_flops  = 5
symv_cublas_time   = 6
symv_cpu_flops     = 7
symv_cpu_time      = 8
symv_gpu_error     = 9
symv_atomics_error = 10
symv_cublas_error  = 11


# ----------------------------------------------------------------------
def plot_getrf_data( data, style='.-', color='y', label=None ):
	pp.semilogx( data[:,getrf_m], data[:,getrf_gpu_flops], style, color=color, label=label );
# end

def plot_getrf_labels( title=None ):
	if ( title ):
		pp.title( title )
	pp.legend( loc='upper left' )
	pp.xlabel( r'size (log scale)' )
	pp.ylabel( r'Gflop/s' )
	xticks = [ 10, 100, 1000, 10000 ]
	pp.xticks( xticks, xticks )
	pp.xlim( 9, 20000 )
	pp.grid( True )
# end

def plot_getrf( versions ):
	pp.figure( 1 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('sgetrf'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_getrf_data( v.sgetrf,     '.-', color=v.color, label=v.version+' sgetrf'     )
		plot_getrf_data( v.sgetrf_gpu, 'x-', color=v.color, label=v.version+' sgetrf_gpu' )
		
		pp.subplot( 2, 2, 2 )
		plot_getrf_data( v.dgetrf,     '.-', color=v.color, label=v.version+' dgetrf'     )
		plot_getrf_data( v.dgetrf_gpu, 'x-', color=v.color, label=v.version+' dgetrf_gpu' )
		
		pp.subplot( 2, 2, 3 )
		plot_getrf_data( v.cgetrf,     '.-', color=v.color, label=v.version+' cgetrf'     )
		plot_getrf_data( v.cgetrf_gpu, 'x-', color=v.color, label=v.version+' cgetrf_gpu' )
		
		pp.subplot( 2, 2, 4 )
		plot_getrf_data( v.zgetrf,     '.-', color=v.color, label=v.version+' zgetrf'     )
		plot_getrf_data( v.zgetrf_gpu, 'x-', color=v.color, label=v.version+' zgetrf_gpu' )
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_getrf_labels()
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
# end


# ----------------------------------------------------------------------
def plot_potrf_data( data, style='.-', color='y', label=None ):
	pp.semilogx( data[:,potrf_n], data[:,potrf_gpu_flops], style, color=color, label=label );
# end

def plot_potrf_labels( title=None ):
	if ( title ):
		pp.title( title )
	pp.legend( loc='upper left' )
	pp.xlabel( r'size (log scale)' )
	pp.ylabel( r'Gflop/s' )
	xticks = [ 10, 100, 1000, 10000 ]
	pp.xticks( xticks, xticks )
	pp.xlim( 9, 20000 )
	pp.grid( True )
# end

def plot_potrf( versions ):
	pp.figure( 1 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('spotrf'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_potrf_data( v.spotrf,     '.-', color=v.color, label=v.version+' spotrf'     )
		plot_potrf_data( v.spotrf_gpu, 'x-', color=v.color, label=v.version+' spotrf_gpu' )
		
		pp.subplot( 2, 2, 2 )
		plot_potrf_data( v.dpotrf,     '.-', color=v.color, label=v.version+' dpotrf'     )
		plot_potrf_data( v.dpotrf_gpu, 'x-', color=v.color, label=v.version+' dpotrf_gpu' )
		
		pp.subplot( 2, 2, 3 )
		plot_potrf_data( v.cpotrf,     '.-', color=v.color, label=v.version+' cpotrf'     )
		plot_potrf_data( v.cpotrf_gpu, 'x-', color=v.color, label=v.version+' cpotrf_gpu' )
		
		pp.subplot( 2, 2, 4 )
		plot_potrf_data( v.zpotrf,     '.-', color=v.color, label=v.version+' zpotrf'     )
		plot_potrf_data( v.zpotrf_gpu, 'x-', color=v.color, label=v.version+' zpotrf_gpu' )
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_potrf_labels()
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
# end


# ----------------------------------------------------------------------
def plot_geqrf_data( data, style='.-', color='y', label=None ):
	pp.semilogx( data[:,geqrf_m], data[:,geqrf_gpu_flops], style, color=color, label=label );
# end

def plot_geqrf_labels( title=None ):
	if ( title ):
		pp.title( title )
	pp.legend( loc='upper left' )
	pp.xlabel( r'size (log scale)' )
	pp.ylabel( r'Gflop/s' )
	xticks = [ 10, 100, 1000, 10000 ]
	pp.xticks( xticks, xticks )
	pp.xlim( 9, 20000 )
	pp.grid( True )
# end

def plot_geqrf( versions ):
	pp.figure( 1 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('sgeqrf'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_geqrf_data( v.sgeqrf,     '.-', color=v.color, label=v.version+' sgeqrf'     )
		plot_geqrf_data( v.sgeqrf_gpu, 'x-', color=v.color, label=v.version+' sgeqrf_gpu' )
		
		pp.subplot( 2, 2, 2 )
		plot_geqrf_data( v.dgeqrf,     '.-', color=v.color, label=v.version+' dgeqrf'     )
		plot_geqrf_data( v.dgeqrf_gpu, 'x-', color=v.color, label=v.version+' dgeqrf_gpu' )
		
		pp.subplot( 2, 2, 3 )
		plot_geqrf_data( v.cgeqrf,     '.-', color=v.color, label=v.version+' cgeqrf'     )
		plot_geqrf_data( v.cgeqrf_gpu, 'x-', color=v.color, label=v.version+' cgeqrf_gpu' )
		
		pp.subplot( 2, 2, 4 )
		plot_geqrf_data( v.zgeqrf,     '.-', color=v.color, label=v.version+' zgeqrf'     )
		plot_geqrf_data( v.zgeqrf_gpu, 'x-', color=v.color, label=v.version+' zgeqrf_gpu' )
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_geqrf_labels()
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
# end


# ----------------------------------------------------------------------
def plot_geev_data( data, vec, style='.-', color='y', label=None ):
	n = data[:,geev_n]
	t = data[:,geev_gpu_time]
	if ( vec ):
		gflop = 1e-9 * 10/3. * n**3 # TODO
	else:
		gflop = 1e-9 * 10/3. * n**3 # TODO
	pp.semilogx( n, gflop/t, style, color=color, label=label );
# end

def plot_geev_labels( title, vec ):
	if ( title ):
		pp.title( title )
	pp.legend( loc='upper left' )
	pp.xlabel( r'size (log scale)' )
	if ( vec ):
		pp.ylabel( r'Gflop/s $\frac{10n^3}{3t}$ TODO' )
	else:
		pp.ylabel( r'Gflop/s $\frac{10n^3}{3t}$ TODO' )
	#pp.ylabel( 'time (sec)' )
	xticks = [ 10, 100, 1000, 10000 ]
	pp.xticks( xticks, xticks )
	pp.xlim( 9, 20000 )
	pp.grid( True )
# end

def plot_geev( versions ):
	pp.figure( 1 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('sgeev_RN'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_geev_data( v.sgeev_RN, False, '.-', color=v.color, label=v.version+' sgeev' )
		
		pp.subplot( 2, 2, 2 )
		plot_geev_data( v.dgeev_RN, False, '.-', color=v.color, label=v.version+' dgeev' )
		
		pp.subplot( 2, 2, 3 )
		plot_geev_data( v.cgeev_RN, False, '.-', color=v.color, label=v.version+' cgeev' )
		
		pp.subplot( 2, 2, 4 )
		plot_geev_data( v.zgeev_RN, False, '.-', color=v.color, label=v.version+' zgeev' )
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_geev_labels( 'no vectors', False )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
	
	# --------------------
	pp.figure( 2 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('sgeev_RV'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_geev_data( v.sgeev_RV, True, '.-', color=v.color, label=v.version+' sgeev' )
		
		pp.subplot( 2, 2, 2 )
		plot_geev_data( v.dgeev_RV, True, '.-', color=v.color, label=v.version+' dgeev' )
		
		pp.subplot( 2, 2, 3 )
		plot_geev_data( v.cgeev_RV, True, '.-', color=v.color, label=v.version+' cgeev' )
		
		pp.subplot( 2, 2, 4 )
		plot_geev_data( v.zgeev_RV, True, '.-', color=v.color, label=v.version+' zgeev' )
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_geev_labels( 'with right vectors', True )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
# end


# ----------------------------------------------------------------------
def plot_syev_data( data, vec, style='.-', color='y', label=None ):
	n = data[:,syev_n]
	t = data[:,syev_gpu_time]
	if ( vec ):
		gflop = 1e-9 * 14/3. * n**3
	else:
		gflop = 1e-9 * 4/3. * n**3
	pp.semilogx( n, gflop/t, style, color=color, label=label );
# end

def plot_syev_labels( title, vec ):
	if ( title ):
		pp.title( title )
	pp.legend( loc='upper left' )
	pp.xlabel( r'size (log scale)' )
	if ( vec ):
		pp.ylabel( r'Gflop/s $\frac{14n^3}{3t}$' )  # TODO
	else:
		pp.ylabel( r'Gflop/s $\frac{4n^3}{3t}$' )
	#pp.ylabel( 'time (sec)' )
	xticks = [ 10, 100, 1000, 10000 ]
	pp.xticks( xticks, xticks )
	pp.xlim( 9, 20000 )
	pp.grid( True )
# end

def plot_syev( versions ):
	pp.figure( 1 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('ssyevd_JN'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_syev_data( v.ssyevd_JN,     False, '.-', color=v.color, label=v.version+' ssyevd'     )
		plot_syev_data( v.ssyevd_gpu_JN, False, 'x-', color=v.color, label=v.version+' ssyevd_gpu' )
		
		pp.subplot( 2, 2, 2 )
		plot_syev_data( v.dsyevd_JN,     False, '.-', color=v.color, label=v.version+' dsyevd'     )
		plot_syev_data( v.dsyevd_gpu_JN, False, 'x-', color=v.color, label=v.version+' dsyevd_gpu' )
		
		pp.subplot( 2, 2, 3 )
		plot_syev_data( v.cheevd_JN,     False, '.-', color=v.color, label=v.version+' cheevd'     )
		plot_syev_data( v.cheevd_gpu_JN, False, 'x-', color=v.color, label=v.version+' cheevd_gpu' )
		
		pp.subplot( 2, 2, 4 )
		plot_syev_data( v.zheevd_JN,     False, '.-', color=v.color, label=v.version+' zheevd'     )
		plot_syev_data( v.zheevd_gpu_JN, False, 'x-', color=v.color, label=v.version+' zheevd_gpu' )
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_syev_labels( 'no vectors', False )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
	
	# --------------------
	pp.figure( 2 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('ssyevd_JV'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_syev_data( v.ssyevd_JV,     True, '.-', color=v.color, label=v.version+' ssyevd'     )
		plot_syev_data( v.ssyevd_gpu_JV, True, 'x-', color=v.color, label=v.version+' ssyevd_gpu' )
		
		pp.subplot( 2, 2, 2 )
		plot_syev_data( v.dsyevd_JV,     True, '.-', color=v.color, label=v.version+' dsyevd'     )
		plot_syev_data( v.dsyevd_gpu_JV, True, 'x-', color=v.color, label=v.version+' dsyevd_gpu' )
		
		pp.subplot( 2, 2, 3 )
		plot_syev_data( v.cheevd_JV,     True, '.-', color=v.color, label=v.version+' cheevd'     )
		plot_syev_data( v.cheevd_gpu_JV, True, 'x-', color=v.color, label=v.version+' cheevd_gpu' )
		
		pp.subplot( 2, 2, 4 )
		plot_syev_data( v.zheevd_JV,     True, '.-', color=v.color, label=v.version+' zheevd'     )
		plot_syev_data( v.zheevd_gpu_JV, True, 'x-', color=v.color, label=v.version+' zheevd_gpu' )
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_syev_labels( 'with vectors', True )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
# end

def plot_syev_2stage( versions ):
	pp.figure( 3 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('ssyevdx_2stage_JN'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_syev_data( v.ssyevdx_2stage_JN, False, '.-', color=v.color, label=v.version+' ssyevdx_2stage' )
		
		pp.subplot( 2, 2, 2 )
		plot_syev_data( v.dsyevdx_2stage_JN, False, '.-', color=v.color, label=v.version+' dsyevdx_2stage' )
		
		pp.subplot( 2, 2, 3 )
		plot_syev_data( v.cheevdx_2stage_JN, False, '.-', color=v.color, label=v.version+' cheevdx_2stage' )
		
		pp.subplot( 2, 2, 4 )
		plot_syev_data( v.zheevdx_2stage_JN, False, '.-', color=v.color, label=v.version+' zheevdx_2stage' )
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_syev_labels( 'no vectors', False )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
	
	# --------------------
	pp.figure( 4 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('ssyevdx_2stage_JV'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_syev_data( v.ssyevdx_2stage_JV, True, '.-', color=v.color, label=v.version+' ssyevdx_2stage' )
		
		pp.subplot( 2, 2, 2 )
		plot_syev_data( v.dsyevdx_2stage_JV, True, '.-', color=v.color, label=v.version+' dsyevdx_2stage' )
		
		pp.subplot( 2, 2, 3 )
		plot_syev_data( v.cheevdx_2stage_JV, True, '.-', color=v.color, label=v.version+' cheevdx_2stage' )
		
		pp.subplot( 2, 2, 4 )
		plot_syev_data( v.zheevdx_2stage_JV, True, '.-', color=v.color, label=v.version+' zheevdx_2stage' )
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_syev_labels( 'with vectors', True )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
# end


# ----------------------------------------------------------------------
def plot_gesvd_data( data, vec, style='.-', color='y', label=None, ratio=1 ):
	(ii,) = numpy.where( data[:,svd_m] == ratio*data[:,svd_n] )
	mn = numpy.zeros(( len(ii), 2 ))
	mn[:,0] = data[ii,svd_m]
	mn[:,1] = data[ii,svd_n]
	M = mn.max( axis=1 )  # M = max(m,n)
	N = mn.min( axis=1 )  # N = min(m,n)
	
	# with vectors (approx.)
	# 2*m*n*(m + 3*n)    => 24n^3/3 = 8n^3 if m == n
	# 8*n**2*(3*m + n)/3 # tall QR optimization
	#
	# no vectors
	# 4*n**2*(3*m - n)/3 => 8n^3/3 if m == n
	# 2*n**2*(3*m - n)   # tall QR optimization
	
	if ( vec ):
		gflop = 1e-9 * 2*M*N*(M + 3*N)
	else:
		gflop = 1e-9 * 4*N**2*(3*M - N)/3
	t = data[ii,svd_gpu_time]
	
	pp.semilogx( N, gflop/t, style, color=color, label=label );
# end

def plot_gesvd_labels( title, vec, square ):
	if ( title ):
		pp.title( title )
	pp.legend( loc='upper left' )
	pp.xlabel( r'size, max(M,N)' )
	if ( vec ):
		if ( square ):
			pp.ylabel( r'Gflop/s $\frac{8n^3}{t}$' )
		else:
			pp.ylabel( r'Gflop/s $\frac{2mn(m + 3n)}{t}$' )
	else:
		if ( square ):
			pp.ylabel( r'Gflop/s $\frac{8n^3}{3t}$' )
		else:
			pp.ylabel( r'Gflop/s $\frac{4n^2(3m - n)}{3t}$' )
	#pp.ylabel( 'time (sec)' )
	xticks = [ 10, 100, 1000, 10000 ]
	pp.xticks( xticks, xticks )
	pp.xlim( 9, 20000 )
	pp.grid( True )
# end

def plot_gesvd( versions, ratio=1 ):
	pp.figure( 1 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('sgesvd_UN'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_gesvd_data( v.sgesvd_UN, False, '.-', color=v.color, label=v.version+' sgesvd', ratio=ratio )
		
		pp.subplot( 2, 2, 2 )
		plot_gesvd_data( v.dgesvd_UN, False, '.-', color=v.color, label=v.version+' dgesvd', ratio=ratio )
		
		pp.subplot( 2, 2, 3 )
		plot_gesvd_data( v.cgesvd_UN, False, '.-', color=v.color, label=v.version+' cgesvd', ratio=ratio )
		
		if v.__dict__.has_key('zgesvd_UN'):
			pp.subplot( 2, 2, 4 )
			plot_gesvd_data( v.zgesvd_UN, False, '.-', color=v.color, label=v.version+' zgesvd', ratio=ratio )
	# end
	m = ratio if (ratio >= 1) else 1
	n = 1     if (ratio >= 1) else 1/ratio
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_gesvd_labels( 'no vectors, M:N ratio %.3g:%.3g' % (m,n), False )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
	
	# --------------------
	pp.figure( 2 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('sgesvd_US'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_gesvd_data( v.sgesvd_US, True, '.-', color=v.color, label=v.version+' sgesvd', ratio=ratio )
		
		pp.subplot( 2, 2, 2 )
		plot_gesvd_data( v.dgesvd_US, True, '.-', color=v.color, label=v.version+' dgesvd', ratio=ratio )
		
		pp.subplot( 2, 2, 3 )
		plot_gesvd_data( v.cgesvd_US, True, '.-', color=v.color, label=v.version+' cgesvd', ratio=ratio )
		
		if v.__dict__.has_key('zgesvd_US'):
			pp.subplot( 2, 2, 4 )
			plot_gesvd_data( v.zgesvd_US, True, '.-', color=v.color, label=v.version+' zgesvd', ratio=ratio )
	# end
	m = ratio if (ratio >= 1) else 1
	n = 1     if (ratio >= 1) else 1/ratio
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_gesvd_labels( 'some vectors, M:N ratio %.3g:%.3g' % (m,n), True )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
# end

def plot_gesdd( versions, ratio=1 ):
	pp.figure( 3 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('sgesdd_UN'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_gesvd_data( v.sgesdd_UN, False, '.-', color=v.color, label=v.version+' sgesdd', ratio=ratio )
		
		pp.subplot( 2, 2, 2 )
		plot_gesvd_data( v.dgesdd_UN, False, '.-', color=v.color, label=v.version+' dgesdd', ratio=ratio )
		
		pp.subplot( 2, 2, 3 )
		plot_gesvd_data( v.cgesdd_UN, False, '.-', color=v.color, label=v.version+' cgesdd', ratio=ratio )
		
		if v.__dict__.has_key('zgesdd_UN'):
			pp.subplot( 2, 2, 4 )
			plot_gesvd_data( v.zgesdd_UN, False, '.-', color=v.color, label=v.version+' zgesdd', ratio=ratio )
	# end
	m = ratio if (ratio >= 1) else 1
	n = 1     if (ratio >= 1) else 1/ratio
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_gesvd_labels( 'no vectors, M:N ratio %.3g:%.3g' % (m,n), False )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
	
	# --------------------
	pp.figure( 4 )
	pp.clf()
	
	for v in versions:
		if not v.__dict__.has_key('sgesdd_US'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_gesvd_data( v.sgesdd_US, True, '.-', color=v.color, label=v.version+' sgesdd', ratio=ratio )
		
		pp.subplot( 2, 2, 2 )
		plot_gesvd_data( v.dgesdd_US, True, '.-', color=v.color, label=v.version+' dgesdd', ratio=ratio )
		
		pp.subplot( 2, 2, 3 )
		plot_gesvd_data( v.cgesdd_US, True, '.-', color=v.color, label=v.version+' cgesdd', ratio=ratio )
		
		if v.__dict__.has_key('zgesdd_US'):
			pp.subplot( 2, 2, 4 )
			plot_gesvd_data( v.zgesdd_US, True, '.-', color=v.color, label=v.version+' zgesdd', ratio=ratio )
	# end
	m = ratio if (ratio >= 1) else 1
	n = 1     if (ratio >= 1) else 1/ratio
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_gesvd_labels( 'some vectors, M:N ratio %.3g:%.3g' % (m,n), True )
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
# end


# ----------------------------------------------------------------------
def plot_symv_data( data, style='.-', color='y', label=None, first=False ):
	if ( first ):
		pp.semilogx( data[:,symv_n], data[:,symv_atomics_flops], 'k--', color='#aaaaaa', label='cublas atomics' )
		pp.semilogx( data[:,symv_n], data[:,symv_cublas_flops],  'k-.', color='#aaaaaa', label='cublas'  )
		pp.semilogx( data[:,symv_n], data[:,symv_cpu_flops],     'k-',                   label='MKL'     )
	pp.semilogx( data[:,symv_n], data[:,symv_gpu_flops], style, color=color, label=label )
# end

def plot_symv_labels( title=None ):
	if ( title ):
		pp.title( title )
	pp.legend( loc='upper left' )
	pp.xlabel( r'size (log scale)' )
	pp.ylabel( r'Gflop/s' )
	xticks = [ 10, 100, 1000, 10000 ]
	pp.xticks( xticks, xticks )
	pp.xlim( 9, 20000 )
	pp.grid( True )
# end

def plot_symv( versions ):
	pp.figure( 1 )
	pp.clf()
	
	first = True
	for v in versions:
		if not v.__dict__.has_key('ssymv'): continue
		
		pp.subplot( 2, 2, 1 )
		plot_symv_data( v.ssymv, '-', color=v.color, label=v.version+' ssymv', first=first )
		
		pp.subplot( 2, 2, 2 )
		plot_symv_data( v.dsymv, '-', color=v.color, label=v.version+' dsymv', first=first )
		
		pp.subplot( 2, 2, 3 )
		plot_symv_data( v.chemv, '-', color=v.color, label=v.version+' chemv', first=first )
		
		pp.subplot( 2, 2, 4 )
		plot_symv_data( v.zhemv, '-', color=v.color, label=v.version+' zhemv', first=first )
		first = False
	# end
	for i in xrange( 1, 5 ):
		pp.subplot( 2, 2, i )
		plot_symv_labels()
	# end
	pyutil.plot_resize( figsize )
	pp.tight_layout( pad=1 )
# end


# ----------------------------------------------------------------------
print('''Available plots:
plot_symv(  versions )
plot_getrf( versions )
plot_potrf( versions )
plot_geqrf( versions )
plot_geev(  versions )
plot_syev(  versions )
plot_syev_2stage( versions )
plot_gesvd( versions, ratio=1 ); where ratio m:n in { 1, 3, 100, 1/3., 1/100. }
plot_gesdd( versions, ratio=1 ); where ratio m:N in { 1, 3, 100, 1/3., 1/100. }

Available versions:''')

for i in xrange( len(versions) ):
	print( "versions[%d] = %s" % (i, versions[i].version) )
# end
