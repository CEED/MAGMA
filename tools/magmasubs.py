#!/usr/bin/env python
# Substitutions are applied in the order listed. This is important in cases
# where multiple substitutions could match, or when one substitution matches
# the result of a previous substitution. For example, these rules are correct
# in this order:
#
#    ('real',   'double precision',  'real',   'double precision' ),  # before double
#    ('float',  'double',            'float',  'double'           ),
#
# but if switched would translate 'double precision' -> 'float precision',
# which is wrong.
#
# Reorganized 5/2012 Mark Gates



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#                                                                             #
#          DO NOT EDIT      OpenCL and MIC versions      DO NOT EDIT          #
#          DO NOT EDIT      OpenCL and MIC versions      DO NOT EDIT          #
#          DO NOT EDIT      OpenCL and MIC versions      DO NOT EDIT          #
#                                                                             #
# Please edit the CUDA MAGMA version, then copy it to MIC and OpenCL MAGMA.   #
# Otherwise they get out-of-sync and it's really hard to sync them up again.  #
#                                                                             #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #



# Dictionary is keyed on substitution type (mixed, normal, etc.)
subs = {
  # ------------------------------------------------------------
  # replacements applied to mixed precision files.
  'mixed' : [
    # ----- Special line indicating column types
    ['ds',                        'zc'                      ],
    
    # ----- Mixed precision prefix
    ('DS',                        'ZC'                      ),
    ('ds',                        'zc'                      ),
    
    # ----- Preprocessor
    ('#define PRECISION_d',       '#define PRECISION_z'     ),
    ('#define PRECISION_s',       '#define PRECISION_c'     ),
    ('#undef PRECISION_d',        '#undef PRECISION_z'      ),
    ('#undef PRECISION_s',        '#undef PRECISION_c'      ),
    
    # ----- Data types
    ('double',                    'double2'                 ),
    ('float',                     'float2'                  ),
    ('double',                    'cuDoubleComplex'         ),
    ('float',                     'cuFloatComplex'          ),
    ('double',                    'MKL_Complex16'           ),
    ('float',                     'MKL_Complex8'            ),
    ('magmaFloat_const_ptr',      'magmaFloatComplex_const_ptr' ),  # before magmaDoubleComplex
    ('magmaDouble_const_ptr',     'magmaDoubleComplex_const_ptr'),
    ('magmaFloat_ptr',            'magmaFloatComplex_ptr'   ),
    ('magmaDouble_ptr',           'magmaDoubleComplex_ptr'  ),
    ('double',                    'magmaDoubleComplex'      ),
    ('float',                     'magmaFloatComplex'       ),
    ('DOUBLE PRECISION',          'COMPLEX_16'              ),
    ('SINGLE PRECISION',          'COMPLEX'                 ),
    ('real',                      'complex'                 ),
    
    # ----- Text
    ('symmetric',                 'hermitian',              ),
    ('symmetric',                 'Hermitian',              ),
    ('\*\*T',                     '\*\*H',                  ),
    
    # ----- CBLAS
    ('',                          'CBLAS_SADDR'             ),
    
    # ----- Complex numbers
    ('(double)',                  'cuComplexFloatToDouble'  ),
    ('(float)',                   'cuComplexDoubleToFloat'  ),
    ('',                          'cuCrealf'                ),
    ('',                          'cuCimagf'                ),
    ('',                          'cuCreal'                 ),
    ('',                          'cuCimag'                 ),
    ('',                          'cuConj'                  ),
    ('abs',                       'cuCabs'                  ),
    ('absf',                      'cuCabsf'                 ),
    
    # ----- PLASMA / MAGMA
    ('magma_sdgetrs',             'magma_czgetrs'           ),
    
    # ----- Constants
    ('CblasTrans',                'CblasConjTrans'          ),
    ('MagmaTrans',                'MagmaConjTrans'          ),
    
    # ----- BLAS and LAPACK, lowercase, alphabetic order
    # copy & paste these to uppercase below and fix case.
    # mixed precision
    ('dsaxpy',                    'zcaxpy'                  ),
    ('dslaswp',                   'zclaswp'                 ),
    
    # real == complex name
    ('daxpy',                     'zaxpy'                   ),
    ('dcopy',                     'zcopy'                   ),
    ('dgemm',                     'zgemm'                   ),
    ('dgeqrf',                    'zgeqrf'                  ),
    ('dgeqrs',                    'zgeqrs'                  ),
    ('dgesv',                     'zgesv'                   ),
    ('dgetrf',                    'zgetrf'                  ),
    ('dgetrs',                    'zgetrs'                  ),
    ('dlacpy',                    'zlacpy'                  ),
    ('dlange',                    'zlange'                  ),
    ('dlansy',                    'zlansy'                  ),
    ('dlarnv',                    'zlarnv'                  ),
    ('dpotrf',                    'zpotrf'                  ),
    ('dpotrs',                    'zpotrs'                  ),
    ('dtrmm',                     'ztrmm'                   ),
    ('dtrsm',                     'ztrsm'                   ),
    ('dtrsv',                     'ztrsv'                   ),
    ('idamax',                    'izamax'                  ),
    ('spotrf',                    'cpotrf'                  ),
    ('strmm',                     'ctrmm'                   ),
    ('strsm',                     'ctrsm'                   ),
    ('strsv',                     'ctrsv'                   ),
    
    # real != complex name
    ('ddot',                      'zdotc'                   ),
    ('dlag2s',                    'zlag2c'                  ),
    ('dlagsy',                    'zlaghe'                  ),
    ('dlansy',                    'zlanhe'                  ),
    ('dlat2s',                    'zlat2c'                  ),
    ('dnrm2',                     'dznrm2'                  ),
    ('dormqr',                    'zunmqr'                  ),
    ('dsymm',                     'zhemm'                   ),
    ('dsymv',                     'zhemv'                   ),
    ('dsyrk',                     'zherk'                   ),
    ('slag2d',                    'clag2z'                  ),
    ('slansy',                    'clanhe'                  ),
    ('slat2d',                    'clat2z'                  ),
    
    # ----- BLAS AND LAPACK, UPPERCASE, ALPHABETIC ORDER
    # MIXED PRECISION
    ('DSAXPY',                    'ZCAXPY'                  ),
    ('DSLASWP',                   'ZCLASWP'                 ),
    
    # REAL == COMPLEX NAME
    ('DAXPY',                     'ZAXPY'                   ),
    ('DCOPY',                     'ZCOPY'                   ),
    ('DGEMM',                     'ZGEMM'                   ),
    ('DGEQRF',                    'ZGEQRF'                  ),
    ('DGEQRS',                    'ZGEQRS'                  ),
    ('DGESV',                     'ZGESV'                   ),
    ('DGETRF',                    'ZGETRF'                  ),
    ('DGETRS',                    'ZGETRS'                  ),
    ('DLACPY',                    'ZLACPY'                  ),
    ('DLANGE',                    'ZLANGE'                  ),
    ('DLANSY',                    'ZLANSY'                  ),
    ('DLARNV',                    'ZLARNV'                  ),
    ('DPOTRF',                    'ZPOTRF'                  ),
    ('DPOTRS',                    'ZPOTRS'                  ),
    ('DTRMM',                     'ZTRMM'                   ),
    ('DTRSM',                     'ZTRSM'                   ),
    ('DTRSV',                     'ZTRSV'                   ),
    ('IDAMAX',                    'IZAMAX'                  ),
    ('SPOTRF',                    'CPOTRF'                  ),
    ('STRMM',                     'CTRMM'                   ),
    ('STRSM',                     'CTRSM'                   ),
    ('STRSV',                     'CTRSV'                   ),
    
    # REAL != COMPLEX NAME
    ('DDOT',                      'ZDOTC'                   ),
    ('DLAG2S',                    'ZLAG2C'                  ),
    ('DLAGSY',                    'ZLAGHE'                  ),
    ('DLANSY',                    'ZLANHE'                  ),
    ('DLAT2S',                    'ZLAT2C'                  ),
    ('DNRM2',                     'DZNRM2'                  ),
    ('DORMQR',                    'ZUNMQR'                  ),
    ('DSYMM',                     'ZHEMM'                   ),
    ('DSYMV',                     'ZHEMV'                   ),
    ('DSYRK',                     'ZHERK'                   ),
    ('SLAG2D',                    'CLAG2Z'                  ),
    ('SLANSY',                    'CLANHE'                  ),
    ('SLAT2D',                    'CLAT2Z'                  ),
    
    # ----- Sparse Stuff
    ('dspgmres',                  'zcpgmres'                ),
    ('dspbicgstab',               'zcpbicgstab'             ),
    ('dsir',                      'zcir'                    ),
    ('dspir',                     'zcpir'                   ),
    
    # ----- Prefixes
    ('blasf77_d',                 'blasf77_z'               ),
    ('blasf77_s',                 'blasf77_c'               ),
    ('cublasId',                  'cublasIz'                ),
    ('cublasD',                   'cublasZ'                 ),
    ('cublasS',                   'cublasC'                 ),
    ('clAmdBlasD',                'clAmdBlasZ'              ),
    ('clAmdBlasS',                'clAmdBlasC'              ),
    ('lapackf77_d',               'lapackf77_z'             ),
    ('lapackf77_s',               'lapackf77_c'             ),
    ('MAGMA_D',                   'MAGMA_Z'                 ),
    ('MAGMA_S',                   'MAGMA_C'                 ),
    ('magmablas_d',               'magmablas_z'             ),
    ('magmablas_s',               'magmablas_c'             ),
    ('magma_d',                   'magma_z'                 ),
    ('magma_s',                   'magma_c'                 ),
    ('magma_get_d',               'magma_get_z'             ),
    ('magma_get_s',               'magma_get_c'             ),
    ('magmasparse_ds',            'magmasparse_zc'          ),
  ],
  
  # ------------------------------------------------------------
  # replacements applied to most files.
  'normal' : [
    # ----- Special line indicating column types
    # old python (2.4) requires this line to be list [] rather than tuple () to use index() function.
    ['s',              'd',              'c',              'z'               ],
    
    # ----- Preprocessor
    ('#define PRECISION_s', '#define PRECISION_d', '#define PRECISION_c', '#define PRECISION_z' ),
    ('#undef PRECISION_s',  '#undef PRECISION_d',  '#undef PRECISION_c',  '#undef PRECISION_z'  ),
    ('#define REAL',        '#define REAL',        '#define COMPLEX',     '#define COMPLEX'     ),
    ('#undef REAL',         '#undef REAL',         '#undef COMPLEX',      '#undef COMPLEX'      ),
    ('#define SINGLE',      '#define DOUBLE',      '#define SINGLE',      '#define DOUBLE'      ),
    ('#undef SINGLE',       '#undef DOUBLE',       '#undef SINGLE',       '#undef DOUBLE'       ),
    
    # ----- Data types
    ('REAL',                'DOUBLE PRECISION',    'REAL',                'DOUBLE PRECISION'    ),
    ('real',                'double precision',    'real',                'double precision'    ),  # before double
    ('float',               'double',              'float _Complex',      'double _Complex'     ),
    ('float',               'double',              'cuFloatComplex',      'cuDoubleComplex'     ),
    ('float',               'double',              'MKL_Complex8',        'MKL_Complex16'       ),
    ('magmaFloat_const_ptr', 'magmaDouble_const_ptr', 'magmaFloatComplex_const_ptr', 'magmaDoubleComplex_const_ptr'),  # before magmaDoubleComplex
    ('magmaFloat_const_ptr', 'magmaDouble_const_ptr', 'magmaFloat_const_ptr',        'magmaDouble_const_ptr'       ),  # before magmaDoubleComplex
    ('magmaFloat_ptr',       'magmaDouble_ptr',       'magmaFloatComplex_ptr',       'magmaDoubleComplex_ptr'      ),  # before magmaDoubleComplex
    ('magmaFloat_ptr',       'magmaDouble_ptr',       'magmaFloat_ptr',              'magmaDouble_ptr'             ),  # before magmaDoubleComplex
    ('float',               'double',              'magmaFloatComplex',   'magmaDoubleComplex'  ),
    ('float',               'double',              'PLASMA_Complex32_t',  'PLASMA_Complex64_t'  ),
    ('PlasmaRealFloat',     'PlasmaRealDouble',    'PlasmaComplexFloat',  'PlasmaComplexDouble' ),
    ('real',                'double precision',    'complex',             'complex\*16'         ),
    ('REAL',                'DOUBLE_PRECISION',    'COMPLEX',             'COMPLEX_16'          ),
    ('REAL',                'DOUBLE PRECISION',    'COMPLEX',             'COMPLEX\*16'         ),
    ('sizeof_real',         'sizeof_double',       'sizeof_complex',      'sizeof_complex_16'   ),  # before complex
    ('real',                'real',                'complex',             'complex'             ),
    ('float',               'double',              'float2',              'double2'             ),
    ('float',               'double',              'float',               'double'              ),
    
    # ----- Text
    ('symmetric',      'symmetric',      'hermitian',      'hermitian'       ),
    ('symmetric',      'symmetric',      'Hermitian',      'Hermitian'       ),
    ('\*\*T',          '\*\*T',          '\*\*H',          '\*\*H'           ),
    ('%f',             '%lf',            '%f',             '%lf'             ),  # for scanf
    
    # ----- CBLAS
    ('',               '',               'CBLAS_SADDR',    'CBLAS_SADDR'     ),
    
    # ----- Complex numbers
    # \b regexp here avoids conjugate -> conjfugate, and fabs -> fabsf -> fabsff.
    # The \b is deleted from replacement strings.
    ('',               '',               'conjf',         r'conj\b'          ),
    ('fabsf',         r'\bfabs\b',       'cabsf',          'cabs'            ),
    ('',               '',               'cuCrealf',       'cuCreal'         ),
    ('',               '',               'cuCimagf',       'cuCimag'         ),
    ('',               '',               'cuConjf',        'cuConj'          ),
    ('fabsf',         r'\bfabs\b',       'cuCabsf',        'cuCabs'          ),
    
    # ----- PLASMA / MAGMA, alphabetic order
    ('bsy2trc',        'bsy2trc',        'bhe2trc',        'bhe2trc'         ),
    ('magma_ssqrt',    'magma_dsqrt',    'magma_ssqrt',    'magma_dsqrt'     ),
    ('SAUXILIARY',     'DAUXILIARY',     'CAUXILIARY',     'ZAUXILIARY'      ),
    ('sauxiliary',     'dauxiliary',     'cauxiliary',     'zauxiliary'      ),
    ('sb2st',          'sb2st',          'hb2st',          'hb2st'           ),
    ('sbcyclic',       'dbcyclic',       'cbcyclic',       'zbcyclic'        ),
    ('sbulge',         'dbulge',         'cbulge',         'zbulge'          ),
    ('SBULGE',         'DBULGE',         'CBULGE',         'ZBULGE'          ),
    ('SCODELETS',      'DCODELETS',      'CCODELETS',      'ZCODELETS'       ),
    ('sgeadd',         'dgeadd',         'cgeadd',         'zgeadd'          ),
    ('sgecfi',         'dgecfi',         'cgecfi',         'zgecfi'          ),
    ('sget',           'dget',           'cget',           'zget'            ),
    ('sgetrl',         'dgetrl',         'cgetrl',         'zgetrl'          ),
    ('slocality',      'dlocality',      'clocality',      'zlocality'       ),
    ('smalloc',        'dmalloc',        'cmalloc',        'zmalloc'         ),
    ('smalloc',        'dmalloc',        'smalloc',        'dmalloc'         ),
    ('smove',          'dmove',          'smove',          'dmove'           ),
    ('spanel_to_q',    'dpanel_to_q',    'cpanel_to_q',    'zpanel_to_q'     ),
    ('spermute',       'dpermute',       'cpermute',       'zpermute'        ),
    ('sprint',         'dprint',         'cprint',         'zprint'          ),
    ('SPRINT',         'DPRINT',         'CPRINT',         'ZPRINT'          ),
    ('sprint',         'dprint',         'sprint',         'dprint'          ),
    ('sprofiling',     'dprofiling',     'cprofiling',     'zprofiling'      ),
    ('sq_to_panel',    'dq_to_panel',    'cq_to_panel',    'zq_to_panel'     ),
    ('sset',           'dset',           'cset',           'zset'            ),
    ('ssign',          'dsign',          'ssign',          'dsign'           ),
    ('SSIZE',          'DSIZE',          'CSIZE',          'ZSIZE'           ),
    ('ssplit',         'dsplit',         'csplit',         'zsplit'          ),
    ('stile',          'dtile',          'ctile',          'ztile'           ),
    ('stranspose',     'dtranspose',     'ctranspose',     'ztranspose'      ),
    ('STRANSPOSE',     'DTRANSPOSE',     'CTRANSPOSE',     'ZTRANSPOSE'      ),
    ('sy2sb',          'sy2sb',          'he2hb',          'he2hb'           ),
    ('szero',          'dzero',          'czero',          'zzero'           ),
    
    # ----- Constants
    ('CblasTrans',     'CblasTrans',     'CblasConjTrans', 'CblasConjTrans'  ),
    ('MagmaTrans',     'MagmaTrans',     'MagmaConjTrans', 'MagmaConjTrans'  ),
    ('PlasmaTrans',    'PlasmaTrans',    'PlasmaConjTrans','PlasmaConjTrans' ),
    
    # ----- special cases for d -> s that need complex (e.g., testing_dgeev)
    # c/z precisions are effectively disabled for these rules
    ('caxpy',             'zaxpy',              'cccccccc', 'zzzzzzzz' ),
    ('clange',            'zlange',             'cccccccc', 'zzzzzzzz' ),
    ('cuFloatComplex',    'cuDoubleComplex',    'cccccccc', 'zzzzzzzz' ),
    ('magmaFloatComplex', 'magmaDoubleComplex', 'cccccccc', 'zzzzzzzz' ),
    ('MAGMA_C',           'MAGMA_Z',            'cccccccc', 'zzzzzzzz' ),
    
    # -----
    # BLAS, lowercase, real != complex name
    # add to lowercase, UPPERCASE, and Titlecase lists, alphabetic order
    ('sasum',          'dasum',          'scasum',         'dzasum'          ),
    ('sdot',           'ddot',           'cdotc',          'zdotc'           ),
    ('sdot_sub',       'ddot_sub',       'cdotc_sub',      'zdotc_sub'       ),
    ('sdot_sub',       'ddot_sub',       'cdotu_sub',      'zdotu_sub'       ),
    ('sger',           'dger',           'cgerc',          'zgerc'           ),
    ('sger',           'dger',           'cgeru',          'zgeru'           ),
    ('snrm2',          'dnrm2',          'scnrm2',         'dznrm2'          ),
    ('scnrm2',         'dznrm2',         'scnrm2',         'dznrm2'          ),  # where d -> s needs complex
    ('srot',           'drot',           'csrot',          'zdrot'           ),
    ('ssymm',          'dsymm',          'chemm',          'zhemm'           ),
    ('ssymv',          'dsymv',          'chemv',          'zhemv'           ),
    ('ssyr',           'dsyr',           'cher',           'zher'            ),
    #('ssyr2',          'dsyr2',          'cher2',          'zher2'           ),  # redundant with zher
    #('ssyr2k',         'dsyr2k',         'cher2k',         'zher2k'          ),  # redundant with zher
    #('ssyrk',          'dsyrk',          'cherk',          'zherk'           ),  # redundant with zher
    
    # BLAS, UPPERCASE, real != complex name
    ('SASUM',          'DASUM',          'SCASUM',         'DZASUM'          ),
    ('SDOT',           'DDOT',           'CDOTC',          'ZDOTC'           ),
    ('SDOT_SUB',       'DDOT_SUB',       'CDOTC_SUB',      'ZDOTC_SUB'       ),
    ('SDOT_SUB',       'DDOT_SUB',       'CDOTU_SUB',      'ZDOTU_SUB'       ),
    ('SGER',           'DGER',           'CGERC',          'ZGERC'           ),
    ('SGER',           'DGER',           'CGERU',          'ZGERU'           ),
    ('SNRM2',          'DNRM2',          'SCNRM2',         'DZNRM2'          ),
    ('SCNRM2',         'DZNRM2',         'SCNRM2',         'DZNRM2'          ),  # WHERE D -> S NEEDS COMPLEX
    ('SROT',           'DROT',           'CSROT',          'ZDROT'           ),
    ('SSYMM',          'DSYMM',          'CHEMM',          'ZHEMM'           ),
    ('SSYMV',          'DSYMV',          'CHEMV',          'ZHEMV'           ),
    ('SSYR',           'DSYR',           'CHER',           'ZHER'            ),
    
    # BLAS, Titlecase, real != complex name (for cublas)
    ('Sasum',          'Dasum',          'Scasum',         'Dzasum'          ),
    ('Sdot',           'Ddot',           'Cdotc',          'Zdotc'           ),
    ('Sdot_sub',       'Ddot_sub',       'Cdotc_sub',      'Zdotc_sub'       ),
    ('Sdot_sub',       'Ddot_sub',       'Cdotu_sub',      'Zdotu_sub'       ),
    ('Sger',           'Dger',           'Cgerc',          'Zgerc'           ),
    ('Sger',           'Dger',           'Cgeru',          'Zgeru'           ),
    ('Snrm2',          'Dnrm2',          'Scnrm2',         'Dznrm2'          ),
    ('Scnrm2',         'Dznrm2',         'Scnrm2',         'Dznrm2'          ),  # where d -> s needs complex
    ('Srot',           'Drot',           'Csrot',          'Zdrot'           ),
    ('Ssymm',          'Dsymm',          'Chemm',          'Zhemm'           ),
    ('Ssymv',          'Dsymv',          'Chemv',          'Zhemv'           ),
    ('Ssyr',           'Dsyr',           'Cher',           'Zher'            ),
    
    # few special cases lacking precision
    # !!! deprecated -- please include precision if you want precision generation to apply !!!
    #('symm',           'symm',           'hemm',           'hemm'            ),
    #('symv',           'symv',           'hemv',           'hemv'            ),
    #('syrk',           'syrk',           'herk',           'herk'            ),
    
    # -----
    # LAPACK, lowercase, real != complex name
    # add to both lowercase and UPPERCASE lists, alphabetic order
    ('slag2d',         'dlag2s',         'clag2z',         'zlag2c'          ),
    ('slagsy',         'dlagsy',         'claghe',         'zlaghe'          ),
    ('slansy',         'dlansy',         'clanhe',         'zlanhe'          ),
    ('slasrt',         'dlasrt',         'slasrt',         'dlasrt'          ),
    ('slasyf',         'dlasyf',         'clahef',         'zlahef'          ),
    ('slatms',         'dlatms',         'clatms',         'zlatms'          ),
    ('slavsy',         'dlavsy',         'clavhe',         'zlavhe'          ),
    ('sorg2r',         'dorg2r',         'cung2r',         'zung2r'          ),
    ('sorgbr',         'dorgbr',         'cungbr',         'zungbr'          ),
    ('sorghr',         'dorghr',         'cunghr',         'zunghr'          ),
    ('sorglq',         'dorglq',         'cunglq',         'zunglq'          ),
    ('sorgql',         'dorgql',         'cungql',         'zungql'          ),
    ('sorgqr',         'dorgqr',         'cungqr',         'zungqr'          ),
    ('sorgtr',         'dorgtr',         'cungtr',         'zungtr'          ),
    ('sorm2r',         'dorm2r',         'cunm2r',         'zunm2r'          ),
    ('sormbr',         'dormbr',         'cunmbr',         'zunmbr'          ),
    ('sormlq',         'dormlq',         'cunmlq',         'zunmlq'          ),
    ('sormql',         'dormql',         'cunmql',         'zunmql'          ),
    ('sormqr',         'dormqr',         'cunmqr',         'zunmqr'          ),
    ('sormr2',         'dormr2',         'cunmr2',         'zunmr2'          ),
    ('sormtr',         'dormtr',         'cunmtr',         'zunmtr'          ),
    ('sort01',         'dort01',         'cunt01',         'zunt01'          ),
    ('splgsy',         'dplgsy',         'cplghe',         'zplghe'          ),
    ('ssbtrd',         'dsbtrd',         'chbtrd',         'zhbtrd'          ),
    ('ssyev',          'dsyev',          'cheev',          'zheev'           ),
    ('ssyevd',         'dsyevd',         'cheevd',         'zheevd'          ),
    ('ssygs2',         'dsygs2',         'chegs2',         'zhegs2'          ),
    ('ssygst',         'dsygst',         'chegst',         'zhegst'          ),
    ('ssygvd',         'dsygvd',         'chegvd',         'zhegvd'          ),
    ('ssygvr',         'dsygvr',         'chegvr',         'zhegvr'          ),
    ('ssygvx',         'dsygvx',         'chegvx',         'zhegvx'          ),
    ('ssyt21',         'dsyt21',         'chet21',         'zhet21'          ),
    ('ssytd2',         'dsytd2',         'chetd2',         'zhetd2'          ),
    ('ssytrd',         'dsytrd',         'chetrd',         'zhetrd'          ),
    ('ssytrf',         'dsytrf',         'chetrf',         'zhetrf'          ),
    
    # LAPACK, UPPERCASE, real != complex name
    ('SLAG2D',         'DLAG2S',         'CLAG2Z',         'ZLAG2C'          ),
    ('SLAGSY',         'DLAGSY',         'CLAGHE',         'ZLAGHE'          ),
    ('SLANSY',         'DLANSY',         'CLANHE',         'ZLANHE'          ),
    ('SLASRT',         'DLASRT',         'SLASRT',         'DLASRT'          ),
    ('SLASYF',         'DLASYF',         'CLAHEF',         'ZLAHEF'          ),
    ('SLATMS',         'DLATMS',         'CLATMS',         'ZLATMS'          ),
    ('SLAVSY',         'DLAVSY',         'CLAVHE',         'ZLAVHE'          ),
    ('SORG2R',         'DORG2R',         'CUNG2R',         'ZUNG2R'          ),
    ('SORGBR',         'DORGBR',         'CUNGBR',         'ZUNGBR'          ),
    ('SORGHR',         'DORGHR',         'CUNGHR',         'ZUNGHR'          ),
    ('SORGLQ',         'DORGLQ',         'CUNGLQ',         'ZUNGLQ'          ),
    ('SORGQL',         'DORGQL',         'CUNGQL',         'ZUNGQL'          ),
    ('SORGQR',         'DORGQR',         'CUNGQR',         'ZUNGQR'          ),
    ('SORGTR',         'DORGTR',         'CUNGTR',         'ZUNGTR'          ),
    ('SORM2R',         'DORM2R',         'CUNM2R',         'ZUNM2R'          ),
    ('SORMBR',         'DORMBR',         'CUNMBR',         'ZUNMBR'          ),
    ('SORMLQ',         'DORMLQ',         'CUNMLQ',         'ZUNMLQ'          ),
    ('SORMQL',         'DORMQL',         'CUNMQL',         'ZUNMQL'          ),
    ('SORMQR',         'DORMQR',         'CUNMQR',         'ZUNMQR'          ),
    ('SORMR2',         'DORMR2',         'CUNMR2',         'ZUNMR2'          ),
    ('SORMTR',         'DORMTR',         'CUNMTR',         'ZUNMTR'          ),
    ('SORT01',         'DORT01',         'CUNT01',         'ZUNT01'          ),
    ('SPLGSY',         'DPLGSY',         'CPLGHE',         'ZPLGHE'          ),
    ('SSBTRD',         'DSBTRD',         'CHBTRD',         'ZHBTRD'          ),
    ('SSYEV',          'DSYEV',          'CHEEV',          'ZHEEV'           ),
    ('SSYEVD',         'DSYEVD',         'CHEEVD',         'ZHEEVD'          ),
    ('SSYGS2',         'DSYGS2',         'CHEGS2',         'ZHEGS2'          ),
    ('SSYGST',         'DSYGST',         'CHEGST',         'ZHEGST'          ),
    ('SSYGVD',         'DSYGVD',         'CHEGVD',         'ZHEGVD'          ),
    ('SSYGVR',         'DSYGVR',         'CHEGVR',         'ZHEGVR'          ),
    ('SSYGVX',         'DSYGVX',         'CHEGVX',         'ZHEGVX'          ),
    ('SSYT21',         'DSYT21',         'CHET21',         'ZHET21'          ),
    ('SSYTD2',         'DSYTD2',         'CHETD2',         'ZHETD2'          ),
    ('SSYTRD',         'DSYTRD',         'CHETRD',         'ZHETRD'          ),
    ('SSYTRF',         'DSYTRF',         'CHETRF',         'ZHETRF'          ),
    
    # -----
    # BLAS, lowercase, real == complex name
    # add to both lowercase and UPPERCASE lists, alphabetic order
    ('isamax',         'idamax',         'icamax',         'izamax'          ),
    ('isamax',         'idamax',         'isamax',         'idamax'          ),
    ('isamin',         'idamin',         'icamin',         'izamin'          ),
    ('saxpy',          'daxpy',          'caxpy',          'zaxpy'           ),
    ('scopy',          'dcopy',          'ccopy',          'zcopy'           ),
    ('sgemm',          'dgemm',          'cgemm',          'zgemm'           ),
    ('sgemv',          'dgemv',          'cgemv',          'zgemv'           ),
    ('srot',           'drot',           'srot',           'drot'            ),
    ('srot',           'drot',           'crot',           'zrot'            ),
    ('sscal',          'dscal',          'cscal',          'zscal'           ),
    ('sscal',          'dscal',          'csscal',         'zdscal'          ),
    ('sscal',          'dscal',          'sscal',          'dscal'           ),
    ('sswap',          'dswap',          'cswap',          'zswap'           ),
    ('ssymm',          'dsymm',          'csymm',          'zsymm'           ),
    ('ssymv',          'dsymv',          'csymv',          'zsymv'           ),
    ('ssyr2k',         'dsyr2k',         'csyr2k',         'zsyr2k'          ),
    ('ssyrk',          'dsyrk',          'csyrk',          'zsyrk'           ),
    ('strmm',          'dtrmm',          'ctrmm',          'ztrmm'           ),
    ('strmv',          'dtrmv',          'ctrmv',          'ztrmv'           ),
    ('strsm',          'dtrsm',          'ctrsm',          'ztrsm'           ),
    ('strsv',          'dtrsv',          'ctrsv',          'ztrsv'           ),
    
    # BLAS, UPPERCASE, real == complex name
    ('ISAMAX',         'IDAMAX',         'ICAMAX',         'IZAMAX'          ),
    ('ISAMAX',         'IDAMAX',         'ISAMAX',         'IDAMAX'          ),
    ('ISAMIN',         'IDAMIN',         'ICAMIN',         'IZAMIN'          ),
    ('SAXPY',          'DAXPY',          'CAXPY',          'ZAXPY'           ),
    ('SCOPY',          'DCOPY',          'CCOPY',          'ZCOPY'           ),
    ('SGEMM',          'DGEMM',          'CGEMM',          'ZGEMM'           ),
    ('SGEMV',          'DGEMV',          'CGEMV',          'ZGEMV'           ),
    ('SROT',           'DROT',           'SROT',           'DROT'            ),
    ('SROT',           'DROT',           'CROT',           'ZROT'            ),
    ('SSCAL',          'DSCAL',          'CSCAL',          'ZSCAL'           ),
    ('SSCAL',          'DSCAL',          'CSSCAL',         'ZDSCAL'          ),
    ('SSCAL',          'DSCAL',          'SSCAL',          'DSCAL'           ),
    ('SSWAP',          'DSWAP',          'CSWAP',          'ZSWAP'           ),
    ('SSYMM',          'DSYMM',          'CSYMM',          'ZSYMM'           ),
    ('SSYMV',          'DSYMV',          'CSYMV',          'ZSYMV'           ),
    ('SSYR2K',         'DSYR2K',         'CSYR2K',         'ZSYR2K'          ),
    ('SSYRK',          'DSYRK',          'CSYRK',          'ZSYRK'           ),
    ('STRMM',          'DTRMM',          'CTRMM',          'ZTRMM'           ),
    ('STRMV',          'DTRMV',          'CTRMV',          'ZTRMV'           ),
    ('STRSM',          'DTRSM',          'CTRSM',          'ZTRSM'           ),
    ('STRSV',          'DTRSV',          'CTRSV',          'ZTRSV'           ),
    
    # -----
    # LAPACK, lowercase, real == complex name
    # add to both lowercase and UPPERCASE lists, alphabetic order
    ('sbdsqr',         'dbdsqr',         'cbdsqr',         'zbdsqr'          ),
    ('sbdt01',         'dbdt01',         'cbdt01',         'zbdt01'          ),
    ('scheck',         'dcheck',         'ccheck',         'zcheck'          ),
    ('sgebak',         'dgebak',         'cgebak',         'zgebak'          ),
    ('sgebal',         'dgebal',         'cgebal',         'zgebal'          ),
    ('sgebd2',         'dgebd2',         'cgebd2',         'zgebd2'          ),
    ('sgebrd',         'dgebrd',         'cgebrd',         'zgebrd'          ),
    ('sgeev',          'dgeev',          'cgeev',          'zgeev'           ),
    ('sgegqr',         'dgegqr',         'cgegqr',         'zgegqr'          ),
    ('sgehd2',         'dgehd2',         'cgehd2',         'zgehd2'          ),
    ('sgehrd',         'dgehrd',         'cgehrd',         'zgehrd'          ),
    ('sgelq2',         'dgelq2',         'cgelq2',         'zgelq2'          ),
    ('sgelqf',         'dgelqf',         'cgelqf',         'zgelqf'          ),
    ('sgelqs',         'dgelqs',         'cgelqs',         'zgelqs'          ),
    ('sgels',          'dgels',          'cgels',          'zgels'           ),
    ('sgeqlf',         'dgeqlf',         'cgeqlf',         'zgeqlf'          ),
    ('sgeqp3',         'dgeqp3',         'cgeqp3',         'zgeqp3'          ),
    ('sgeqr2',         'dgeqr2',         'cgeqr2',         'zgeqr2'          ),
    ('sgeqrf',         'dgeqrf',         'cgeqrf',         'zgeqrf'          ),
    ('sgeqrs',         'dgeqrs',         'cgeqrs',         'zgeqrs'          ),
    ('sgeqrt',         'dgeqrt',         'cgeqrt',         'zgeqrt'          ),
    ('sgessm',         'dgessm',         'cgessm',         'zgessm'          ),
    ('sgesv',          'dgesv',          'cgesv',          'zgesv'           ),
    ('sgesv',          'sgesv',          'cgesv',          'cgesv'           ),
    ('sget22',         'dget22',         'cget22',         'zget22'          ),
    ('sgetf2',         'dgetf2',         'cgetf2',         'zgetf2'          ),
    ('sgetmi',         'dgetmi',         'cgetmi',         'zgetmi'          ),
    ('sgetmo',         'dgetmo',         'cgetmo',         'zgetmo'          ),
    ('sgetrf',         'dgetrf',         'cgetrf',         'zgetrf'          ),
    ('sgetri',         'dgetri',         'cgetri',         'zgetri'          ),
    ('sgetrs',         'dgetrs',         'cgetrs',         'zgetrs'          ),
    ('ssytrs',         'dsytrs',         'chetrs',         'zhetrs'          ),
    ('shseqr',         'dhseqr',         'chseqr',         'zhseqr'          ),
    ('shst01',         'dhst01',         'chst01',         'zhst01'          ),
    ('slabad',         'dlabad',         'slabad',         'dlabad'          ),
    ('slabrd',         'dlabrd',         'clabrd',         'zlabrd'          ),
    ('slacgv',         'dlacgv',         'clacgv',         'zlacgv'          ),
    ('slacpy',         'dlacpy',         'clacpy',         'zlacpy'          ),
    ('sladiv',         'dladiv',         'cladiv',         'zladiv'          ),
    ('slaed',          'dlaed',          'slaed',          'dlaed'           ),
    ('slaex',          'dlaex',          'slaex',          'dlaex'           ),
    ('slagsy',         'dlagsy',         'clagsy',         'zlagsy'          ),
    ('slahr',          'dlahr',          'clahr',          'zlahr'           ),
    ('slaln2',         'dlaln2',         'slaln2',         'dlaln2'          ),
    ('slamc3',         'dlamc3',         'slamc3',         'dlamc3'          ),
    ('slamch',         'dlamch',         'slamch',         'dlamch'          ),
    ('slamrg',         'dlamrg',         'slamrg',         'dlamrg'          ),
    ('slange',         'dlange',         'clange',         'zlange'          ),
    ('slapy3',         'dlapy3',         'slapy3',         'dlapy3'          ),
    ('slanst',         'dlanst',         'clanht',         'zlanht'          ),
    ('slansy',         'dlansy',         'clansy',         'zlansy'          ),
    ('slantr',         'dlantr',         'clantr',         'zlantr'          ),
    ('slaqps',         'dlaqps',         'claqps',         'zlaqps'          ),
    ('slaqp2',         'dlaqp2',         'claqp2',         'zlaqp2'          ),
    ('slaqtrs',        'dlaqtrs',        'claqtrs',        'zlaqtrs'         ),
    ('slarf',          'dlarf',          'clarf',          'zlarf'           ),  # handles zlarf[ bgtxy]
    ('slarnv',         'dlarnv',         'clarnv',         'zlarnv'          ),
    ('slarnv',         'dlarnv',         'slarnv',         'dlarnv'          ),
    ('slartg',         'dlartg',         'clartg',         'zlartg'          ),
    ('slascl',         'dlascl',         'clascl',         'zlascl'          ),
    ('slaset',         'dlaset',         'claset',         'zlaset'          ),
    ('slaswp',         'dlaswp',         'claswp',         'zlaswp'          ),
    ('slatrd',         'dlatrd',         'clatrd',         'zlatrd'          ),
    ('slatrs',         'dlatrs',         'clatrs',         'zlatrs'          ),
    ('slauum',         'dlauum',         'clauum',         'zlauum'          ),
    ('spack',          'dpack',          'cpack',          'zpack'           ),
    ('splgsy',         'dplgsy',         'cplgsy',         'zplgsy'          ),
    ('splrnt',         'dplrnt',         'cplrnt',         'zplrnt'          ),
    ('sposv',          'dposv',          'cposv',          'zposv'           ),
    ('sposv',          'sposv',          'cposv',          'cposv'           ),
    ('spotrf',         'dpotrf',         'cpotrf',         'zpotrf'          ),
    ('spotf2',         'dpotf2',         'cpotf2',         'zpotf2'          ),
    ('spotri',         'dpotri',         'cpotri',         'zpotri'          ),
    ('spotrs',         'dpotrs',         'cpotrs',         'zpotrs'          ),
    ('sqpt01',         'dqpt01',         'cqpt01',         'zqpt01'          ),
    ('sqrt02',         'dqrt02',         'cqrt02',         'zqrt02'          ),
    ('sshift',         'dshift',         'cshift',         'zshift'          ),
    ('sssssm',         'dssssm',         'cssssm',         'zssssm'          ),
    ('sstebz',         'dstebz',         'sstebz',         'dstebz'          ),
    ('sstedc',         'dstedc',         'cstedc',         'zstedc'          ),
    ('sstedx',         'dstedx',         'cstedx',         'zstedx'          ),
    ('sstedx',         'dstedx',         'sstedx',         'dstedx'          ),
    ('sstein',         'dstein',         'cstein',         'zstein'          ),
    ('sstemr',         'dstemr',         'cstemr',         'zstemr'          ),
    ('ssteqr',         'dsteqr',         'csteqr',         'zsteqr'          ),
    ('ssterf',         'dsterf',         'ssterf',         'dsterf'          ),
    ('ssterm',         'dsterm',         'csterm',         'zsterm'          ),
    ('sstt21',         'dstt21',         'cstt21',         'zstt21'          ),
    ('strevc',         'dtrevc',         'ctrevc',         'ztrevc'          ),
    ('strsmpl',        'dtrsmpl',        'ctrsmpl',        'ztrsmpl'         ),
    ('strtri',         'dtrtri',         'ctrtri',         'ztrtri'          ),
    ('stsmqr',         'dtsmqr',         'ctsmqr',         'ztsmqr'          ),
    ('stsqrt',         'dtsqrt',         'ctsqrt',         'ztsqrt'          ),
    ('ststrf',         'dtstrf',         'ctstrf',         'ztstrf'          ),
    ('sungesv',        'sungesv',        'cungesv',        'cungesv'         ),
    ('sstegr',         'dstegr',         'cstegr',         'zstegr'          ),
    
    # LAPACK, UPPERCASE, real == complex name
    ('SBDSQR',         'DBDSQR',         'CBDSQR',         'ZBDSQR'          ),
    ('SBDT01',         'DBDT01',         'CBDT01',         'ZBDT01'          ),
    ('SCHECK',         'DCHECK',         'CCHECK',         'ZCHECK'          ),
    ('SGEBAK',         'DGEBAK',         'CGEBAK',         'ZGEBAK'          ),
    ('SGEBAL',         'DGEBAL',         'CGEBAL',         'ZGEBAL'          ),
    ('SGEBD2',         'DGEBD2',         'CGEBD2',         'ZGEBD2'          ),
    ('SGEBRD',         'DGEBRD',         'CGEBRD',         'ZGEBRD'          ),
    ('SGEEV',          'DGEEV',          'CGEEV',          'ZGEEV'           ),
    ('SGEGQR',         'DGEGQR',         'CGEGQR',         'ZGEGQR'          ),
    ('SGEHD2',         'DGEHD2',         'CGEHD2',         'ZGEHD2'          ),
    ('SGEHRD',         'DGEHRD',         'CGEHRD',         'ZGEHRD'          ),
    ('SGELQ2',         'DGELQ2',         'CGELQ2',         'ZGELQ2'          ),
    ('SGELQF',         'DGELQF',         'CGELQF',         'ZGELQF'          ),
    ('SGELQS',         'DGELQS',         'CGELQS',         'ZGELQS'          ),
    ('SGELS',          'DGELS',          'CGELS',          'ZGELS'           ),
    ('SGEQLF',         'DGEQLF',         'CGEQLF',         'ZGEQLF'          ),
    ('SGEQP3',         'DGEQP3',         'CGEQP3',         'ZGEQP3'          ),
    ('SGEQR2',         'DGEQR2',         'CGEQR2',         'ZGEQR2'          ),
    ('SGEQRF',         'DGEQRF',         'CGEQRF',         'ZGEQRF'          ),
    ('SGEQRS',         'DGEQRS',         'CGEQRS',         'ZGEQRS'          ),
    ('SGEQRT',         'DGEQRT',         'CGEQRT',         'ZGEQRT'          ),
    ('SGESSM',         'DGESSM',         'CGESSM',         'ZGESSM'          ),
    ('SGESV',          'DGESV',          'CGESV',          'ZGESV'           ),
    ('SGESV',          'SGESV',          'CGESV',          'CGESV'           ),
    ('SGET22',         'DGET22',         'CGET22',         'ZGET22'          ),
    ('SGETF2',         'DGETF2',         'CGETF2',         'ZGETF2'          ),
    ('SGETMI',         'DGETMI',         'CGETMI',         'ZGETMI'          ),
    ('SGETMO',         'DGETMO',         'CGETMO',         'ZGETMO'          ),
    ('SGETRF',         'DGETRF',         'CGETRF',         'ZGETRF'          ),
    ('SGETRI',         'DGETRI',         'CGETRI',         'ZGETRI'          ),
    ('SGETRS',         'DGETRS',         'CGETRS',         'ZGETRS'          ),
    ('SSYTRS',         'DSYTRS',         'CHETRS',         'ZHETRS'          ),
    ('SHSEQR',         'DHSEQR',         'CHSEQR',         'ZHSEQR'          ),
    ('SHST01',         'DHST01',         'CHST01',         'ZHST01'          ),
    ('SLABAD',         'DLABAD',         'SLABAD',         'DLABAD'          ),
    ('SLABRD',         'DLABRD',         'CLABRD',         'ZLABRD'          ),
    ('SLACGV',         'DLACGV',         'CLACGV',         'ZLACGV'          ),
    ('SLACPY',         'DLACPY',         'CLACPY',         'ZLACPY'          ),
    ('SLADIV',         'DLADIV',         'CLADIV',         'ZLADIV'          ),
    ('SLAED',          'DLAED',          'SLAED',          'DLAED'           ),
    ('SLAEX',          'DLAEX',          'SLAEX',          'DLAEX'           ),
    ('SLAGSY',         'DLAGSY',         'CLAGSY',         'ZLAGSY'          ),
    ('SLAHR',          'DLAHR',          'CLAHR',          'ZLAHR'           ),
    ('SLALN2',         'DLALN2',         'SLALN2',         'DLALN2'          ),
    ('SLAMC3',         'DLAMC3',         'SLAMC3',         'DLAMC3'          ),
    ('SLAMCH',         'DLAMCH',         'SLAMCH',         'DLAMCH'          ),
    ('SLAMRG',         'DLAMRG',         'SLAMRG',         'DLAMRG'          ),
    ('SLANGE',         'DLANGE',         'CLANGE',         'ZLANGE'          ),
    ('SLAPY3',         'DLAPY3',         'SLAPY3',         'DLAPY3'          ),
    ('SLANST',         'DLANST',         'CLANHT',         'ZLANHT'          ),
    ('SLANSY',         'DLANSY',         'CLANSY',         'ZLANSY'          ),
    ('SLANTR',         'DLANTR',         'CLANTR',         'ZLANTR'          ),
    ('SLAQPS',         'DLAQPS',         'CLAQPS',         'ZLAQPS'          ),
    ('SLAQP2',         'DLAQP2',         'CLAQP2',         'ZLAQP2'          ),
    ('SLARF',          'DLARF',          'CLARF',          'ZLARF'           ),  # HANDLES ZLARF[ BGTXY]
    ('SLARNV',         'DLARNV',         'CLARNV',         'ZLARNV'          ),
    ('SLARNV',         'DLARNV',         'SLARNV',         'DLARNV'          ),
    ('SLARTG',         'DLARTG',         'CLARTG',         'ZLARTG'          ),
    ('SLASCL',         'DLASCL',         'CLASCL',         'ZLASCL'          ),
    ('SLASET',         'DLASET',         'CLASET',         'ZLASET'          ),
    ('SLASWP',         'DLASWP',         'CLASWP',         'ZLASWP'          ),
    ('SLATRD',         'DLATRD',         'CLATRD',         'ZLATRD'          ),
    ('SLATRS',         'DLATRS',         'CLATRS',         'ZLATRS'          ),
    ('SLAUUM',         'DLAUUM',         'CLAUUM',         'ZLAUUM'          ),
    ('SPACK',          'DPACK',          'CPACK',          'ZPACK'           ),
    ('SPLGSY',         'DPLGSY',         'CPLGSY',         'ZPLGSY'          ),
    ('SPLRNT',         'DPLRNT',         'CPLRNT',         'ZPLRNT'          ),
    ('SPOSV',          'DPOSV',          'CPOSV',          'ZPOSV'           ),
    ('SPOSV',          'SPOSV',          'CPOSV',          'CPOSV'           ),
    ('SPOTRF',         'DPOTRF',         'CPOTRF',         'ZPOTRF'          ),
    ('SPOTF2',         'DPOTF2',         'CPOTF2',         'ZPOTF2'          ),
    ('SPOTRI',         'DPOTRI',         'CPOTRI',         'ZPOTRI'          ),
    ('SPOTRS',         'DPOTRS',         'CPOTRS',         'ZPOTRS'          ),
    ('SQPT01',         'DQPT01',         'CQPT01',         'ZQPT01'          ),
    ('SQRT02',         'DQRT02',         'CQRT02',         'ZQRT02'          ),
    ('SSHIFT',         'DSHIFT',         'CSHIFT',         'ZSHIFT'          ),
    ('SSSSSM',         'DSSSSM',         'CSSSSM',         'ZSSSSM'          ),
    ('SSTEBZ',         'DSTEBZ',         'SSTEBZ',         'DSTEBZ'          ),
    ('SSTEDC',         'DSTEDC',         'CSTEDC',         'ZSTEDC'          ),
    ('SSTEDX',         'DSTEDX',         'CSTEDX',         'ZSTEDX'          ),
    ('SSTEDX',         'DSTEDX',         'SSTEDX',         'DSTEDX'          ),
    ('SSTEIN',         'DSTEIN',         'CSTEIN',         'ZSTEIN'          ),
    ('SSTEMR',         'DSTEMR',         'CSTEMR',         'ZSTEMR'          ),
    ('SSTEQR',         'DSTEQR',         'CSTEQR',         'ZSTEQR'          ),
    ('SSTERF',         'DSTERF',         'SSTERF',         'DSTERF'          ),
    ('SSTERM',         'DSTERM',         'CSTERM',         'ZSTERM'          ),
    ('SSTT21',         'DSTT21',         'CSTT21',         'ZSTT21'          ),
    ('STREVC',         'DTREVC',         'CTREVC',         'ZTREVC'          ),
    ('STRSMPL',        'DTRSMPL',        'CTRSMPL',        'ZTRSMPL'         ),
    ('STRTRI',         'DTRTRI',         'CTRTRI',         'ZTRTRI'          ),
    ('STSMQR',         'DTSMQR',         'CTSMQR',         'ZTSMQR'          ),
    ('STSQRT',         'DTSQRT',         'CTSQRT',         'ZTSQRT'          ),
    ('STSTRF',         'DTSTRF',         'CTSTRF',         'ZTSTRF'          ),
    ('SUNGESV',        'SUNGESV',        'CUNGESV',        'CUNGESV'         ),
    ('SSTEGR',         'DSTEGR',         'CSTEGR',         'ZSTEGR'          ),


    # ----- SPARSE BLAS
    ('cusparseS',      'cusparseD',      'cusparseC',      'cusparseZ'       ),
    ('sgecsrmv',       'dgecsrmv',       'cgecsrmv',       'zgecsrmv'        ),
    ('smgecsrmv',      'dmgecsrmv',      'cmgecsrmv',      'zmgecsrmv'       ),
    ('sgeellmv',       'dgeellmv',       'cgeellmv',       'zgeellmv'        ),
    ('smgeellmv',      'dmgeellmv',      'cmgeellmv',      'zmgeellmv'       ),
    ('sgeelltmv',      'dgeelltmv',      'cgeelltmv',      'zgeelltmv'       ),
    ('smgeelltmv',     'dmgeelltmv',     'cmgeelltmv',     'zmgeelltmv'      ),
    ('smdot',          'dmdot',          'cmdot',          'zmdot'           ),
    ('spipelined',     'dpipelined',     'cpipelined',     'zpipelined'      ),
    ('mkl_scsrmv',     'mkl_dcsrmv',     'mkl_ccsrmv',     'mkl_zcsrmv'      ),
    ('mkl_sbsrmv',     'mkl_dbsrmv',     'mkl_cbsrmv',     'mkl_zbsrmv'      ),
    ('smerge',         'dmerge',         'cmerge',         'zmerge'          ),
    ('sbcsr',          'dbcsr',          'cbcsr',          'zbcsr'           ),

    # ----- SPARSE Iterative Solvers
    ('scg',            'dcg',            'ccg',            'zcg'             ),
    ('sgmres',         'dgmres',         'cgmres',         'zgmres'          ),
    ('sbicgstab',      'dbicgstab',      'cbicgstab',      'zbicgstab',      ),

    ('spcg',           'dpcg',           'cpcg',           'zpcg',           ),
    ('spbicgstab',     'dpbicgstab',     'cpbicgstab',     'zpbicgstab',     ),
    ('spgmres',        'dpgmres',        'cpgmres',        'zpgmres'         ),
    ('sp1gmres',       'dp1gmres',       'cp1gmres',       'zp1gmres'        ),
    ('sjacobi',        'djacobi',        'cjacobi',        'zjacobi'         ),
    ('siterref',       'diterref',       'citerref',       'ziterref'        ),
    ('silu',           'dilu',           'cilu',           'zilu'            ),

    # ----- SPARSE auxiliary tools
    ('matrix_s',       'matrix_d',       'matrix_c',       'matrix_z'        ),
    ('svjacobi',       'dvjacobi',       'cvjacobi',       'zvjacobi'        ),
    ('s_csr2array',    'd_csr2array',    'c_csr2array',    'z_csr2array'     ),
    ('s_array2csr',    'd_array2csr',    'c_array2csr',    'z_array2csr'     ),
    ('read_s_csr',     'read_d_csr',     'read_c_csr',     'read_z_csr'      ),
    ('print_s_csr',    'print_d_csr',    'print_c_csr',    'print_z_csr'     ),
    ('write_s_csr',    'write_d_csr',    'write_c_csr',    'write_z_csr'     ),
    ('s_transpose',    'd_transpose',    'c_transpose',    'z_transpose'     ),
    ('SPARSE_S_H',     'SPARSE_D_H',     'SPARSE_C_H',     'SPARSE_Z_H'      ),
    ('_TYPES_S_H',     '_TYPES_D_H',     '_TYPES_C_H',     '_TYPES_Z_H'      ),
    ('sresidual',      'dresidual',      'cresidual',      'zresidual'       ), 


    # ----- Xeon Phi (MIC) specific, alphabetic order unless otherwise required
    ('SREG_WIDTH',                  'DREG_WIDTH',                  'CREG_WIDTH',                  'ZREG_WIDTH' ),
    ('_MM512_I32LOEXTSCATTER_PPS',  '_MM512_I32LOEXTSCATTER_PPD',  '_MM512_I32LOEXTSCATTER_PPC',  '_MM512_I32LOEXTSCATTER_PPZ' ),
    ('_MM512_LOAD_PPS',             '_MM512_LOAD_PPD',             '_MM512_LOAD_PPC',             '_MM512_LOAD_PPZ' ),
    ('_MM512_STORE_PPS',            '_MM512_STORE_PPD',            '_MM512_STORE_PPC',            '_MM512_STORE_PPZ' ),
    ('_MM_DOWNCONV_PS_NONE',        '_MM_DOWNCONV_PD_NONE',        '_MM_DOWNCONV_PC_NONE',        '_MM_DOWNCONV_PZ_NONE' ),
    ('__M512S',                     '__M512D',                     '__M512C',                     '__M512Z' ),
    ('somatcopy',                   'domatcopy',                   'comatcopy',                   'zomatcopy'),

    # ----- Prefixes
    # Most routines have already been renamed by above BLAS/LAPACK rules.
    # cublas[SDCZ] functions where real == complex name are handled here;
    # if real != complex name, it must be handled above.
    ('blasf77_s',      'blasf77_d',      'blasf77_c',      'blasf77_z'       ),
    ('blasf77_s',      'blasf77_d',      'blasf77_s',      'blasf77_d'       ),
    ('BLAS_S',         'BLAS_D',         'BLAS_C',         'BLAS_Z'          ),
    ('BLAS_s',         'BLAS_d',         'BLAS_c',         'BLAS_z'          ),
    ('BLAS_s',         'BLAS_d',         'BLAS_s',         'BLAS_d'          ),
    ('blas_is',        'blas_id',        'blas_ic',        'blas_iz'         ),
    ('blas_s',         'blas_d',         'blas_c',         'blas_z'          ),
    ('cl_ps',          'cl_pd',          'cl_pc',          'cl_pz'           ),
    ('cl_s',           'cl_d',           'cl_c',           'cl_z'            ),
    ('CODELETS_S',     'CODELETS_D',     'CODELETS_C',     'CODELETS_Z'      ),
    ('codelet_s',      'codelet_d',      'codelet_c',      'codelet_z'       ),
    ('compute_s',      'compute_d',      'compute_c',      'compute_z'       ),
    ('control_s',      'control_d',      'control_c',      'control_z'       ),
    ('coreblas_s',     'coreblas_d',     'coreblas_c',     'coreblas_z'      ),
    ('CORE_S',         'CORE_D',         'CORE_C',         'CORE_Z'          ),
    ('CORE_s',         'CORE_d',         'CORE_c',         'CORE_z'          ),
    ('core_s',         'core_d',         'core_c',         'core_z'          ),
    ('CORE_s',         'CORE_d',         'CORE_s',         'CORE_d'          ),
    ('cpu_gpu_s',      'cpu_gpu_d',      'cpu_gpu_c',      'cpu_gpu_z'       ),
    ('cublasIs',       'cublasId',       'cublasIs',       'cublasId'        ),
    ('cublasIs',       'cublasId',       'cublasIc',       'cublasIz'        ),
    ('cublasS',        'cublasD',        'cublasC',        'cublasZ'         ),
    ('clAmdBlasS',     'clAmdBlasD',     'clAmdBlasC',     'clAmdBlasZ'      ),
    ('example_s',      'example_d',      'example_c',      'example_z'       ),
    ('ipt_s',          'ipt_d',          'ipt_c',          'ipt_z'           ),
    ('LAPACKE_s',      'LAPACKE_d',      'LAPACKE_c',      'LAPACKE_z'       ),
    ('lapackf77_s',    'lapackf77_d',    'lapackf77_c',    'lapackf77_z'     ),
    ('lapackf77_s',    'lapackf77_d',    'lapackf77_s',    'lapackf77_d'     ),
    ('lapack_s',       'lapack_d',       'lapack_c',       'lapack_z'        ),
    ('lapack_s',       'lapack_d',       'lapack_s',       'lapack_d'        ),
    ('MAGMABLAS_S',    'MAGMABLAS_D',    'MAGMABLAS_C',    'MAGMABLAS_Z'     ),
    ('magmablas_s',    'magmablas_d',    'magmablas_c',    'magmablas_z'     ),
    ('magmaf_s',       'magmaf_d',       'magmaf_c',       'magmaf_z'        ),
    ('magma_get_s',    'magma_get_d',    'magma_get_c',    'magma_get_z'     ),
    ('magma_ps',       'magma_pd',       'magma_pc',       'magma_pz'        ),
    ('MAGMA_S',        'MAGMA_D',        'MAGMA_C',        'MAGMA_Z'         ),
    ('MAGMA_s',        'MAGMA_d',        'MAGMA_c',        'MAGMA_z'         ),
    ('magma_s',        'magma_d',        'magma_c',        'magma_z'         ),
    ('magmasparse_s',  'magmasparse_d',  'magmasparse_c',  'magmasparse_z'   ),
    ('morse_ps',       'morse_pd',       'morse_pc',       'morse_pz'        ),
    ('MORSE_S',        'MORSE_D',        'MORSE_C',        'MORSE_Z'         ),
    ('morse_s',        'morse_d',        'morse_c',        'morse_z'         ),
    ('plasma_ps',      'plasma_pd',      'plasma_pc',      'plasma_pz'       ),
    ('PLASMA_S',       'PLASMA_D',       'PLASMA_C',       'PLASMA_Z'        ),
    ('PLASMA_sor',     'PLASMA_dor',     'PLASMA_cun',     'PLASMA_zun'      ),
    ('PLASMA_s',       'PLASMA_d',       'PLASMA_c',       'PLASMA_z'        ),
    ('plasma_s',       'plasma_d',       'plasma_c',       'plasma_z'        ),
    ('PROFILE_S',      'PROFILE_D',      'PROFILE_C',      'PROFILE_Z'       ),
    ('profile_s',      'profile_d',      'profile_c',      'profile_z'       ),
    ('SCHED_s',        'SCHED_d',        'SCHED_c',        'SCHED_z'         ),
    ('starpu_s',       'starpu_d',       'starpu_c',       'starpu_z'        ),
    ('testing_ds',     'testing_ds',     'testing_zc',     'testing_zc'      ),
    ('testing_s',      'testing_d',      'testing_c',      'testing_z'       ),
    ('time_s',         'time_d',         'time_c',         'time_z'          ),
    ('WRAPPER_S',      'WRAPPER_D',      'WRAPPER_C',      'WRAPPER_Z'       ),
    ('wrapper_s',      'wrapper_d',      'wrapper_c',      'wrapper_z'       ),
    ('Workspace_s',    'Workspace_d',    'Workspace_c',    'Workspace_z'     ),
    ('workspace_s',    'workspace_d',    'workspace_c',    'workspace_z'     ),
    ('QUARK_Insert_Task_s', 'QUARK_Insert_Task_d', 'QUARK_Insert_Task_c', 'QUARK_Insert_Task_z' ),
    
    # magma_[get_]d -> magma_[get_]s, so revert _sevice to _device
    ('_device',        '_sevice',        '_device',        '_sevice'         ),
  ],
  
  # ------------------------------------------------------------
  # replacements applied for profiling with tau
  'tracing' :[
    # ----- Special line indicating column types
    ['plain', 'tau'],
    
    # ----- Replacements
    ('(\w+\*?)\s+(\w+)\s*\(([a-z* ,A-Z_0-9]*)\)\s*{\s+(.*)\s*#pragma tracing_start\s+(.*)\s+#pragma tracing_end\s+(.*)\s+}',
      r'\1 \2(\3){\n\4tau("\2");\5tau();\6}'),
    ('\.c','.c.tau'),
  ],
};
