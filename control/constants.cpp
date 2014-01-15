#include <assert.h>
#include <stdio.h>

#include "magma_types.h"

// ----------------------------------------
// Convert LAPACK character constants to MAGMA constants.
// This is a one-to-many mapping, requiring multiple translators
// (e.g., "N" can be NoTrans or NonUnit or NoVec).
// These functions and cases are in the same order as the constants are
// declared in magma_types.h

extern "C"
magma_order_t  magma_order_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'R': case 'r': return MagmaRowMajor;
        case 'C': case 'c': return MagmaColMajor;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaRowMajor;
    }
}

extern "C"
magma_trans_t  magma_trans_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return MagmaNoTrans;
        case 'T': case 't': return MagmaTrans;
        case 'C': case 'c': return MagmaConjTrans;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaNoTrans;
    }
}

extern "C"
magma_uplo_t   magma_uplo_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'U': case 'u': return MagmaUpper;
        case 'L': case 'l': return MagmaLower;
        case 'H': case 'h': return MagmaHessenberg;  // see lascl
        default:            return MagmaFull;        // see laset
    }
}

extern "C"
magma_diag_t   magma_diag_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return MagmaNonUnit;
        case 'U': case 'u': return MagmaUnit;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaNonUnit;
    }
}

extern "C"
magma_side_t   magma_side_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'L': case 'l': return MagmaLeft;
        case 'R': case 'r': return MagmaRight;
        case 'B': case 'b': return MagmaBothSides;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaLeft;
    }
}

extern "C"
magma_norm_t   magma_norm_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'O': case 'o': case '1': return MagmaOneNorm;
        case '2':           return MagmaTwoNorm;
        case 'F': case 'f': case 'E': case 'e': return MagmaFrobeniusNorm;
        case 'I': case 'i': return MagmaInfNorm;
        case 'M': case 'm': return MagmaMaxNorm;
        // MagmaRealOneNorm
        // MagmaRealInfNorm
        // MagmaRealMaxNorm
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaOneNorm;
    }
}

extern "C"
magma_dist_t   magma_dist_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'U': case 'u': return MagmaDistUniform;
        case 'S': case 's': return MagmaDistSymmetric;
        case 'N': case 'n': return MagmaDistNormal;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaDistUniform;
    }
}

extern "C"
magma_sym_t    magma_sym_const   ( char lapack_char )
{
    switch( lapack_char ) {
        case 'H': case 'h': return MagmaHermGeev;
        case 'P': case 'p': return MagmaHermPoev;
        case 'N': case 'n': return MagmaNonsymPosv;
        case 'S': case 's': return MagmaSymPosv;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaHermGeev;
    }
}

extern "C"
magma_pack_t   magma_pack_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return MagmaNoPacking;
        case 'U': case 'u': return MagmaPackSubdiag;
        case 'L': case 'l': return MagmaPackSupdiag;
        case 'C': case 'c': return MagmaPackColumn;
        case 'R': case 'r': return MagmaPackRow;
        case 'B': case 'b': return MagmaPackLowerBand;
        case 'Q': case 'q': return MagmaPackUpeprBand;
        case 'Z': case 'z': return MagmaPackAll;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaNoPacking;
    }
}

extern "C"
magma_vec_t    magma_vec_const   ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return MagmaNoVec;
        case 'V': case 'v': return MagmaVec;
        case 'I': case 'i': return MagmaIVec;
        case 'A': case 'a': return MagmaAllVec;
        case 'S': case 's': return MagmaSomeVec;
        case 'O': case 'o': return MagmaOverwriteVec;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaNoVec;
    }
}

extern "C"
magma_range_t  magma_range_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'A': case 'a': return MagmaRangeAll;
        case 'V': case 'v': return MagmaRangeV;
        case 'I': case 'i': return MagmaRangeI;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaRangeAll;
    }
}

extern "C"
magma_direct_t magma_direct_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'F': case 'f': return MagmaForward;
        case 'B': case 'b': return MagmaBackward;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaForward;
    }
}

extern "C"
magma_storev_t magma_storev_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'C': case 'c': return MagmaColumnwise;
        case 'R': case 'r': return MagmaRowwise;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaColumnwise;
    }
}


// ----------------------------------------
// Convert MAGMA constants to LAPACK constants.

const char *magma2lapack_constants[] =
{
    "",                                      //  0
    "", "", "", "", "", "", "", "", "", "",  //  1-10
    "", "", "", "", "", "", "", "", "", "",  // 11-20
    "", "", "", "", "", "", "", "", "", "",  // 21-30
    "", "", "", "", "", "", "", "", "", "",  // 31-40
    "", "", "", "", "", "", "",              // 41-47
    
    "0",                                     // 48: 0
    "1",                                     // 49: 1
    "2",                                     // 50: 2
    "3",                                     // 51: 3
    "4",                                     // 52: 4
    "5",                                     // 53: 5
    "6",                                     // 54: 6
    "7",                                     // 55: 7
    "8",                                     // 56: 8
    "9",                                     // 57: 9
    
    "", "", "", "", "", "", "",              // 58-64
    
    "A",                                     // 65: A
    "B",                                     // 66: B
    "C",                                     // 67: C
    "D",                                     // 68: D
    "E",                                     // 69: E
    "F",                                     // 70: F
    "G",                                     // 71: G
    "H",                                     // 72: H
    "I",                                     // 73: I
    "J",                                     // 74: J
    "K",                                     // 75: K
    "L",                                     // 76: L
    "M",                                     // 77: M
    "N",                                     // 78: N
    "O",                                     // 79: O
    "P",                                     // 80: P
    "Q",                                     // 81: Q
    "R",                                     // 82: R
    "S",                                     // 83: S
    "T",                                     // 84: T
    "U",                                     // 85: U
    "V",                                     // 86: V
    "W",                                     // 87: W
    "X",                                     // 88: X
    "Y",                                     // 89: Y
    "Z",                                     // 90: Z
    
    "", "", "", "", "", "",                  // 91-96
    
    "A",                                     //  97: a
    "B",                                     //  98: b
    "C",                                     //  99: c
    "D",                                     // 100: d
    "E",                                     // 101: e
    "F",                                     // 102: f
    "G",                                     // 103: g
    "H",                                     // 104: h
    "I",                                     // 105: i
    "J",                                     // 106: j
    "K",                                     // 107: k
    "L",                                     // 108: l
    "M",                                     // 109: m
    "N",                                     // 110: n
    "O",                                     // 111: o
    "P",                                     // 112: p
    "Q",                                     // 113: q
    "R",                                     // 114: r
    "S",                                     // 115: s
    "T",                                     // 116: t
    "U",                                     // 117: u
    "V",                                     // 118: v
    "W",                                     // 119: w
    "X",                                     // 120: x
    "Y",                                     // 121: y
    "Z",                                     // 122: z
    
    "", "", "", "", "", "", "", "",          // 123-130
    
    // -----------------------------------------------------------------
    // Above here are old char constants, e.g., MagmaNoTrans is 'n' or 'N' => "N";
    // those will get replaced soon.
    // Below here are new enum constants, e.g., MagmaDistUniform is 201    => "Uniform"
    // -----------------------------------------------------------------
    
    "Non-unit",                              // 131: MagmaNonUnit
    "Unit",                                  // 132: MagmaUnit
    "", "", "", "", "", "", "", "",          // 133-140
    "Left",                                  // 141: MagmaLeft
    "Right",                                 // 142: MagmaRight
    "Both",                                  // 143: MagmaBothSides (dtrevc)
    "", "", "", "", "", "", "",              // 144-150
    "", "", "", "", "", "", "", "", "", "",  // 151-160
    "", "", "", "", "", "", "", "", "", "",  // 161-170
    "One norm",                              // 171: MagmaOneNorm
    "",                                      // 172: MagmaRealOneNorm
    "",                                      // 173: MagmaTwoNorm
    "Frobenius norm",                        // 174: MagmaFrobeniusNorm
    "Infinity norm",                         // 175: MagmaInfNorm
    "",                                      // 176: MagmaRealInfNorm
    "Maximum norm",                          // 177: MagmaMaxNorm
    "",                                      // 178: MagmaRealMaxNorm
    "", "",                                  // 179-180
    "", "", "", "", "", "", "", "", "", "",  // 181-190
    "", "", "", "", "", "", "", "", "", "",  // 191-200
    "Uniform",                               // 201: MagmaDistUniform
    "Symmetric",                             // 202: MagmaDistSymmetric
    "Normal",                                // 203: MagmaDistNormal
    "", "", "", "", "", "", "",              // 204-210
    "", "", "", "", "", "", "", "", "", "",  // 211-220
    "", "", "", "", "", "", "", "", "", "",  // 221-230
    "", "", "", "", "", "", "", "", "", "",  // 231-240
    "Hermitian",                             // 241 MagmaHermGeev
    "Positive ev Hermitian",                 // 242 MagmaHermPoev
    "NonSymmetric pos sv",                   // 243 MagmaNonsymPosv
    "Symmetric pos sv",                      // 244 MagmaSymPosv
    "", "", "", "", "", "",                  // 245-250
    "", "", "", "", "", "", "", "", "", "",  // 251-260
    "", "", "", "", "", "", "", "", "", "",  // 261-270
    "", "", "", "", "", "", "", "", "", "",  // 271-280
    "", "", "", "", "", "", "", "", "", "",  // 281-290
    "No Packing",                            // 291 MagmaNoPacking
    "U zero out subdiag",                    // 292 MagmaPackSubdiag
    "L zero out superdiag",                  // 293 MagmaPackSupdiag
    "C",                                     // 294 MagmaPackColumn
    "R",                                     // 295 MagmaPackRow
    "B",                                     // 296 MagmaPackLowerBand
    "Q",                                     // 297 MagmaPackUpeprBand
    "Z",                                     // 298 MagmaPackAll
    "", "",                                  // 299-300
    "No vectors",                            // 301 MagmaNoVec
    "Vectors needed",                        // 302 MagmaVec
    "I",                                     // 303 MagmaIVec
    "All",                                   // 304 MagmaAllVec
    "Some",                                  // 305 MagmaSomeVec
    "Overwrite",                             // 306 MagmaOverwriteVec
    "", "", "", "",                          // 307-310
    "", "", "", "", "", "", "", "", "", "",  // 311-320
    "", "", "", "", "", "", "", "", "", "",  // 321-330
    "", "", "", "", "", "", "", "", "", "",  // 331-340
    "", "", "", "", "", "", "", "", "", "",  // 341-350
    "", "", "", "", "", "", "", "", "", "",  // 351-360
    "", "", "", "", "", "", "", "", "", "",  // 361-370
    "", "", "", "", "", "", "", "", "", "",  // 371-380
    "", "", "", "", "", "", "", "", "", "",  // 381-390
    "Forward",                               // 391: MagmaForward
    "Backward",                              // 392: MagmaBackward
    "", "", "", "", "", "", "", "",          // 393-400
    "Columnwise",                            // 401: MagmaColumnwise
    "Rowwise",                               // 402: MagmaRowwise
    "", "", "", "", "", "", "", ""           // 403-410
    // Remember to add a comma!
};

const char* lapack_const( int magma_const )
{
    assert( magma_const >= MagmaMinConst );
    assert( magma_const <= MagmaMaxConst );
    return magma2lapack_constants[ magma_const ];
}

char lapacke_const( int magma_const )
{
    return lapack_const( magma_const )[0];
}


// ----------------------------------------
// Convert MAGMA constants to CBLAS constants.
#ifdef HAVE_CBLAS

extern "C"
enum CBLAS_ORDER     cblas_order_const  ( magma_order_t magma_const )
{
    switch( magma_const ) {
        case 'r': case 'R': return CblasRowMajor;
        case 'c': case 'C': return CblasColMajor;
        default:
            fprintf( stderr, "Error in %s: unexpected value %d\n", __func__, magma_const );
            return CblasColMajor;
    }
}

extern "C"
enum CBLAS_TRANSPOSE cblas_trans_const  ( magma_trans_t magma_const )
{
    switch( magma_const ) {
        case 'n': case 'N': return CblasNoTrans;
        case 't': case 'T': return CblasTrans;
        case 'c': case 'C': return CblasConjTrans;
        default:
            fprintf( stderr, "Error in %s: unexpected value %d\n", __func__, magma_const );
            return CblasNoTrans;
    }
}

extern "C"
enum CBLAS_SIDE      cblas_side_const   ( magma_side_t magma_const )
{
    switch( magma_const ) {
        case 'l': case 'L': return CblasLeft;
        case 'r': case 'R': return CblasRight;
        default:
            fprintf( stderr, "Error in %s: unexpected value %d\n", __func__, magma_const );
            return CblasLeft;
    }
}

extern "C"
enum CBLAS_DIAG      cblas_diag_const   ( magma_diag_t magma_const )
{
    switch( magma_const ) {
        case 'n': case 'N': return CblasNonUnit;
        case 'u': case 'U': return CblasUnit;
        default:
            fprintf( stderr, "Error in %s: unexpected value %d\n", __func__, magma_const );
            return CblasNonUnit;
    }
}

extern "C"
enum CBLAS_UPLO      cblas_uplo_const   ( magma_uplo_t magma_const )
{
    switch( magma_const ) {
        case 'u': case 'U': return CblasUpper;
        case 'l': case 'L': return CblasLower;
        default:
            fprintf( stderr, "Error in %s: unexpected value %d\n", __func__, magma_const );
            return CblasUpper;
    }
}

#endif  // HAVE_CBLAS
