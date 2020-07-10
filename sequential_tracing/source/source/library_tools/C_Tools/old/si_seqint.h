/******************************************************************************
** seqint.h
**
** C functions (prefixed with si_) and respective Python C API
** functions for converting between DNA sequences (containing the bases A, T,
** G, and C) and lightweight integer representations for compact storage
** or indexing schemes (e.g., bitarrays). Binary mapping is as follows:
**
**                   Base   Binary   Int   ASCII char int
**                   A        00      0     65 (A) 97 (a)
**                   T        11      3     84 (T) 116 (t)
**                   G        10      1     71 (G) 102 (g)
**                   C        01      2     67 (C) 99 (c)
**
** For example, converting the DNA sequence ATGGC to an integer could be
** thought of in terms of these conversions:
**
**  Sequence:                       ATGGC
**  Binary representation:          0011101001
**  Integer:                        233
**
******************************************************************************/

#ifndef SEQ_INT_H
#define SEQ_INT_H

#include <stdlib.h>

/* Convert the DNA sequence at `seq` of `length` bp to an integer
 *
 * For example:
 *      Initial sequence:       ATGGC
 *      Convert to binary:      00 11 10 10 01
 *                              A  T  G  G  C
 *      Convert binary to int:  0011101001 -> 233
 */
uint64_t si_seq2Int(char* seq, int length);

/* Convert the DNA sequence represented by seqint with given length to a
 * C string. The length of the sequence is necessary to insure that
 * leading zero bits are not interpreted as A bases. Continuing the
 * above example:
 *
 * 64-bit representation of ATGGC:
 *      000000000000000000000000000000000000000000000000000011101001
 * Length-agnostic sequence interpretation:
 *      AAAAAAAAAAAAAAAAAAAAAAAAAATGGC
 * Length-aware sequence interpretation (length=5)
 *      ATGGC
 */
int si_int2Seq(uint64_t seqint, char* out_str, int length);

/* Calculate the integer representation of the reverse of `seqint`
 * of length `length`. This reverses the sequence represented by
 * the provided `seqint` without proceeding through a string
 * intermediate. Again, length is required to define the scope
 * of the bit representation that is reversed.
 */
uint64_t si_revSeqInt(uint64_t seqint, int length);

/* Calculate the integer representation of the complement of `seqint`
 * of length `length`. This reverses the sequence represented by
 * the provided `seqint` without proceeding through a string
 * intermediate. Again, length is required to define the scope
 * of the bit representation that is reversed.
 */
uint64_t si_compSeqInt(uint64_t seqint, int length);

/* Calculate the integer representation of the reverse complement of `seqint`
 * of length `length`. This reverses the sequence represented by
 * the provided `seqint` without proceeding through a string
 * intermediate. Again, length is required to define the scope
 * of the bit representation that is reversed.
 */
uint64_t si_revCompSeqInt(uint64_t seqint, int length);

/* Add a single base to the right-hand side of the `seqint`.
 * si_addBase(233, 'A') is the equivalent of 'ATGGC' + 'A' in
 * sequence space.
 */
uint64_t si_addBase(uint64_t seqint, char base);

/* Remove a single base from the right-hand side of the `seqint`.
 * si_removeBase(233) is the equivalent of 'ATGGC'[:-1] in
 * sequence space.
 */
uint64_t si_removeBase(uint64_t seqint);

/* Return the seqint representation of the substring w/ provided indices
 * si_seqIntSubstring(233, 0, 3, 6) is the equivalent of 'ATGGC'[0:3] in
 * sequence space.
 */
uint64_t si_seqIntSubstring(uint64_t seqint, int start_idx, int end_idx,
                          int length);

/* Add a single base to the right-hand side of the seq-int of given length
 * and mask to the original length
 * si_addToWindow(233, 'A', 6) is the equivalent of 'ATGGC'[1:] + 'A' in
 * sequence space.
 */
uint64_t si_addToWindow(uint64_t seqint, char base, int length);

#define SI_MASK3   0x000000000000003F
#define SI_MASK6   0x0000000000000FFF
#define SI_A_X_6   0x0000000000000000 // AAAAAA
#define SI_T_X_6   0x0000000000000FFF // TTTTTT
#define SI_C_X_3   0x0000000000000015 // CCC 00010101
#define SI_G_X_3   0x000000000000002A // GGG

#define SI_CG_X_3  0x0000000000000666 // CGCGCG
#define SI_GC_X_3  0x0000000000000999 // GCGCGC

#endif