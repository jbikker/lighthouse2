/*
 * Common bit i/o utils
 * Copyright (c) 2000, 2001 Fabrice Bellard.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Modified by Matt Campbell <mattcampbell@pobox.com> for the stand-alone
 * mpaudec library.  Based on common.c from libavcodec.
 */

#include "internal.h"

/**
 * init GetBitContext.
 * @param buffer bitstream buffer
 * @param bit_size the size of the buffer in bits
 */
void init_get_bits(GetBitContext *s,
                   const uint8_t *buffer, int bit_size)
{
    s->buffer= buffer;
    s->size_in_bits= bit_size;
    s->index=0;
}

unsigned int show_bits(const GetBitContext *s, int n)
{
    int i;
    unsigned int result = 0;
    assert(s->size_in_bits - s->index >= n);
    for (i = s->index; i < s->index + n; i++) {
        int byte_index = i / 8;
        unsigned int right_shift = 7 - (i % 8);
        uint8_t byte = s->buffer[byte_index];
        uint8_t bit;
        result <<= 1;
        if (right_shift == 0)
            bit = byte & 0x1;
        else
            bit = (byte >> right_shift) & 0x1;
        result |= (unsigned int)bit;
    }
    return result;
}

void skip_bits(GetBitContext *s, int n)
{
    s->index += n;
}

unsigned int get_bits(GetBitContext *s, int n)
{
    unsigned int result = show_bits(s, n);
    skip_bits(s, n);
    return result;
}

int get_bits_count(const GetBitContext *s)
{
    return s->index;
}

/* VLC decoding */

/*#define DEBUG_VLC*/

#define GET_DATA(v, table, i, wrap, size) \
{\
    const uint8_t *ptr = (const uint8_t *)table + i * wrap;\
    switch(size) {\
    case 1:\
        v = *(const uint8_t *)ptr;\
        break;\
    case 2:\
        v = *(const uint16_t *)ptr;\
        break;\
    default:\
        v = *(const uint32_t *)ptr;\
        break;\
    }\
}


static int alloc_table(VLC *vlc, int size)
{
    int index;
    index = vlc->table_size;
    vlc->table_size += size;
    if (vlc->table_size > vlc->table_allocated) {
        vlc->table_allocated += (1 << vlc->bits);
        vlc->table = realloc(vlc->table,
                             sizeof(VLC_TYPE) * 2 * vlc->table_allocated);
        if (!vlc->table)
            return -1;
    }
    return index;
}

static int build_table(VLC *vlc, int table_nb_bits,
                       int nb_codes,
                       const void *bits, int bits_wrap, int bits_size,
                       const void *codes, int codes_wrap, int codes_size,
                       uint32_t code_prefix, int n_prefix)
{
    int i, j, k, n, table_size, table_index, nb, n1, index;
    uint32_t code;
    VLC_TYPE (*table)[2];

    table_size = 1 << table_nb_bits;
    table_index = alloc_table(vlc, table_size);
#ifdef DEBUG_VLC
    printf("new table index=%d size=%d code_prefix=%x n=%d\n",
           table_index, table_size, code_prefix, n_prefix);
#endif
    if (table_index < 0)
        return -1;
    table = &vlc->table[table_index];

    for(i=0;i<table_size;i++) {
        table[i][1] = 0; /*bits*/
        table[i][0] = -1; /*codes*/
    }

    /* first pass: map codes and compute auxillary table sizes */
    for(i=0;i<nb_codes;i++) {
        GET_DATA(n, bits, i, bits_wrap, bits_size);
        GET_DATA(code, codes, i, codes_wrap, codes_size);
        /* we accept tables with holes */
        if (n <= 0)
            continue;
#if defined(DEBUG_VLC) && 0
        printf("i=%d n=%d code=0x%x\n", i, n, code);
#endif
        /* if code matches the prefix, it is in the table */
        n -= n_prefix;
        if (n > 0 && (code >> n) == code_prefix) {
            if (n <= table_nb_bits) {
                /* no need to add another table */
                j = (code << (table_nb_bits - n)) & (table_size - 1);
                nb = 1 << (table_nb_bits - n);
                for(k=0;k<nb;k++) {
#ifdef DEBUG_VLC
                    printf("%4x: code=%d n=%d\n",
                           j, i, n);
#endif
                    assert(table[j][1] /*bits*/ == 0);
                    table[j][1] = n; /*bits*/
                    table[j][0] = i; /*code*/
                    j++;
                }
            } else {
                n -= table_nb_bits;
                j = (code >> n) & ((1 << table_nb_bits) - 1);
#ifdef DEBUG_VLC
                printf("%4x: n=%d (subtable)\n",
                       j, n);
#endif
                /* compute table size */
                n1 = -table[j][1]; /*bits*/
                if (n > n1)
                    n1 = n;
                table[j][1] = -n1; /*bits*/
            }
        }
    }

    /* second pass : fill auxillary tables recursively */
    for(i=0;i<table_size;i++) {
        n = table[i][1]; /*bits*/
        if (n < 0) {
            n = -n;
            if (n > table_nb_bits) {
                n = table_nb_bits;
                table[i][1] = -n; /*bits*/
            }
            index = build_table(vlc, n, nb_codes,
                                bits, bits_wrap, bits_size,
                                codes, codes_wrap, codes_size,
                                (code_prefix << table_nb_bits) | i,
                                n_prefix + table_nb_bits);
            if (index < 0)
                return -1;
            /* note: realloc has been done, so reload tables */
            table = &vlc->table[table_index];
            table[i][0] = index; /*code*/
        }
    }
    return table_index;
}


/* Build VLC decoding tables suitable for use with get_vlc().

   'nb_bits' set thee decoding table size (2^nb_bits) entries. The
   bigger it is, the faster is the decoding. But it should not be too
   big to save memory and L1 cache. '9' is a good compromise.
   
   'nb_codes' : number of vlcs codes

   'bits' : table which gives the size (in bits) of each vlc code.

   'codes' : table which gives the bit pattern of of each vlc code.

   'xxx_wrap' : give the number of bytes between each entry of the
   'bits' or 'codes' tables.

   'xxx_size' : gives the number of bytes of each entry of the 'bits'
   or 'codes' tables.

   'wrap' and 'size' allows to use any memory configuration and types
   (byte/word/long) to store the 'bits' and 'codes' tables.  
*/
int init_vlc(VLC *vlc, int nb_bits, int nb_codes,
             const void *bits, int bits_wrap, int bits_size,
             const void *codes, int codes_wrap, int codes_size)
{
    vlc->bits = nb_bits;
    vlc->table = NULL;
    vlc->table_allocated = 0;
    vlc->table_size = 0;
#ifdef DEBUG_VLC
    printf("build table nb_codes=%d\n", nb_codes);
#endif

    if (build_table(vlc, nb_bits, nb_codes,
                    bits, bits_wrap, bits_size,
                    codes, codes_wrap, codes_size,
                    0, 0) < 0) {
        free(vlc->table);
        return -1;
    }
    return 0;
}


void free_vlc(VLC *vlc)
{
    free(vlc->table);
}

int get_vlc(GetBitContext *s, const VLC *vlc)
{
    int code = 0;
    int depth = 0, max_depth = 3;
    int n, index, bits = vlc->bits;
    
    do {
        index = show_bits(s, bits) + code;
        code = vlc->table[index][0];
        n = vlc->table[index][1];
        depth++;

        if (n < 0 && depth < max_depth) {
            skip_bits(s, bits);
            bits = -n;
        }
    } while (n < 0 && depth < max_depth);

    skip_bits(s, n);
    return code;
}
