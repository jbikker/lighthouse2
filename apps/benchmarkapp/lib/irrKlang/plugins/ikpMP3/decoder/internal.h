/* Based on common.h from libavcodec.  Modified extensively by Matt Campbell
   <mattcampbell@pobox.com> for the stand-alone mpaudec library. */

#ifndef INTERNAL_H
#define INTERNAL_H

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
#    define CONFIG_WIN32
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stddef.h>
#include "mpaudec.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#ifdef CONFIG_WIN32

/* windows */

typedef unsigned short uint16_t;
typedef signed short int16_t;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned __int64 uint64_t;
typedef signed char int8_t;
typedef signed int int32_t;
typedef signed __int64 int64_t;

#    ifdef _DEBUG
#        define DEBUG
#    endif

/* CONFIG_WIN32 end */
#else

/* unix */

#include <inttypes.h>

#endif /* !CONFIG_WIN32 */

/* debug stuff */

#if !defined(DEBUG) && !defined(NDEBUG)
#    define NDEBUG
#endif
#include <assert.h>

/* bit input */

typedef struct GetBitContext {
    const uint8_t *buffer;
    int index;
    int size_in_bits;
} GetBitContext;

int get_bits_count(const GetBitContext *s);

#define VLC_TYPE int16_t

typedef struct VLC {
    int bits;
    VLC_TYPE (*table)[2];
    int table_size, table_allocated;
} VLC;

unsigned int get_bits(GetBitContext *s, int n);
unsigned int show_bits(const GetBitContext *s, int n);
void skip_bits(GetBitContext *s, int n);
void init_get_bits(GetBitContext *s,
                   const uint8_t *buffer, int buffer_size);

int init_vlc(VLC *vlc, int nb_bits, int nb_codes,
             const void *bits, int bits_wrap, int bits_size,
             const void *codes, int codes_wrap, int codes_size);
void free_vlc(VLC *vlc);
int get_vlc(GetBitContext *s, const VLC *vlc);

#endif /* INTERNAL_H */
