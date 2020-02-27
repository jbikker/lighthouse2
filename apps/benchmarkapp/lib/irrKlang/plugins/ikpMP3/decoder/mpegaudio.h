/* Modified slightly by Matt Campbell <mattcampbell@pobox.com> for the
   stand-alone mpaudec library.  Based on mpegaudio.h from libavcodec. */

/* max frame size, in samples */
#define MPA_FRAME_SIZE 1152 

/* max compressed frame size */
#define MPA_MAX_CODED_FRAME_SIZE 1792

#define MPA_MAX_CHANNELS 2

#define SBLIMIT 32 /* number of subbands */

#define MPA_STEREO  0
#define MPA_JSTEREO 1
#define MPA_DUAL    2
#define MPA_MONO    3
