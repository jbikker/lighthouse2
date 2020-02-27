/* Portions based on avcodec.h from libavcodec. */

#ifndef MPAUDEC_H
#define MPAUDEC_H

#ifdef __cplusplus
extern "C" {
#endif

/* in bytes */
#define MPAUDEC_MAX_AUDIO_FRAME_SIZE 4608

typedef struct MPAuDecContext {
    int bit_rate;
    int layer;
    int sample_rate;
    int channels;
    int frame_size;
    void *priv_data;
    int parse_only;
    int coded_frame_size;
} MPAuDecContext;

int mpaudec_init(MPAuDecContext *mpctx);
int mpaudec_decode_frame(MPAuDecContext * mpctx,
                         void *data, int *data_size,
                         const unsigned char * buf, int buf_size);
void mpaudec_clear(MPAuDecContext *mpctx);

#ifdef __cplusplus
}
#endif

#endif /* MPAUDEC_H */
