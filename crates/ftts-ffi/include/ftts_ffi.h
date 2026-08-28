/* franken_tts C ABI — the native engine behind the iOS app.
 *
 * Contract (mirrored in crates/ftts-ffi/src/lib.rs, which is the source of truth):
 *   - int-returning functions use 0 for success; on failure,
 *     ftts_last_error_message() describes the problem (thread-local, valid until the
 *     next failing call on the same thread; never NULL, empty before any failure).
 *   - FttsEngine is NOT thread-safe: the caller serializes all access to one handle.
 *   - Strings are NUL-terminated UTF-8.
 *   - PCM from ftts_synthesize is mono f32 at 24 kHz, owned by the caller, released
 *     with ftts_pcm_free using the exact returned length.
 */
#ifndef FTTS_FFI_H
#define FTTS_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Floats in a speaker x-vector (every preset and enroll output). */
#define FTTS_SPEAKER_WIDTH 1024

typedef struct FttsEngine FttsEngine;

/* Last failure on this thread, as UTF-8. Never NULL. */
const char *ftts_last_error_message(void);

/* Static JSON array of {name, character} for the built-in voices. */
const char *ftts_presets_json(void);

/* Copies the named preset's x-vector into out[FTTS_SPEAKER_WIDTH]. 0 on success. */
int32_t ftts_preset_vector(const char *name, float *out);

/* Opens the engine over a complete model directory. NULL on failure. */
FttsEngine *ftts_engine_open(const char *model_dir);

/* Releases an engine. NULL is a no-op. */
void ftts_engine_close(FttsEngine *engine);

/* Synthesizes text with the given speaker vector (must be FTTS_SPEAKER_WIDTH floats).
 * On success writes a caller-owned buffer to *out_pcm / *out_len. */
int32_t ftts_synthesize(FttsEngine *engine, const char *text, const float *speaker,
                        size_t speaker_len, uint64_t seed, float **out_pcm,
                        size_t *out_len);

/* JSON attribution for the most recent successful synthesis. The pointer remains valid
 * until the next successful synthesis or engine close. Durations are milliseconds;
 * codec_active_ms overlaps generation_ms because codec decode runs concurrently. */
const char *ftts_last_synthesis_profile_json(const FttsEngine *engine);

/* Releases a buffer from ftts_synthesize. len must be the returned length. */
void ftts_pcm_free(float *pcm, size_t len);

/* Streaming synthesis: `on_packet` receives each decoded packet the moment it exists,
 * ON THE ENGINE'S DECODE THREAD — return promptly, hand samples to your audio queue,
 * and do not call back into this engine from inside it. `samples` are mono 24 kHz f32
 * in [-1, 1] (f32 end to end: AVAudioEngine plays Float32 natively), valid only for
 * the duration of the call — copy them out. `frame_index` counts 80 ms frames already
 * delivered before this packet. `packet_frames` picks the cadence (1 = lowest first-
 * audio latency, 4 = the whole-buffer call's historical cadence). Returning nonzero
 * from `on_packet` requests cancellation: delivery stops within one packet and the
 * call returns FTTS_SYNTH_CANCELLED. The callback must not throw/unwind (C contract).
 * Returns 0 on success, FTTS_SYNTH_CANCELLED (6) when the callback cancelled,
 * any other nonzero on failure (see ftts_last_error_message). The engine handle
 * stays externally serialized, exactly as for ftts_synthesize. */
#define FTTS_SYNTH_CANCELLED 6
typedef int32_t (*FttsPacketFn)(void *ctx, const float *samples, size_t len,
                                uint64_t frame_index);
int32_t ftts_synthesize_streaming(FttsEngine *engine, const char *text,
                                  const float *speaker, size_t speaker_len,
                                  uint64_t seed, size_t packet_frames,
                                  FttsPacketFn on_packet, void *ctx);

/* Enrolls a voice from mono 24 kHz f32 PCM into out[FTTS_SPEAKER_WIDTH]. 0 on success. */
int32_t ftts_enroll(FttsEngine *engine, const float *pcm, size_t len, float *out);

/* ---- branded share video: identical frames to `ftts make-video` -------------------- */

/* 1 when the neural denoiser artifact is in the engine's model directory. */
int32_t ftts_denoise_available(const FttsEngine *engine);

/* Denoises mono 24 kHz f32 PCM. Writes an owned buffer of the SAME length to out_pcm
 * (release with ftts_pcm_free). 0 on success; nonzero when the denoiser is absent or
 * fails — the caller keeps its original audio. */
int32_t ftts_denoise(const FttsEngine *engine, const float *pcm, size_t len,
                     float **out_pcm);

typedef struct FttsVideoRenderer FttsVideoRenderer;

uint32_t ftts_video_width(void);
uint32_t ftts_video_height(void);
uint32_t ftts_video_fps(void);

/* Opens a renderer over finished speech PCM. NULL on failure. */
FttsVideoRenderer *ftts_video_open(const float *pcm, size_t len, uint32_t sample_rate,
                                   const char *voice_label);
size_t ftts_video_frame_count(const FttsVideoRenderer *renderer);
/* Renders RGB24 into out (width*height*3 bytes). 0 on success. Frames are pure
 * functions of immutable renderer state, so concurrent calls over ONE renderer are
 * allowed (open/close still serialized against renders). */
int32_t ftts_video_render_frame(const FttsVideoRenderer *renderer, size_t frame,
                                uint8_t *out);
/* As above, but BGRA32 with a caller-chosen row stride in bytes (CoreVideo's
 * layout). out must hold stride * height bytes; stride >= width * 4. */
int32_t ftts_video_render_frame_bgra(const FttsVideoRenderer *renderer, size_t frame,
                                     uint8_t *out, size_t stride);
void ftts_video_close(FttsVideoRenderer *renderer);

#ifdef __cplusplus
}
#endif

#endif /* FTTS_FFI_H */
