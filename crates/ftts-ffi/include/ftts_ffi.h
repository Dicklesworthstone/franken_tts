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

/* Releases a buffer from ftts_synthesize. len must be the returned length. */
void ftts_pcm_free(float *pcm, size_t len);

/* Enrolls a voice from mono 24 kHz f32 PCM into out[FTTS_SPEAKER_WIDTH]. 0 on success. */
int32_t ftts_enroll(FttsEngine *engine, const float *pcm, size_t len, float *out);

#ifdef __cplusplus
}
#endif

#endif /* FTTS_FFI_H */
