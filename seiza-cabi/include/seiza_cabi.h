#ifndef SEIZA_CABI_H
#define SEIZA_CABI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Memory ownership and lifetimes
 * ==============================
 * Everything this library returns is allocated by Rust. Release it ONLY with
 * the matching seiza_*_free function below -- never with free(), delete,
 * CoTaskMemFree, Marshal.FreeHGlobal, or any other allocator. Mixing allocators
 * is undefined behavior. Every free function accepts NULL and no-ops, so it is
 * always safe to call.
 *
 * OWNED by the caller (you must release it):
 *   - SeizaRenderedImage* returned by the seiza_rendered_image_open* functions
 *       -> release with seiza_rendered_image_free().
 *   - char* returned by seiza_catalog_status_json and seiza_solve_image_json,
 *     and any char* stored into an `error_out` argument on failure
 *       -> release with seiza_string_free().
 *
 * BORROWED by the caller (do NOT free; valid only while the owner is alive):
 *   - const uint8_t* from seiza_rendered_image_rgba / _bgra and const char*
 *     from seiza_rendered_image_metadata_json point INTO the SeizaRenderedImage
 *     and dangle the moment it is freed. Copy them out before freeing the image
 *     if you need them to outlive it.
 *   - const char* from seiza_core_version is static and is never freed.
 *   - the `json` argument to a SeizaCatalogSetupProgressCallback is valid only
 *     for the duration of that one callback; copy it if you need it afterward.
 *
 * Convention: a non-const `char*` return transfers ownership (free it); a
 * `const` return is borrowed (do not free).
 *
 * Errors: functions taking `char** error_out` set *error_out to NULL on success
 * and to an OWNED message on failure (and return NULL / false). Free that
 * message with seiza_string_free().
 *
 * Threading: do not free a SeizaRenderedImage while another thread is still
 * reading its borrowed pointers. Do not free the same pointer twice, and do not
 * use a pointer after freeing its owner.
 */

typedef struct SeizaRenderedImage SeizaRenderedImage;
typedef void (*SeizaCatalogSetupProgressCallback)(const char *json, void *context);

/* borrowed: static string, never free */
const char *seiza_core_version(void);

/* returns owned char* (free with seiza_string_free); NULL on failure */
char *seiza_catalog_status_json(
    const char *catalog_directory,
    char **error_out);

bool seiza_catalog_setup(
    const char *catalog_directory,
    uint32_t preset,
    SeizaCatalogSetupProgressCallback progress,
    void *context,
    char **error_out);

/* returns owned SeizaRenderedImage* (free with seiza_rendered_image_free) */
SeizaRenderedImage *seiza_rendered_image_open(
    const char *path,
    double target_median,
    double shadows_clip,
    uint32_t max_dimension,
    char **error_out);

/* returns owned SeizaRenderedImage* (free with seiza_rendered_image_free) */
SeizaRenderedImage *seiza_rendered_image_open_with_rgb_stretch(
    const char *path,
    double target_median,
    double shadows_clip,
    uint32_t max_dimension,
    uint32_t rgb_stretch_mode,
    char **error_out);

/* Renders a FITS image with a parameterized stretch. config_json is a
   serialized seiza-stretch StretchConfig (GHS/MTF/percentile pipeline).
   Returns owned SeizaRenderedImage* (free with seiza_rendered_image_free). */
SeizaRenderedImage *seiza_rendered_image_open_with_stretch_config(
    const char *path,
    const char *config_json,
    uint32_t max_dimension,
    char **error_out);

uint32_t seiza_rendered_image_width(const SeizaRenderedImage *image);
uint32_t seiza_rendered_image_height(const SeizaRenderedImage *image);
/* borrowed: points into `image`; invalid after seiza_rendered_image_free */
const uint8_t *seiza_rendered_image_rgba(const SeizaRenderedImage *image);
size_t seiza_rendered_image_rgba_length(const SeizaRenderedImage *image);
/* borrowed: points into `image`; invalid after seiza_rendered_image_free */
const uint8_t *seiza_rendered_image_bgra(const SeizaRenderedImage *image);
size_t seiza_rendered_image_bgra_length(const SeizaRenderedImage *image);
/* borrowed: points into `image`; invalid after seiza_rendered_image_free */
const char *seiza_rendered_image_metadata_json(const SeizaRenderedImage *image);
/* releases a SeizaRenderedImage; accepts NULL */
void seiza_rendered_image_free(SeizaRenderedImage *image);

/* returns owned char* (free with seiza_string_free); NULL on failure */
char *seiza_solve_image_json(
    const char *path,
    const char *catalog_directory,
    double minimum_scale_arcsec_per_pixel,
    double maximum_scale_arcsec_per_pixel,
    uint8_t sip_order,
    char **error_out);

/* releases any owned char* from this library (JSON returns, error_out); NULL ok */
void seiza_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif
