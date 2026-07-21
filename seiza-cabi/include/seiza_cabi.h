#ifndef SEIZA_CABI_H
#define SEIZA_CABI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SeizaRenderedImage SeizaRenderedImage;
typedef void (*SeizaCatalogSetupProgressCallback)(const char *json, void *context);

const char *seiza_core_version(void);

char *seiza_catalog_status_json(
    const char *catalog_directory,
    char **error_out);

bool seiza_catalog_setup(
    const char *catalog_directory,
    uint32_t preset,
    SeizaCatalogSetupProgressCallback progress,
    void *context,
    char **error_out);

SeizaRenderedImage *seiza_rendered_image_open(
    const char *path,
    double target_median,
    double shadows_clip,
    uint32_t max_dimension,
    char **error_out);

SeizaRenderedImage *seiza_rendered_image_open_with_rgb_stretch(
    const char *path,
    double target_median,
    double shadows_clip,
    uint32_t max_dimension,
    uint32_t rgb_stretch_mode,
    char **error_out);

/* Renders a FITS image with a parameterized stretch. config_json is a
   serialized seiza-stretch StretchConfig (GHS/MTF/percentile pipeline). */
SeizaRenderedImage *seiza_rendered_image_open_with_stretch_config(
    const char *path,
    const char *config_json,
    uint32_t max_dimension,
    char **error_out);

uint32_t seiza_rendered_image_width(const SeizaRenderedImage *image);
uint32_t seiza_rendered_image_height(const SeizaRenderedImage *image);
const uint8_t *seiza_rendered_image_rgba(const SeizaRenderedImage *image);
size_t seiza_rendered_image_rgba_length(const SeizaRenderedImage *image);
const uint8_t *seiza_rendered_image_bgra(const SeizaRenderedImage *image);
size_t seiza_rendered_image_bgra_length(const SeizaRenderedImage *image);
const char *seiza_rendered_image_metadata_json(const SeizaRenderedImage *image);
void seiza_rendered_image_free(SeizaRenderedImage *image);

char *seiza_solve_image_json(
    const char *path,
    const char *catalog_directory,
    double minimum_scale_arcsec_per_pixel,
    double maximum_scale_arcsec_per_pixel,
    uint8_t sip_order,
    char **error_out);

void seiza_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif
