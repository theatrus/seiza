/* Measure cached pixels. No image files or Seiza-owned cache are required.
 * Build from the repository root after `cargo build -p seiza-cabi`:
 * cc -I seiza-cabi/include seiza-cabi/examples/master_flat_stability.c \
 *    -L target/debug -lseiza_cabi -Wl,-rpath,$PWD/target/debug -o /tmp/flat-stability
 */
#include "seiza_cabi.h"
#include <stdio.h>

int main(void) {
    enum { width = 128, height = 128, length = width * height };
    float master[length], clean[length];
    for (size_t i = 0; i < length; ++i) master[i] = clean[i] = 1.0f;
    for (size_t y = 56; y < 72; ++y)
        for (size_t x = 56; x < 72; ++x) master[y * width + x] = 0.5f;

    const SeizaFlatStabilityImage reference = {
        .data = master, .length = length, .width = width, .height = height, .channels = 1
    };
    const SeizaFlatStabilityImage inputs[] = {
        reference,
        { .data = clean, .length = length, .width = width, .height = height, .channels = 1 }
    };
    char *error = NULL;
    char *result = seiza_master_flat_stability_images_json(
        &reference, inputs, 2, NULL, NULL, "{}", &error
    );
    if (result == NULL) {
        fprintf(stderr, "%s\n", error == NULL ? "measurement failed" : error);
        seiza_string_free(error);
        return 1;
    }
    puts(result); /* owned camelCase JSON; input 0 agrees and input 1 drifts */
    seiza_string_free(result);
    return 0;
}
