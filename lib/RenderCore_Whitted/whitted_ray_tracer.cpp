#include "whitted_ray_tracer.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h"

using namespace lh2core;

void WhittedRayTracer::Render(const ViewPyramid& view, const Bitmap* screen) {
	for (int j = 0; j < screen->height; j++) {
		for (int i = 0; i < screen->width; i++) {
			int index = i + j * screen->width;
			screen->pixels[index] = 255 << 8;
		}
	}
}
