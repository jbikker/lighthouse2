#include "whitted_ray_tracer.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h"

using namespace lh2core;

int* WhittedRayTracer::Render(const ViewPyramid& view, const int height, const int width) {
	int amountPixels = height * width;
	int* pixels = new int[amountPixels];
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			int index = i + j * width;
			pixels[index] = 255 << 8;
		}
	}

	return pixels;
}
