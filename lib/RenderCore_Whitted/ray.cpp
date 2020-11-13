#include "ray.h"
#include "core_settings.h"

Ray::Ray(float4 orig, float4 dir) {
	origin = orig;
	direction = dir;
}