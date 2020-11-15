#include "ray.h"
#include "core_settings.h"
#include "limits"

Ray::Ray(float4 _origin, float4 _direction) {
	origin = _origin;
	direction = _direction;
}