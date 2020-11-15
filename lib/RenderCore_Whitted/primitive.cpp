#include "primitive.h"
#include "core_settings.h"
#include "material.h";

Primitive::Primitive(float4 _origin, Material* _material) {
	origin = _origin;
	material = _material;
}