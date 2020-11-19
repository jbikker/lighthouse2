#include "material.h"
#include "core_settings.h"

Material::Material(Material::Type type, float4 color) {
	this->type = type;
	this->color = color;
}

Material::Material(Material::Type type) {
	this->type = type;
}
