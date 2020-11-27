#pragma once
#include "core_settings.h"

class Material
{
public:

	enum class Type {
		Diffuse,
		Mirror,
		Glass
	};

	float4 color;
	Type type;

	Material(Type type, float4 _color);
	Material(Type type);

};

