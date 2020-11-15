#pragma once
#include "core_settings.h"

class Light
{
public:
	float intensity;
	float4 origin;
	Light(float4 _origin, float _intensity);
};

