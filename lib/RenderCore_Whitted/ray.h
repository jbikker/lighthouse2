# include "core_settings.h"

#pragma once
class Ray
{
public:
	Ray(float4 _origin, float4 _direction);
	float4 origin;
	float4 direction;
};
