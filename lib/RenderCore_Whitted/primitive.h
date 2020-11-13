#include "core_settings.h"
#include "ray.h"

#pragma once
class Primitive
{
public:
	explicit Primitive(float4 _origin);
	float4 origin;
	virtual void Intersect(Ray& ray) = 0;
};

