#include "core_settings.h"
#include "ray.h"
#include "material.h";

#pragma once
class Primitive
{
public:
	explicit Primitive(float4 _origin, Material* _material);
	float4 origin;
	Material* material;
	virtual float Intersect(Ray& ray) = 0;
	virtual float4 GetNormal(float4 point) = 0;
};

