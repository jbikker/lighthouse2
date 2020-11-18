#pragma once

#include "primitive.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h";

class Plane : public Primitive
{
public:
	explicit Plane(float4 _origin, Material* _material, float4 _normal);
	float Intersect(Ray& ray);
	float4 GetNormal(float4 point);
private:
	float4 normal;
};

