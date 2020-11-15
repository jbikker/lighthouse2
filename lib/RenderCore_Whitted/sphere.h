#include "primitive.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h";

#pragma once
class Sphere : public Primitive
{
public:
	explicit Sphere(float4 _origin, Material* _material, float _radius2);
	float radius2;
	float Intersect(Ray& ray);
};

