#include "primitive.h"
#include "core_settings.h"
#include "ray.h"

#pragma once
class Sphere : public Primitive
{
public:
	explicit Sphere(float4 _origin, float _radius2);
	float radius2;
	void Intersect(Ray& ray);
};

