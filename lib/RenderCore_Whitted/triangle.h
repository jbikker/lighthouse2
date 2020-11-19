#pragma once

#include "primitive.h"
#include "core_settings.h"

class Triangle : public Primitive {
public:
	explicit Triangle(float4 _origin, Material* _material, float4 _v0, float4 _v1, float4 _v2);
	float Intersect(Ray& ray);
	float4 GetNormal(float4 point);
private:
	float4 v0;
	float4 v1;
	float4 v2;
	float4 v0v2;
	float4 v0v1;
};

