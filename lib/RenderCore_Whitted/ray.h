#pragma once
#include "core_settings.h"
#include "light.h"
#include "tuple"
#include "vector"

class Primitive;
class WhittedRayTracer;

class Ray
{
public:
	Ray(float4 _origin, float4 _direction);
	float4 origin;
	float4 direction;
	float4 GetIntersectionPoint(float intersectionDistance);
	float4 Trace(int recursionDepth = 0);
	tuple<Primitive*, float> GetNearestIntersection();
};
